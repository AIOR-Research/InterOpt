from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# [模块目标]：从脚本所在位置向上定位仓库根目录，让脚本无论从哪个工作目录启动都能找到数据和 .env。
# [输入输出]：无业务输入；返回同时包含 AGENTS.md 与 doc-wiki.md 的目录路径。
def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "PACKAGE_ROOT.marker").exists() or (
            (candidate / "AGENTS.md").exists() and (candidate / "doc-wiki.md").exists()
        ):
            return candidate
    raise RuntimeError("Could not find repo root from evaluation_protocol/run_pipeline.py")


# [模块目标]：读取仓库根目录的本地 .env，但不覆盖调用者已经在系统环境中显式设置的变量。
# [输入输出]：输入 .env 文件路径；输出体现在当前 Python 进程的环境变量中。
def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


# [模块目标]：解析批量实验的默认输出目录。
# [输入输出]：读取环境变量 AIOR_RUNS_ROOT；返回一个目录路径，供 --output_dir 未显式传入时使用。
def resolve_runs_root() -> Path:
    configured = os.getenv("AIOR_RUNS_ROOT")
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    return REPO_ROOT / "runs"


REPO_ROOT = find_repo_root()
load_env_file(REPO_ROOT / ".env")
DEFAULT_TOML_DIRS = (
    Path(os.environ.get("BENCHMARK_CASES_DIR", str(REPO_ROOT / "data"))),
)
DEFAULT_OUTPUT_DIR = resolve_runs_root()


MODEL_PROFILES = {
    "generic_agent": {
        "model_env": "GENERIC_AGENT_MODEL",
        "fallback_env": "OPENAI_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
    "user_simulator": {
        "model_env": "USER_SIMULATOR_MODEL",
        "fallback_env": "GENERIC_AGENT_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
    "detector": {
        "model_env": "DETECTOR_MODEL",
        "fallback_env": "GENERIC_AGENT_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
    "judge": {
        "model_env": "JUDGE_MODEL",
        "fallback_env": "GENERIC_AGENT_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
}


# [模块目标]：把“角色名”解析成 .env 中配置的真实模型名，同时允许命令行直接传模型名做对照实验。
# [输入输出]：输入 generic_agent、detector 等角色名或真实模型名；输出最终发给 API 的模型名。
def resolve_model_name(profile_name: str) -> str:
    profile = MODEL_PROFILES.get(profile_name)
    if profile is None:
        return profile_name
    model = (
        os.getenv(profile["model_env"])
        or os.getenv(profile["fallback_env"])
        or profile["fallback_model"]
    )
    return model.strip()


PRICING_PER_TOKEN = {
    "deepseek-v4-pro": {
        "prompt_cache_hit": 0.000000003625,
        "prompt_cache_miss": 0.000000435,
        "prompt": 0.000000435,
        "completion": 0.00000087,
    },
    "google/gemini-3.1-pro-preview": {"prompt": 0.000002, "completion": 0.000012},
    "openai/gpt-5.5": {"prompt": 0.000005, "completion": 0.00003},
    "anthropic/claude-opus-4.6": {"prompt": 0.000005, "completion": 0.000025},
}


SEVERITY_WEIGHTS = {"P0": 3, "P1": 2, "P2": 1}
HIT_VALUES = {"yes": 1.0, "partial": 0.5, "no": 0.0}

# Agent 协议干预默认关闭；传入 --max_agent_retries 才会真正重采样。
MAX_AGENT_RETRIES_PER_TURN = 0


@dataclass
class ChatResult:
    content: str
    usage: dict[str, int]
    estimated_cost_usd: float


# [模块目标]：统一四个 LLM 角色的调用方式，并累计 token 与估算成本。
# [输入输出]：输入角色/profile 和温度；complete 接收对话消息并返回文本、token 用量和成本。
# [LLM 交互]：所有角色都通过仓库 .env 指定的 DeepSeek-compatible chat/completions 接口调用，模型名由各角色变量决定。
class ChatClient:
    def __init__(self, profile_name: str, temperature: float):
        self.profile_name = profile_name
        self.model = resolve_model_name(profile_name)
        self.temperature = temperature
        self.total_usage: dict[str, int] = {}
        self.total_estimated_cost_usd = 0.0

    def _api_settings(self) -> tuple[str, str]:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Put it in repo root .env or the process environment.")
        return base_url, api_key

    def complete(self, messages: list[dict[str, str]], timeout: int = 180, max_retries: int = 5) -> ChatResult:
        retry = 0
        while True:
            try:
                return self._complete_once(messages, timeout)
            except Exception as exc:
                retry += 1
                if retry > max_retries:
                    raise RuntimeError(f"Maximum retries exceeded for {self.model}: {exc}") from exc
                delay = min(60.0, 1.0 * (2 ** (retry - 1)))
                print(f"Error: {exc}. Retrying in {delay:.2f}s (attempt {retry}/{max_retries})", flush=True)
                time.sleep(delay)

    def _complete_once(self, messages: list[dict[str, str]], timeout: int) -> ChatResult:
        base_url, api_key = self._api_settings()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        content = str(content).strip()
        if not content:
            raise RuntimeError(f"Empty content returned: {data}")
        usage = normalize_usage(data.get("usage") or {})
        cost = estimate_cost(self.model, usage)
        add_usage(self.total_usage, usage)
        self.total_estimated_cost_usd += cost
        return ChatResult(content=content, usage=usage, estimated_cost_usd=cost)


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    prompt_cache_hit_tokens = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    prompt_cache_miss_tokens = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        prompt_cache_hit_tokens += int(details.get("cached_tokens", 0) or 0)
    return {
        "total_calls": 1,
        "failed_calls": 0,
        "total_retries": 0,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
        "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
    }


def add_usage(dst: dict[str, int], src: dict[str, int]) -> None:
    for key in [
        "total_calls",
        "failed_calls",
        "total_retries",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ]:
        dst[key] = dst.get(key, 0) + int(src.get(key, 0) or 0)


def estimate_cost(model_version: str, usage: dict[str, int]) -> float:
    pricing = PRICING_PER_TOKEN.get(model_version)
    if not pricing:
        return 0.0
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)
    if cache_hit or cache_miss:
        return (
            cache_hit * pricing.get("prompt_cache_hit", pricing["prompt"])
            + cache_miss * pricing.get("prompt_cache_miss", pricing["prompt"])
            + usage.get("completion_tokens", 0) * pricing["completion"]
        )
    return usage.get("prompt_tokens", 0) * pricing["prompt"] + usage.get("completion_tokens", 0) * pricing["completion"]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    elif not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def should_stop(agent_text: str) -> bool:
    return agent_text.strip().upper().startswith("READY_TO_MODEL")


def normalize_detector_result(result: dict[str, Any]) -> dict[str, Any]:
    action = str(result.get("action", "")).strip().lower()
    if action not in {"question", "ready_to_model", "invalid"}:
        raise ValueError(f"Unsupported detector action: {action!r}")

    try:
        question_count = int(result.get("question_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Detector question_count must be an integer.") from exc

    atomic_questions = result.get("atomic_questions", [])
    if not isinstance(atomic_questions, list):
        raise ValueError("Detector atomic_questions must be a list.")
    atomic_questions = [str(item).strip() for item in atomic_questions if str(item).strip()]
    is_atomic = bool(result.get("is_atomic", False))

    if action == "question":
        if question_count < 1:
            raise ValueError("A question action must contain at least one question.")
        if len(atomic_questions) != question_count:
            raise ValueError("Detector question_count must match atomic_questions length.")
        if is_atomic != (question_count == 1):
            raise ValueError("Detector is_atomic must be true only for exactly one question.")
    if action == "ready_to_model" and not (question_count == 0 and is_atomic and not atomic_questions):
        raise ValueError("A ready_to_model action must contain zero questions.")
    if action == "invalid" and is_atomic:
        raise ValueError("An invalid action cannot be atomic.")

    return {
        "action": action,
        "question_count": question_count,
        "is_atomic": is_atomic,
        "atomic_questions": atomic_questions,
        "rationale": str(result.get("rationale", "")).strip(),
    }


def build_agent_protocol_feedback(
    detector_result: dict[str, Any] | None = None,
    detector_error: str | None = None,
) -> str:
    lines = [
        "Protocol feedback from the interaction supervisor:",
        "",
        "Your previous response did not pass the single-question interaction protocol.",
    ]
    if detector_error:
        lines.extend(
            [
                f"Detector issue: {detector_error}",
                "Please rewrite your response in the required format.",
            ]
        )
    elif detector_result:
        action = detector_result.get("action", "invalid")
        question_count = detector_result.get("question_count", "unknown")
        atomic_questions = detector_result.get("atomic_questions", []) or []
        rationale = detector_result.get("rationale", "")
        lines.extend(
            [
                f"Detected action: {action}",
                f"Detected independent question count: {question_count}",
            ]
        )
        if atomic_questions:
            lines.append("Detected independent questions:")
            for idx, question in enumerate(atomic_questions, start=1):
                lines.append(f"{idx}. {question}")
        if rationale:
            lines.append(f"Detector rationale: {rationale}")
    lines.extend(
        [
            "",
            "Rewrite the response as exactly one minimal necessary clarification question.",
            "Do not combine multiple business conditions in one question.",
            "Do not ask for implementation details, solver choices, or output formatting.",
            "Do not infer or mention any hidden benchmark labels.",
            "",
            "Allowed output formats:",
            "QUESTION: <one atomic clarification question>",
            "READY_TO_MODEL",
        ]
    )
    return "\n".join(lines)


def normalize_answer_scope_result(result: dict[str, Any]) -> dict[str, Any]:
    in_scope = result.get("in_scope")
    if not isinstance(in_scope, bool):
        raise ValueError("Answer-scope detector in_scope must be a boolean.")
    try:
        violation_count = int(result.get("scope_violation_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Answer-scope detector scope_violation_count must be an integer.") from exc
    violations = result.get("scope_violations", [])
    if not isinstance(violations, list):
        raise ValueError("Answer-scope detector scope_violations must be a list.")
    violations = [str(item).strip() for item in violations if str(item).strip()]
    if violation_count != len(violations):
        raise ValueError("Answer-scope detector count does not match scope_violations.")
    if in_scope and violation_count != 0:
        raise ValueError("An in-scope answer cannot contain scope violations.")
    if not in_scope and violation_count < 1:
        raise ValueError("An out-of-scope answer must identify at least one violation.")
    try:
        independent_fact_count = int(result.get("independent_answered_fact_count", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Answer-scope detector independent_answered_fact_count must be an integer.") from exc
    disclosed_slot_ids = result.get("disclosed_hidden_slot_ids", [])
    if not isinstance(disclosed_slot_ids, list):
        raise ValueError("Answer-scope detector disclosed_hidden_slot_ids must be a list.")
    disclosed_slot_ids = [str(item).strip() for item in disclosed_slot_ids if str(item).strip()]
    try:
        disclosed_slot_count = int(result.get("disclosed_hidden_slot_count", len(disclosed_slot_ids)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Answer-scope detector disclosed_hidden_slot_count must be an integer.") from exc
    if disclosed_slot_count != len(disclosed_slot_ids):
        raise ValueError("Answer-scope detector disclosed slot count does not match disclosed_hidden_slot_ids.")
    return {
        "in_scope": in_scope,
        "scope_violation_count": violation_count,
        "scope_violations": violations,
        "independent_answered_fact_count": independent_fact_count,
        "disclosed_hidden_slot_ids": disclosed_slot_ids,
        "disclosed_hidden_slot_count": disclosed_slot_count,
        "rationale": str(result.get("rationale", "")).strip(),
    }


def build_minimal_agent_retry_feedback(*, detector_error: bool = False) -> str:
    if detector_error:
        return "Your previous output did not follow the required format. Please output exactly one question, or output READY_TO_MODEL."
    return "Your previous question contained multiple information points and did not satisfy the one-question-per-turn requirement. Please ask exactly one minimal necessary clarification question."


# [模块目标]：读取单个 TOML case，并补充运行期需要的来源路径和统一 case_id。
# [输入输出]：输入一个 TOML 文件路径；输出可供 Agent、Simulator 与 Judge 共用的 case 字典。
def load_case(path: Path) -> dict[str, Any]:
    case = tomllib.loads(path.read_text(encoding="utf-8"))
    case["_path"] = str(path)
    case["_case_id"] = case.get("metadata", {}).get("case_id", path.stem)
    return case


def case_sort_key(case: dict[str, Any]) -> str:
    return str(case["_case_id"])


# [模块目标]：把多个 OR-Clarify 案例批次合并成一次实验输入，并在运行前阻止同一 case 被重复计算。
# [输入输出]：输入一个或多个 TOML 目录及可选数量上限；输出按 case_id 排序的 case 列表。
def load_cases(toml_dirs: list[Path], limit: int | None) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for toml_dir in toml_dirs:
        if not toml_dir.is_dir():
            raise FileNotFoundError(f"TOML directory not found: {toml_dir}")
        paths.extend(sorted(toml_dir.glob("*.toml")))
    if not paths:
        raise FileNotFoundError("No TOML files found in the selected directories.")

    cases = [load_case(path) for path in paths]
    case_paths: dict[str, Path] = {}
    duplicate_messages: list[str] = []
    for case in cases:
        case_id = str(case["_case_id"])
        current_path = Path(case["_path"])
        previous_path = case_paths.get(case_id)
        if previous_path is not None:
            duplicate_messages.append(f"{case_id}: {previous_path} | {current_path}")
        else:
            case_paths[case_id] = current_path
    if duplicate_messages:
        details = "\n".join(f"- {message}" for message in duplicate_messages)
        raise ValueError(f"Duplicate case_id values found across TOML directories:\n{details}")

    cases.sort(key=case_sort_key)
    if limit is not None:
        cases = cases[:limit]
    return cases


def render_problem_units(units: list[dict[str, Any]]) -> str:
    lines = []
    for unit in units:
        unit_id = unit.get("id") or unit.get("unit_id") or ""
        kind = unit.get("kind") or unit.get("role") or ""
        lines.append(f"- {unit_id} ({kind}): {unit.get('content', '')}")
    return "\n".join(lines)


def render_private_business_facts(slots: list[dict[str, Any]]) -> str:
    lines = []
    for slot in slots:
        answer = str(slot.get("simulator_answer", "")).strip()
        if answer:
            lines.append(f"- {answer}")
    return "\n".join(lines)


def render_hidden_slots_for_answer_audit(slots: list[dict[str, Any]]) -> str:
    lines = []
    for slot in slots:
        answer = str(slot.get("simulator_answer", "")).strip()
        lines.append(f"- {slot.get('slot_id')}: {slot.get('name')}")
        if answer:
            lines.append(f"  simulator_answer: {answer}")
    return "\n".join(lines)


def render_user_simulator_case(case: dict[str, Any]) -> str:
    sim = case.get("simulator") or {
        "business_role": "business stakeholder providing the original optimization problem",
    }
    initial = case["initial_brief"]
    chunks = [
        "## Your business role",
        f"Business role: {sim.get('business_role', '')}",
        "",
        "## Original request",
        initial["content"],
        "",
        "## Private business context",
        render_problem_units(case.get("problem_units", [])),
        "",
        "## Private business facts",
        render_private_business_facts(case.get("hidden_slots", [])),
    ]
    return "\n".join(chunks)


def render_hidden_slots_for_judge(slots: list[dict[str, Any]]) -> str:
    lines = []
    for slot in slots:
        lines.append(f"## {slot.get('slot_id')}: {slot.get('name')}")
        lines.append(f"- Severity: {slot.get('severity')}")
        lines.append(f"- Severity reason: {slot.get('severity_reason')}")
        lines.append(f"- Problem unit ID: {slot.get('problem_unit_id')}")
        lines.append(f"- Semantic hit rule: {slot.get('semantic_hit_rule')}")
        lines.append("- Reference acceptable questions:")
        for question in slot.get("reference_acceptable_questions", []):
            lines.append(f"  - {question}")
        lines.append("- Failure modes:")
        for failure in slot.get("failure_modes", []):
            lines.append(f"  - {failure}")
        lines.append("")
    return "\n".join(lines)


def render_judge_case(case: dict[str, Any]) -> str:
    initial = case["initial_brief"]
    chunks = [
        "# Case facts for Judge",
        "",
        "## Initial brief shown to Generic Agent",
        f"Visible unit IDs: {', '.join(initial.get('visible_unit_ids', []))}",
        initial["content"],
        "",
        "## Problem units",
        render_problem_units(case.get("problem_units", [])),
        "",
        "## Hidden slot scoring rules",
        render_hidden_slots_for_judge(case.get("hidden_slots", [])),
    ]
    return "\n".join(chunks)


def render_transcript(rows: list[dict[str, Any]]) -> str:
    chunks = []
    for row in rows:
        speaker = str(row["speaker"]).replace("_", " ").title()
        chunks.append(f"## Turn {row['turn']} - {speaker}")
        chunks.append("")
        chunks.append(str(row["content"]).strip())
        chunks.append("")
    return "\n".join(chunks).strip()


def calculate_weighted_slot_score(slot_scores: list[dict[str, Any]]) -> dict[str, Any]:
    earned_weight = 0.0
    total_weight = 0.0
    counts = {"yes": 0, "partial": 0, "no": 0, "other": 0}
    severity_totals: dict[str, dict[str, float]] = {}
    for sc in slot_scores:
        severity = str(sc.get("severity", ""))
        hit = str(sc.get("hit", ""))
        weight = SEVERITY_WEIGHTS.get(severity, 0)
        value = HIT_VALUES.get(hit, 0.0)
        earned_weight += weight * value
        total_weight += weight
        counts[hit if hit in counts else "other"] += 1
        sev = severity_totals.setdefault(severity, {"earned_weight": 0.0, "total_weight": 0.0})
        sev["earned_weight"] += weight * value
        sev["total_weight"] += weight
    for sev in severity_totals.values():
        total = sev["total_weight"]
        sev["score"] = sev["earned_weight"] / total if total else None
    return {
        "weighted_slot_score": earned_weight / total_weight if total_weight else None,
        "earned_weight": earned_weight,
        "total_weight": total_weight,
        "hit_counts": counts,
        "severity_scores": severity_totals,
    }


def calculate_restoration_summary(slot_scores: list[dict[str, Any]]) -> dict[str, Any]:
    core_slots = [
        score
        for score in slot_scores
        if score.get("severity") in {"P0", "P1"}
    ]
    p2_slots = [
        score
        for score in slot_scores
        if score.get("severity") == "P2"
    ]
    all_slots = list(slot_scores)

    def is_exact(scores: list[dict[str, Any]]) -> bool:
        return bool(scores) and all(score.get("hit") == "yes" for score in scores)

    return {
        "core_exact_restore": is_exact(core_slots),
        "core_slot_count": len(core_slots),
        "core_unresolved_slots": [
            str(score.get("slot_id", ""))
            for score in core_slots
            if score.get("hit") != "yes"
        ],
        "p2_slot_count": len(p2_slots),
        "p2_unresolved_slots": [
            str(score.get("slot_id", ""))
            for score in p2_slots
            if score.get("hit") != "yes"
        ],
        "all_slot_exact_restore": is_exact(all_slots),
        "all_slot_count": len(all_slots),
        "rule": "Exact restoration uses only P0/P1 slots; unresolved P2 slots do not affect exact restoration.",
    }


def audit_stopping_behavior(
    slot_scores: list[dict[str, Any]],
    stopping_behavior: dict[str, Any],
    completed_ready_to_model: bool,
) -> dict[str, Any]:
    unresolved_p0 = [
        str(score.get("slot_id", ""))
        for score in slot_scores
        if score.get("severity") == "P0" and score.get("hit") != "yes"
    ]
    unresolved_p1 = [
        str(score.get("slot_id", ""))
        for score in slot_scores
        if score.get("severity") == "P1" and score.get("hit") != "yes"
    ]
    judge_status = stopping_behavior.get("status")
    if not completed_ready_to_model:
        rule_based_status = "no_stop"
    elif unresolved_p0 or unresolved_p1:
        rule_based_status = "premature_stop"
    elif judge_status == "over_questioning":
        rule_based_status = "over_questioning"
    else:
        rule_based_status = "appropriate_stop"
    return {
        "judge_status": judge_status,
        "rule_based_status": rule_based_status,
        "status_mismatch": judge_status != rule_based_status,
        "unresolved_p0_slots_from_scores": unresolved_p0,
        "unresolved_p1_slots_from_scores": unresolved_p1,
        "rule": "Any non-yes P0/P1 slot at READY_TO_MODEL is unresolved.",
    }


def load_prompt_files(prompts_dir: Path) -> tuple[str, str, str, str, str]:
    return (
        (prompts_dir / "generic_agent_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "question_detector_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "answer_scope_detector_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "user_simulator_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "judge_prompt.md").read_text(encoding="utf-8").strip(),
    )


# [模块目标]：完成一个 case 的 Agent 访谈、Detector 审计、User 模拟和 Judge 评分，并保存全部证据文件。
# [输入输出]：输入 case、四角色配置和实验参数；输出该次运行的统计字典，同时把 transcript 与评分写入 run 目录。
# [LLM 交互]：Agent 负责提问，Question Detector 默认只计数；User Simulator 作答；Answer-Scope Detector 按需审计；Judge 最后评分。
def run_interaction(
    case: dict[str, Any],
    agent_profile: str,
    detector_profile: str,
    user_profile: str,
    judge_profile: str,
    run_index: int,
    output_root: Path,
    prompts: tuple[str, str, str, str, str],
    agent_temperature: float,
    detector_temperature: float,
    user_temperature: float,
    judge_temperature: float,
    max_turns: int,
    monitor_user_answers: bool = False,
    detector_feedback_mode: str = "none",
    retry_limit_behavior: str = "pass_through",
    pipeline_mode: str | None = None,
) -> dict[str, Any]:
    case_id = case["_case_id"]
    run_dir = output_root / agent_profile / f"run_{run_index:02d}" / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "statistics.json"
    judge_path = run_dir / "judge_result.json"
    if stats_path.exists() and judge_path.exists():
        stat = json.loads(stats_path.read_text(encoding="utf-8"))
        judge_result = json.loads(judge_path.read_text(encoding="utf-8"))
        stopping_audit = audit_stopping_behavior(
            judge_result.get("slot_scores", []),
            judge_result.get("stopping_behavior") or {},
            bool(stat.get("completed_ready_to_model")),
        )
        judge_result["stopping_consistency_audit"] = stopping_audit
        stat["rule_based_stopping_status"] = stopping_audit["rule_based_status"]
        stat["stopping_status_mismatch"] = stopping_audit["status_mismatch"]
        judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
        stats_path.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
        return stat

    generic_prompt, question_detector_prompt, answer_detector_prompt, simulator_prompt_base, judge_prompt = prompts
    agent_client = ChatClient(agent_profile, agent_temperature)
    detector_client = ChatClient(detector_profile, detector_temperature)
    user_client = ChatClient(user_profile, user_temperature)
    judge_client = ChatClient(judge_profile, judge_temperature)

    first_user_message = (
        "Here is the user's initial request. Interview the user if clarification is needed.\n\n"
        + case["initial_brief"]["content"]
    )
    simulator_system_prompt = simulator_prompt_base + "\n\n" + render_user_simulator_case(case)

    agent_messages = [
        {"role": "system", "content": generic_prompt},
        {"role": "user", "content": first_user_message},
    ]
    simulator_messages = [
        {"role": "system", "content": simulator_system_prompt},
    ]
    if detector_feedback_mode not in {"visible", "minimal", "none"}:
        raise ValueError(f"Unsupported detector_feedback_mode: {detector_feedback_mode}")
    if retry_limit_behavior not in {"pass_through", "protocol_failed"}:
        raise ValueError(f"Unsupported retry_limit_behavior: {retry_limit_behavior}")
    if pipeline_mode is None:
        intervention_enabled = (
            MAX_AGENT_RETRIES_PER_TURN > 0
            or detector_feedback_mode != "none"
            or retry_limit_behavior == "protocol_failed"
        )
        pipeline_mode = "agent_protocol_intervention" if intervention_enabled else "passive_question_audit"
        if monitor_user_answers:
            pipeline_mode += "_with_answer_audit"

    transcript: list[dict[str, Any]] = []
    detector_events: list[dict[str, Any]] = []
    answer_scope_events: list[dict[str, Any]] = []
    completed = False
    protocol_failed = False
    protocol_failure_type = ""
    protocol_failure_reason = ""
    final_agent_text = ""
    detector_rejection_count = 0
    detector_error_count = 0
    answer_scope_rejection_count = 0
    answer_scope_error_count = 0
    agent_question_turn_count = 0
    agent_atomic_question_count = 0
    agent_multi_question_turn_count = 0
    answer_disclosed_hidden_slot_count = 0
    answer_multi_hidden_slot_disclosure_count = 0
    answer_independent_fact_count = 0

    # 新增重试统计
    agent_retry_total = 0
    agent_max_retries_exceeded = False
    agent_retry_limit_pass_through_count = 0

    for turn in range(1, max_turns + 1):
        # ----- Agent 重试循环 -----
        agent_retry = 0
        agent_retry_feedback: str | None = None
        while True:
            agent_call_messages = agent_messages
            if detector_feedback_mode == "minimal" and agent_retry_feedback:
                agent_call_messages = agent_messages + [{"role": "user", "content": agent_retry_feedback}]
            agent_reply = agent_client.complete(agent_call_messages).content
            final_agent_text = agent_reply

            detector_user_message = "Assistant latest response:\n\n" + agent_reply
            detector_messages = [
                {"role": "system", "content": question_detector_prompt},
                {"role": "user", "content": detector_user_message},
            ]
            detector_raw = detector_client.complete(detector_messages).content
            event: dict[str, Any] = {
                "turn": turn,
                "agent_response": agent_reply,
                "detector_raw": detector_raw,
                "retry_count": agent_retry,
            }
            try:
                detector_result = normalize_detector_result(extract_json_object(detector_raw))
                event["detector_result"] = detector_result
            except Exception as exc:
                detector_error_count += 1
                event["detector_error"] = str(exc)
                event["accepted"] = False
                event["entered_transcript"] = False
                event["feedback_mode"] = detector_feedback_mode
                event["protocol_violation_type"] = "agent_detector_parse_error"
                event["detected_action"] = "parse_error"
                event["detected_question_count"] = None
                event["detected_atomic_questions"] = []
                detector_events.append(event)
                agent_retry += 1
                if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                    event["retry_limit_exceeded"] = True
                    if retry_limit_behavior == "pass_through":
                        agent_retry_limit_pass_through_count += 1
                        event["pass_through_after_retry_limit"] = True
                        event["entered_transcript"] = True
                        transcript.append(
                            {
                                "turn": turn,
                                "speaker": "generic_agent",
                                "content": agent_reply,
                                "detector_pass_through_after_retry_limit": True,
                                "detector_pass_through_reason": "detector_parse_error",
                            }
                        )
                        agent_messages.append({"role": "assistant", "content": agent_reply})
                    else:
                        agent_max_retries_exceeded = True
                    break
                if detector_feedback_mode == "visible":
                    agent_messages.append({"role": "assistant", "content": agent_reply})
                    agent_messages.append(
                        {
                            "role": "user",
                            "content": build_agent_protocol_feedback(detector_error=str(exc)),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    agent_retry_feedback = build_minimal_agent_retry_feedback(detector_error=True)
                agent_retry_total += 1
                continue

            detector_events.append(event)
            action = detector_result["action"]

            if action == "ready_to_model":
                event["accepted"] = True
                event["entered_transcript"] = True
                transcript.append({"turn": turn, "speaker": "generic_agent", "content": agent_reply})
                completed = True
                break  # 跳出 agent 重试循环

            if action == "question":
                # 单问与多问都进入正式 transcript；Detector 只负责记录原子问题数量。
                event["accepted"] = True
                event["entered_transcript"] = True
                agent_question_turn_count += 1
                agent_atomic_question_count += int(detector_result["question_count"])
                if int(detector_result["question_count"]) > 1:
                    agent_multi_question_turn_count += 1
                transcript.append(
                    {
                        "turn": turn,
                        "speaker": "generic_agent",
                        "content": agent_reply,
                        "detected_question_count": detector_result["question_count"],
                        "detected_atomic_questions": detector_result["atomic_questions"],
                    }
                )
                agent_messages.append({"role": "assistant", "content": agent_reply})
                break  # 跳出 agent 重试循环

            # 其他情况：invalid（零问题非 ready，或 READY_TO_MODEL 与问题混写）。
            detector_rejection_count += 1
            agent_retry += 1
            event["accepted"] = False
            event["entered_transcript"] = False
            event["feedback_mode"] = detector_feedback_mode
            event["protocol_violation_type"] = "agent_question_protocol"
            event["detected_action"] = action
            event["detected_question_count"] = detector_result.get("question_count")
            event["detected_atomic_questions"] = detector_result.get("atomic_questions", [])
            event["detector_rationale"] = detector_result.get("rationale", "")
            if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                event["retry_limit_exceeded"] = True
                if retry_limit_behavior == "pass_through":
                    agent_retry_limit_pass_through_count += 1
                    event["pass_through_after_retry_limit"] = True
                    event["entered_transcript"] = True
                    transcript.append(
                        {
                            "turn": turn,
                            "speaker": "generic_agent",
                            "content": agent_reply,
                            "detector_pass_through_after_retry_limit": True,
                            "detector_pass_through_reason": action,
                        }
                        )
                    agent_messages.append({"role": "assistant", "content": agent_reply})
                else:
                    agent_max_retries_exceeded = True
                break
            if detector_feedback_mode == "visible":
                agent_messages.append({"role": "assistant", "content": agent_reply})
                agent_messages.append(
                    {
                        "role": "user",
                        "content": build_agent_protocol_feedback(detector_result=detector_result),
                    }
                )
            elif detector_feedback_mode == "minimal":
                agent_retry_feedback = build_minimal_agent_retry_feedback()
            agent_retry_total += 1
            # 继续重试，不加入历史

        if agent_max_retries_exceeded:
            protocol_failed = True
            protocol_failure_type = "agent_max_retries_exceeded"
            protocol_failure_reason = f"Exceeded {MAX_AGENT_RETRIES_PER_TURN} retries in turn {turn}"
            break

        if completed:
            break  # READY_TO_MODEL 结束

        # ----- User response loop -----
        simulator_messages.append({"role": "user", "content": agent_reply})
        if not monitor_user_answers:
            simulator_reply = user_client.complete(simulator_messages).content
            simulator_messages.append({"role": "assistant", "content": simulator_reply})
            transcript.append({"turn": turn, "speaker": "user_simulator", "content": simulator_reply})
            agent_messages.append({"role": "user", "content": "Business user response:\n\n" + simulator_reply})
            time.sleep(0.2)
            continue

        simulator_reply = user_client.complete(simulator_messages).content

        answer_detector_user_message = (
            "# Current question\n\n"
            + agent_reply
            + "\n\n# Business user response\n\n"
            + simulator_reply
            + "\n\n# Hidden slots for passive audit\n\n"
            + render_hidden_slots_for_answer_audit(case.get("hidden_slots", []))
        )
        answer_detector_messages = [
            {"role": "system", "content": answer_detector_prompt},
            {"role": "user", "content": answer_detector_user_message},
        ]
        answer_detector_raw = detector_client.complete(answer_detector_messages).content
        answer_event: dict[str, Any] = {
            "turn": turn,
            "question": agent_reply,
            "business_user_response": simulator_reply,
            "detector_raw": answer_detector_raw,
            "retry_count": 0,
            "accepted": True,
            "entered_transcript": True,
            "feedback_mode": detector_feedback_mode,
            "passive_audit_only": True,
        }
        try:
            answer_result = normalize_answer_scope_result(extract_json_object(answer_detector_raw))
            answer_event["detector_result"] = answer_result
            answer_event["scope_violation_count"] = answer_result.get("scope_violation_count")
            answer_event["scope_violations"] = answer_result.get("scope_violations", [])
            answer_event["independent_answered_fact_count"] = answer_result.get("independent_answered_fact_count")
            answer_event["disclosed_hidden_slot_ids"] = answer_result.get("disclosed_hidden_slot_ids", [])
            answer_event["disclosed_hidden_slot_count"] = answer_result.get("disclosed_hidden_slot_count")
            answer_event["detector_rationale"] = answer_result.get("rationale", "")
            answer_independent_fact_count += int(answer_result.get("independent_answered_fact_count") or 0)
            disclosed_count = int(answer_result.get("disclosed_hidden_slot_count") or 0)
            answer_disclosed_hidden_slot_count += disclosed_count
            if disclosed_count > 1:
                answer_multi_hidden_slot_disclosure_count += 1
            if not answer_result["in_scope"]:
                answer_scope_rejection_count += 1
        except Exception as exc:
            answer_scope_error_count += 1
            answer_event["detector_error"] = str(exc)
            answer_event["scope_violation_count"] = None
            answer_event["scope_violations"] = []
            answer_event["disclosed_hidden_slot_ids"] = []
            answer_event["disclosed_hidden_slot_count"] = None
        answer_scope_events.append(answer_event)

        simulator_messages.append({"role": "assistant", "content": simulator_reply})
        transcript.append({"turn": turn, "speaker": "user_simulator", "content": simulator_reply})
        agent_messages.append({"role": "user", "content": "Business user response:\n\n" + simulator_reply})

        time.sleep(0.2)

    # ----- Judge 与统计（同原逻辑，增加重试字段）-----
    judge_user_message = render_judge_case(case) + "\n\n# Full Transcript\n\n" + render_transcript(transcript)
    judge_messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": judge_user_message},
    ]
    judge_raw = ""
    judge_result: dict[str, Any] | None = None
    judge_parse_errors: list[str] = []
    for judge_attempt in range(1, 4):
        judge_raw = judge_client.complete(judge_messages, timeout=240).content
        try:
            judge_result = extract_json_object(judge_raw)
            break
        except Exception as exc:
            judge_parse_errors.append(f"attempt {judge_attempt}: {exc}")
            if judge_attempt >= 3:
                raise
            judge_messages = judge_messages + [
                {"role": "assistant", "content": judge_raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous judge output was not valid JSON. "
                        "Return only one valid JSON object that follows the required schema. "
                        "Do not include markdown fences or unescaped line breaks inside JSON strings."
                    ),
                },
            ]
    assert judge_result is not None
    if judge_parse_errors:
        judge_result["judge_parse_errors_before_success"] = judge_parse_errors
    weighted_summary = calculate_weighted_slot_score(judge_result.get("slot_scores", []))
    judge_result["weighted_summary"] = weighted_summary
    restoration_summary = calculate_restoration_summary(judge_result.get("slot_scores", []))
    judge_result["restoration_summary"] = restoration_summary
    stopping_audit = audit_stopping_behavior(
        judge_result.get("slot_scores", []),
        judge_result.get("stopping_behavior") or {},
        completed,
    )
    judge_result["stopping_consistency_audit"] = stopping_audit

    (run_dir / "initial_brief.txt").write_text(case["initial_brief"]["content"], encoding="utf-8")
    (run_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (run_dir / "detector_events.json").write_text(
        json.dumps(detector_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "answer_scope_events.json").write_text(
        json.dumps(answer_scope_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "judge_prompt_user_message.md").write_text(judge_user_message, encoding="utf-8")
    (run_dir / "judge_raw.txt").write_text(judge_raw, encoding="utf-8")
    judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")

    stat = {
        "mode": pipeline_mode,
        "pipeline_mode": pipeline_mode,
        "monitor_user_answers": monitor_user_answers,
        "detector_feedback_mode": detector_feedback_mode,
        "retry_limit_behavior": retry_limit_behavior,
        "detector_rationale_visible": detector_feedback_mode == "visible",
        "detector_feedback_in_context": detector_feedback_mode == "visible",
        "rejected_attempt_in_transcript": False,
        "case_id": case_id,
        "agent_profile": agent_profile,
        "agent_model": agent_client.model,
        "detector_profile": detector_profile,
        "detector_model": detector_client.model,
        "user_profile": user_profile,
        "user_model": user_client.model,
        "judge_profile": judge_profile,
        "judge_model": judge_client.model,
        "run_index": run_index,
        "turn_count": max((int(row["turn"]) for row in transcript), default=0),
        "attempted_turn_count": turn,
        "transcript_rows": len(transcript),
        "completed_ready_to_model": completed,
        "protocol_failed": protocol_failed,
        "protocol_failure_type": protocol_failure_type,
        "protocol_failure_reason": protocol_failure_reason,
        "protocol_failure_turn": turn if protocol_failed else None,
        "detector_call_count": len(detector_events),
        "detector_rejection_count": detector_rejection_count,
        "detector_error_count": detector_error_count,
        "agent_question_turn_count": agent_question_turn_count,
        "agent_atomic_question_count": agent_atomic_question_count,
        "agent_multi_question_turn_count": agent_multi_question_turn_count,
        "answer_scope_detector_call_count": len(answer_scope_events),
        "answer_scope_rejection_count": answer_scope_rejection_count,
        "answer_scope_violation_count": answer_scope_rejection_count,
        "answer_scope_error_count": answer_scope_error_count,
        "answer_independent_fact_count": answer_independent_fact_count,
        "answer_disclosed_hidden_slot_count": answer_disclosed_hidden_slot_count,
        "answer_multi_hidden_slot_disclosure_count": answer_multi_hidden_slot_disclosure_count,
        "final_agent_text": final_agent_text,
        "weighted_slot_score": weighted_summary["weighted_slot_score"],
        "earned_weight": weighted_summary["earned_weight"],
        "total_weight": weighted_summary["total_weight"],
        "core_exact_restore": restoration_summary["core_exact_restore"],
        "core_slot_count": restoration_summary["core_slot_count"],
        "core_unresolved_slot_count": len(restoration_summary["core_unresolved_slots"]),
        "p2_slot_count": restoration_summary["p2_slot_count"],
        "p2_unresolved_slot_count": len(restoration_summary["p2_unresolved_slots"]),
        "all_slot_exact_restore": restoration_summary["all_slot_exact_restore"],
        "all_slot_count": restoration_summary["all_slot_count"],
        "stopping_status": (judge_result.get("stopping_behavior") or {}).get("status"),
        "rule_based_stopping_status": stopping_audit["rule_based_status"],
        "stopping_status_mismatch": stopping_audit["status_mismatch"],
        "silent_assumption_count": len(judge_result.get("silent_assumptions", []) or []),
        "agent_usage": agent_client.total_usage,
        "detector_usage": detector_client.total_usage,
        "user_usage": user_client.total_usage,
        "judge_usage": judge_client.total_usage,
        "agent_estimated_cost_usd": agent_client.total_estimated_cost_usd,
        "detector_estimated_cost_usd": detector_client.total_estimated_cost_usd,
        "user_estimated_cost_usd": user_client.total_estimated_cost_usd,
        "judge_estimated_cost_usd": judge_client.total_estimated_cost_usd,
        # 新增重试统计
        "agent_retry_total": agent_retry_total,
        "agent_max_retries_exceeded": agent_max_retries_exceeded,
        "agent_retry_limit_pass_through_count": agent_retry_limit_pass_through_count,
        "run_dir": str(run_dir),
    }
    stat["estimated_cost_usd"] = (
        stat["agent_estimated_cost_usd"]
        + stat["detector_estimated_cost_usd"]
        + stat["user_estimated_cost_usd"]
        + stat["judge_estimated_cost_usd"]
    )
    stats_path.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    return stat


def summarize(stats: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for stat in stats:
        profile = stat["agent_profile"]
        group = groups.setdefault(
            profile,
            {
                "agent_model": stat["agent_model"],
                "runs": 0,
                "weighted_slot_score_sum": 0.0,
                "weighted_slot_score_count": 0,
                "earned_weight_sum": 0.0,
                "total_weight_sum": 0.0,
                "turn_sum": 0,
                "attempted_turn_sum": 0,
                "ready_to_model_sum": 0,
                "protocol_failed_sum": 0,
                "protocol_failure_type_counts": {},
                "detector_call_sum": 0,
                "detector_rejection_sum": 0,
                "detector_error_sum": 0,
                "agent_question_turn_sum": 0,
                "agent_atomic_question_sum": 0,
                "agent_multi_question_turn_sum": 0,
                "answer_scope_detector_call_sum": 0,
                "answer_scope_rejection_sum": 0,
                "answer_scope_error_sum": 0,
                "answer_independent_fact_sum": 0,
                "answer_disclosed_hidden_slot_sum": 0,
                "answer_multi_hidden_slot_disclosure_sum": 0,
                "silent_assumption_sum": 0,
                "stopping_status_counts": {},
                "rule_based_stopping_status_counts": {},
                "stopping_status_mismatch_sum": 0,
                "agent_usage": {},
                "detector_usage": {},
                "user_usage": {},
                "judge_usage": {},
                "estimated_cost_usd": 0.0,
                "core_exact_restore_sum": 0,
                "core_eligible_run_count": 0,
                "all_slot_exact_restore_sum": 0,
                "core_slot_sum": 0,
                "p2_slot_sum": 0,
                "core_unresolved_slot_sum": 0,
                "p2_unresolved_slot_sum": 0,
                # 新增重试聚合
                "agent_retry_total_sum": 0,
                "agent_max_retries_exceeded_sum": 0,
                "agent_retry_limit_pass_through_sum": 0,
            },
        )
        group["runs"] += 1
        score = stat.get("weighted_slot_score")
        if score is not None:
            group["weighted_slot_score_sum"] += float(score)
            group["weighted_slot_score_count"] += 1
        group["earned_weight_sum"] += float(stat.get("earned_weight", 0) or 0)
        group["total_weight_sum"] += float(stat.get("total_weight", 0) or 0)
        group["turn_sum"] += int(stat.get("turn_count", 0) or 0)
        group["attempted_turn_sum"] += int(stat.get("attempted_turn_count", 0) or 0)
        group["ready_to_model_sum"] += 1 if stat.get("completed_ready_to_model") else 0
        group["protocol_failed_sum"] += 1 if stat.get("protocol_failed") else 0
        failure_type = stat.get("protocol_failure_type") or "none"
        group["protocol_failure_type_counts"][failure_type] = (
            group["protocol_failure_type_counts"].get(failure_type, 0) + 1
        )
        group["detector_call_sum"] += int(stat.get("detector_call_count", 0) or 0)
        group["detector_rejection_sum"] += int(stat.get("detector_rejection_count", 0) or 0)
        group["detector_error_sum"] += int(stat.get("detector_error_count", 0) or 0)
        group["agent_question_turn_sum"] += int(stat.get("agent_question_turn_count", 0) or 0)
        group["agent_atomic_question_sum"] += int(stat.get("agent_atomic_question_count", 0) or 0)
        group["agent_multi_question_turn_sum"] += int(stat.get("agent_multi_question_turn_count", 0) or 0)
        group["answer_scope_detector_call_sum"] += int(stat.get("answer_scope_detector_call_count", 0) or 0)
        group["answer_scope_rejection_sum"] += int(stat.get("answer_scope_rejection_count", 0) or 0)
        group["answer_scope_error_sum"] += int(stat.get("answer_scope_error_count", 0) or 0)
        group["answer_independent_fact_sum"] += int(stat.get("answer_independent_fact_count", 0) or 0)
        group["answer_disclosed_hidden_slot_sum"] += int(stat.get("answer_disclosed_hidden_slot_count", 0) or 0)
        group["answer_multi_hidden_slot_disclosure_sum"] += int(stat.get("answer_multi_hidden_slot_disclosure_count", 0) or 0)
        group["silent_assumption_sum"] += int(stat.get("silent_assumption_count", 0) or 0)
        status = stat.get("stopping_status") or "unknown"
        group["stopping_status_counts"][status] = group["stopping_status_counts"].get(status, 0) + 1
        rule_status = stat.get("rule_based_stopping_status") or status
        group["rule_based_stopping_status_counts"][rule_status] = (
            group["rule_based_stopping_status_counts"].get(rule_status, 0) + 1
        )
        group["stopping_status_mismatch_sum"] += 1 if stat.get("stopping_status_mismatch") else 0
        add_usage(group["agent_usage"], stat.get("agent_usage", {}))
        add_usage(group["detector_usage"], stat.get("detector_usage", {}))
        add_usage(group["user_usage"], stat.get("user_usage", {}))
        add_usage(group["judge_usage"], stat.get("judge_usage", {}))
        group["estimated_cost_usd"] += float(stat.get("estimated_cost_usd", 0) or 0)
        if int(stat.get("core_slot_count", 0) or 0) > 0:
            group["core_eligible_run_count"] += 1
        group["core_exact_restore_sum"] += 1 if stat.get("core_exact_restore") else 0
        group["all_slot_exact_restore_sum"] += 1 if stat.get("all_slot_exact_restore") else 0
        group["core_slot_sum"] += int(stat.get("core_slot_count", 0) or 0)
        group["p2_slot_sum"] += int(stat.get("p2_slot_count", 0) or 0)
        group["core_unresolved_slot_sum"] += int(stat.get("core_unresolved_slot_count", 0) or 0)
        group["p2_unresolved_slot_sum"] += int(stat.get("p2_unresolved_slot_count", 0) or 0)

        # 重试聚合
        group["agent_retry_total_sum"] += int(stat.get("agent_retry_total", 0))
        group["agent_max_retries_exceeded_sum"] += 1 if stat.get("agent_max_retries_exceeded") else 0
        group["agent_retry_limit_pass_through_sum"] += int(stat.get("agent_retry_limit_pass_through_count", 0) or 0)

    summary = {"profiles": {}, "total_estimated_cost_usd": 0.0}
    for profile, group in groups.items():
        runs = max(1, group["runs"])
        core_eligible_runs = group["core_eligible_run_count"]
        score_count = max(1, group["weighted_slot_score_count"])
        weighted_micro = (
            group["earned_weight_sum"] / group["total_weight_sum"] if group["total_weight_sum"] else None
        )
        question_turns = max(1, group["agent_question_turn_sum"])
        summary["profiles"][profile] = {
            "agent_model": group["agent_model"],
            "runs": group["runs"],
            "metrics": {
                "WeightedSlotScore_macro": group["weighted_slot_score_sum"] / score_count,
                "WeightedSlotScore_micro": weighted_micro,
                "AverageTurns": group["turn_sum"] / runs,
                "AverageAttemptedTurns": group["attempted_turn_sum"] / runs,
                "ReadyToModelRate": group["ready_to_model_sum"] / runs,
                "CoreExactRestoreRate": (
                    group["core_exact_restore_sum"] / core_eligible_runs
                    if core_eligible_runs
                    else None
                ),
                "CoreExactRestoreCount": group["core_exact_restore_sum"],
                "CoreEligibleRunCount": core_eligible_runs,
                "AllSlotExactRestoreRate": group["all_slot_exact_restore_sum"] / runs,
                "AllSlotExactRestoreCount": group["all_slot_exact_restore_sum"],
                "CoreSlotsPerRun": group["core_slot_sum"] / runs,
                "P2SlotsPerRun": group["p2_slot_sum"] / runs,
                "CoreUnresolvedSlotsPerRun": group["core_unresolved_slot_sum"] / runs,
                "P2UnresolvedSlotsPerRun": group["p2_unresolved_slot_sum"] / runs,
                "ProtocolFailureRate": group["protocol_failed_sum"] / runs,
                "ProtocolFailureTypeCounts": group["protocol_failure_type_counts"],
                "DetectorCallsPerRun": group["detector_call_sum"] / runs,
                "DetectorRejectionsPerRun": group["detector_rejection_sum"] / runs,
                "DetectorErrorsPerRun": group["detector_error_sum"] / runs,
                "AgentQuestionTurnsPerRun": group["agent_question_turn_sum"] / runs,
                "AtomicQuestionsPerRun": group["agent_atomic_question_sum"] / runs,
                "MultiQuestionTurnsPerRun": group["agent_multi_question_turn_sum"] / runs,
                "MultiQuestionTurnRate": group["agent_multi_question_turn_sum"] / question_turns,
                "AvgQuestionsPerQuestionTurn": group["agent_atomic_question_sum"] / question_turns,
                "AnswerScopeDetectorCallsPerRun": group["answer_scope_detector_call_sum"] / runs,
                "AnswerScopeViolationsPerRun": group["answer_scope_rejection_sum"] / runs,
                "AnswerScopeDetectorErrorsPerRun": group["answer_scope_error_sum"] / runs,
                "AnswerIndependentFactsPerRun": group["answer_independent_fact_sum"] / runs,
                "AnswerDisclosedHiddenSlotsPerRun": group["answer_disclosed_hidden_slot_sum"] / runs,
                "AnswerMultiHiddenSlotDisclosurePerRun": group["answer_multi_hidden_slot_disclosure_sum"] / runs,
                "SilentAssumptionsPerRun": group["silent_assumption_sum"] / runs,
                "StoppingStatusCounts": group["stopping_status_counts"],
                "RuleBasedStoppingStatusCounts": group["rule_based_stopping_status_counts"],
                "StoppingStatusMismatchRate": group["stopping_status_mismatch_sum"] / runs,
                # 新增重试指标
                "AvgAgentRetriesPerRun": group["agent_retry_total_sum"] / runs,
                "AgentMaxRetriesExceededRate": group["agent_max_retries_exceeded_sum"] / runs,
            },
            "agent_usage": group["agent_usage"],
            "detector_usage": group["detector_usage"],
            "user_usage": group["user_usage"],
            "judge_usage": group["judge_usage"],
            "estimated_cost_usd": group["estimated_cost_usd"],
        }
        summary["total_estimated_cost_usd"] += group["estimated_cost_usd"]
    (output_root / "summary_standalone_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_markdown_report(summary: dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Detector-Agent Standalone Evaluation Report",
        "",
        "This run uses independent protocol detectors around the interactive OR modeling pipeline.",
        "",
        f"- Pipeline mode: `{args.pipeline_mode}`",
        f"- Detector feedback mode: `{args.detector_feedback_mode}`",
        f"- Retry limit behavior: `{args.retry_limit_behavior}`",
        "- Agent detector records the number of independent clarification questions in each Agent message",
        "- Agent resampling is used only for structurally invalid messages: zero-question non-stop responses or READY_TO_MODEL mixed with questions",
        f"- User-answer monitoring enabled: `{args.monitor_user_answers}`",
        "- If enabled, User answers are passively audited for scope violations and hidden-slot disclosure; they are not retried",
        "- In `none` feedback mode, detector findings are logged but never injected into Agent/User context",
        "- In `protocol_failed` mode, exceeding retry limits stops the current case instead of passing through",
        "",
        "## Run Configuration",
        "",
        f"- TOML dirs: `{', '.join(args.toml_dirs)}`",
        f"- K: `{args.k}`",
        f"- max turns: `{args.max_turns}`",
        f"- agent profiles: `{', '.join(args.agent_profiles)}`",
        f"- detector profile: `{args.detector_profile}`",
        f"- user profile: `{args.user_profile}`",
        f"- judge profile: `{args.judge_profile}`",
        f"- prompts dir: `{args.prompts_dir}`",
        f"- Max agent retries per turn: `{MAX_AGENT_RETRIES_PER_TURN}`",
        "",
        "## Summary",
        "",
        "Core exact restore uses only P0/P1 slots. Unresolved P2 slots do not affect this metric.",
        "",
        "| agent | runs | all-slot exact restore | core exact restore | weighted macro | weighted micro | accepted turns | attempted turns | ready rate | protocol failure rate | failure types | agent retries/run | atomic questions/run | multi-question turn rate | avg questions/question-turn | question checks/run | structural invalid/run | answer audits/run | answer scope violations/run | answer disclosed slots/run | answer multi-slot disclosures/run | silent assumptions/run | stopping mismatch | est. cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, item in summary["profiles"].items():
        m = item["metrics"]
        status = ", ".join(f"{k}={v}" for k, v in sorted(m["StoppingStatusCounts"].items()))
        rule_status = ", ".join(f"{k}={v}" for k, v in sorted(m["RuleBasedStoppingStatusCounts"].items()))
        failure_types = ", ".join(f"{k}={v}" for k, v in sorted(m["ProtocolFailureTypeCounts"].items()))
        core_exact_rate = (
            f"{m['CoreExactRestoreRate']:.3f} ({m['CoreExactRestoreCount']})"
            if m["CoreExactRestoreRate"] is not None
            else "n/a (0)"
        )
        lines.append(
            f"| {profile} | {item['runs']} | {m['AllSlotExactRestoreRate']:.3f} ({m['AllSlotExactRestoreCount']}) | "
            f"{core_exact_rate} | "
            f"{m['WeightedSlotScore_macro']:.3f} | "
            f"{m['WeightedSlotScore_micro']:.3f} | "
            f"{m['AverageTurns']:.2f} | {m['AverageAttemptedTurns']:.2f} | "
            f"{m['ReadyToModelRate']:.3f} | {m['ProtocolFailureRate']:.3f} | {failure_types} | "
            f"{m['AvgAgentRetriesPerRun']:.2f} | {m['AtomicQuestionsPerRun']:.2f} | "
            f"{m['MultiQuestionTurnRate']:.3f} | {m['AvgQuestionsPerQuestionTurn']:.2f} | "
            f"{m['DetectorCallsPerRun']:.2f} | {m['DetectorRejectionsPerRun']:.2f} | "
            f"{m['AnswerScopeDetectorCallsPerRun']:.2f} | {m['AnswerScopeViolationsPerRun']:.2f} | "
            f"{m['AnswerDisclosedHiddenSlotsPerRun']:.2f} | {m['AnswerMultiHiddenSlotDisclosurePerRun']:.2f} | "
            f"{m['SilentAssumptionsPerRun']:.2f} | {m['StoppingStatusMismatchRate']:.3f} | "
            f"{item['estimated_cost_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Total estimated cost: `${summary['total_estimated_cost_usd']:.4f}`.",
            "",
            "Detailed per-run transcripts and judge JSON files are stored under the run directory.",
            "",
        ]
    )
    (output_root / "standalone_eval_report.md").write_text("\n".join(lines), encoding="utf-8")


# [模块目标]：集中定义命令行接口，便于 notebook、终端和自动测试共用同一套默认值。
# [输入输出]：无业务输入；返回配置完成的 argparse 解析器。
def build_argument_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_prompts_dir = script_dir / "prompts"
    parser = argparse.ArgumentParser(description="Run the passive-detector Open interaction pipeline.")
    parser.add_argument(
        "--toml_dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_TOML_DIRS],
        help="One or more non-recursive TOML directories. Duplicate case_id values are rejected.",
    )
    parser.add_argument(
        "--output_dir",
        help=(
            "Run output directory. Defaults to a new timestamped directory under "
            "AIOR_RUNS_ROOT when set, otherwise <repository root>/runs/."
        ),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--agent_profiles", nargs="+", default=["generic_agent"])
    parser.add_argument("--detector_profile", default="detector")
    parser.add_argument("--user_profile", default="user_simulator")
    parser.add_argument("--judge_profile", default="judge")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--detector_temperature", type=float, default=0.0)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--prompts_dir", default=str(default_prompts_dir))
    parser.add_argument(
        "--max_agent_retries",
        type=int,
        default=0,
        help="Structural-invalid retry count. Zero keeps the default passive, no-resampling condition.",
    )
    parser.add_argument(
        "--monitor_user_answers",
        action="store_true",
        help="Also use the independent detector to check whether the simulated user answers only the current question.",
    )
    parser.add_argument(
        "--detector_feedback_mode",
        choices=["visible", "minimal", "none"],
        default="none",
        help="Feedback used only when max_agent_retries is greater than zero; none keeps retries silent.",
    )
    parser.add_argument(
        "--retry_limit_behavior",
        choices=["pass_through", "protocol_failed"],
        default="pass_through",
        help="pass_through continues after retry limit; protocol_failed stops the current case after retry limit.",
    )
    return parser


def determine_pipeline_mode(args: argparse.Namespace) -> str:
    intervention_enabled = (
        args.max_agent_retries > 0
        or args.detector_feedback_mode != "none"
        or args.retry_limit_behavior == "protocol_failed"
    )
    behavior = "agent_protocol_intervention" if intervention_enabled else "passive_question_audit"
    if args.monitor_user_answers:
        behavior += "_with_answer_audit"
    return behavior


def make_default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"evaluation_protocol_{timestamp}"


def print_milestone(stage: str, status: str) -> None:
    print(f"| {stage} | {status} |", flush=True)


# [模块目标]：完成参数解析、数据加载、批量运行和最终汇总，是脚本的机器执行入口。
# [输入输出]：输入来自命令行；输出写入时间戳 run 目录，并在控制台按约 10% 粒度汇报里程碑。
def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.max_agent_retries < 0:
        parser.error("--max_agent_retries must be zero or greater")
    args.pipeline_mode = determine_pipeline_mode(args)

    # 使用命令行参数覆盖全局重试限制
    global MAX_AGENT_RETRIES_PER_TURN
    MAX_AGENT_RETRIES_PER_TURN = args.max_agent_retries

    prompts = load_prompt_files(Path(args.prompts_dir))
    cases = load_cases([Path(path) for path in args.toml_dirs], args.limit)
    output_root = Path(args.output_dir) if args.output_dir else make_default_output_dir()
    args.output_dir = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    total_runs = len(args.agent_profiles) * len(cases) * args.k
    print("| 里程碑 | 状态 |", flush=True)
    print("|---|---|", flush=True)
    print_milestone("数据加载完成", f"{len(cases)} 个 case，计划 {total_runs} 次运行")
    print_milestone("输出目录已创建", str(output_root))

    stats: list[dict[str, Any]] = []
    completed_runs = 0
    next_progress_percent = 10
    for profile in args.agent_profiles:
        for case in cases:
            for run_index in range(1, args.k + 1):
                stat = run_interaction(
                    case=case,
                    agent_profile=profile,
                    detector_profile=args.detector_profile,
                    user_profile=args.user_profile,
                    judge_profile=args.judge_profile,
                    run_index=run_index,
                    output_root=output_root,
                    prompts=prompts,
                    agent_temperature=args.agent_temperature,
                    detector_temperature=args.detector_temperature,
                    user_temperature=args.user_temperature,
                    judge_temperature=args.judge_temperature,
                    max_turns=args.max_turns,
                    monitor_user_answers=args.monitor_user_answers,
                    detector_feedback_mode=args.detector_feedback_mode,
                    retry_limit_behavior=args.retry_limit_behavior,
                    pipeline_mode=args.pipeline_mode,
                )
                stats.append(stat)
                completed_runs += 1
                progress_percent = int(completed_runs * 100 / max(1, total_runs))
                if progress_percent >= next_progress_percent or completed_runs == total_runs:
                    print_milestone("LLM 批量处理", f"{progress_percent}%（{completed_runs}/{total_runs}）")
                    while next_progress_percent <= progress_percent:
                        next_progress_percent += 10
    summary = summarize(stats, output_root)
    write_markdown_report(summary, output_root, args)
    print_milestone("评估汇总完成", f"报告：{output_root / 'standalone_eval_report.md'}")
    print_milestone("估算总成本", f"${summary['total_estimated_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
