"""
Choice MC and MC-D pipeline：独立实现两种选择式澄清协议。

协议与审计边界：
1. User Simulator 每次选择时同时输出 `choice`、`rationale` 与 `match_status`；
   不引入 `best_available_option`。
2. `rationale` 必须由作出选择的 User Simulator 当场陈述，记录其真实选择理由；
   `match_status` 结构化标记 A/B/C 与私有业务事实的匹配程度。
3. 使用 `user_choice_audit_events.json` 保存上述审计信息。审计字段不进入 Transcript、
   不提供给 Agent、不触发重试或改写；MC 与 MC-D 的 A/B/C 回答仍只向 Agent 传递 `choice`。
   MC-D 选择 D 时继续沿用既有 `comment` 作为必要的业务纠正内容。
4. 汇总选项覆盖与 MC 强制选择指标；默认读取包内 OR-Clarify 案例，
   每个批次使用独立的 `choice_mc_mcd_*` 输出目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
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


INTERACTION_MODES = ("mc", "mc_d")
FIXED_OPTION_D_TEXT = "None of the above — I'd like to explain in my own words."
CHOICE_MATCH_STATUSES = ("exact_match", "acceptable_match", "no_match", "undetermined")
USER_CHOICE_AUDIT_FILENAME = "user_choice_audit_events.json"

# 默认实验只审计、不干预；命令行显式提高重试次数后才启用重采样。
MAX_AGENT_RETRIES_PER_TURN = 0
MAX_USER_RETRIES_PER_TURN = 0


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "PACKAGE_ROOT.marker").exists() or (
            (candidate / "AGENTS.md").exists() and (candidate / "doc-wiki.md").exists()
        ):
            return candidate
    raise RuntimeError("Could not find repo root from choice_mc_mcd/run_pipeline.py")


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
CANONICAL_JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.md"

MODEL_PROFILES = {
    "generic_agent": {
        "model_env": "GENERIC_AGENT_MODEL",
        "fallback_env": "OPENAI_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
    "user_simulator": {
        "model_env": "USER_SIMULATOR_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
    "detector": {
        "model_env": "DETECTOR_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
    "judge": {
        "model_env": "JUDGE_MODEL",
        "fallback_model": "deepseek-v4-pro",
    },
}


def resolve_model_name(profile_name: str) -> str:
    profile = MODEL_PROFILES.get(profile_name)
    if profile is None:
        return profile_name
    fallback_env = profile.get("fallback_env")
    fallback_model = os.getenv(fallback_env) if fallback_env else None
    model = os.getenv(profile["model_env"]) or fallback_model or profile["fallback_model"]
    return model.strip()


def _resolve_api_settings(profile_name: str) -> tuple[str, str, bool]:
    if profile_name == "generic_agent" and (
        os.getenv("GENERIC_AGENT_BASE_URL") or os.getenv("GENERIC_AGENT_API_KEY")
    ):
        api_key = os.getenv("GENERIC_AGENT_API_KEY", "")
        base_url = os.getenv("GENERIC_AGENT_BASE_URL", "").rstrip("/")
        if not api_key or not base_url:
            raise RuntimeError(
                "GENERIC_AGENT_BASE_URL and GENERIC_AGENT_API_KEY must be set together "
                "when overriding only the generic_agent endpoint."
            )
        return base_url, api_key, True

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Put it in repo root .env or the process environment.")
    return base_url, api_key, False


def _parse_generic_agent_rpm_limit() -> int | None:
    raw = os.getenv("GENERIC_AGENT_RPM_LIMIT", "").strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError("GENERIC_AGENT_RPM_LIMIT must be a positive integer.") from exc
    if limit <= 0:
        raise RuntimeError("GENERIC_AGENT_RPM_LIMIT must be a positive integer.")
    return limit


def _rate_limit_bucket_name(base_url: str) -> str:
    explicit_bucket = os.getenv("GENERIC_AGENT_RATE_LIMIT_BUCKET", "").strip()
    if explicit_bucket:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", explicit_bucket)
    digest = hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:16]
    return f"generic_agent_{digest}"


def _rate_limit_dir() -> Path:
    explicit_dir = os.getenv("GENERIC_AGENT_RATE_LIMIT_DIR", "").strip()
    if explicit_dir:
        return Path(explicit_dir)
    return Path(tempfile.gettempdir()) / "ai_interaction_or_rate_limits"


def _acquire_lock(lock_path: Path) -> None:
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 120:
                    lock_path.unlink()
            except OSError:
                pass
            time.sleep(0.05)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _wait_for_generic_agent_rate_limit(base_url: str) -> None:
    rpm_limit = _parse_generic_agent_rpm_limit()
    if rpm_limit is None:
        return

    rate_dir = _rate_limit_dir()
    rate_dir.mkdir(parents=True, exist_ok=True)
    bucket = _rate_limit_bucket_name(base_url)
    stamps_path = rate_dir / f"{bucket}.json"
    lock_path = rate_dir / f"{bucket}.lock"

    while True:
        _acquire_lock(lock_path)
        try:
            now = time.time()
            stamps: list[float] = []
            if stamps_path.exists():
                try:
                    stamps = [
                        float(item)
                        for item in json.loads(stamps_path.read_text(encoding="utf-8"))
                        if now - float(item) < 60.0
                    ]
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    stamps = []
            if len(stamps) < rpm_limit:
                stamps.append(now)
                stamps_path.write_text(json.dumps(stamps), encoding="utf-8")
                return
            wait_seconds = max(0.1, 60.0 - (now - min(stamps)) + 0.05)
        finally:
            _release_lock(lock_path)
        time.sleep(min(wait_seconds, 5.0))

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

# statistics.json 只保留研究结论需要的核心字段；其余工程/诊断字段写入 run_health.json。
CORE_STAT_KEYS = (
    "pipeline_version",
    "pipeline_mode",
    "interaction_mode",
    "case_id",
    "run_index",
    "agent_profile",
    "agent_model",
    "weighted_slot_score",
    "earned_weight",
    "total_weight",
    "completed_ready_to_model",
    "turn_count",
    "stopping_status",
    "rule_based_stopping_status",
    "stopping_status_mismatch",
    "silent_assumption_count",
    "protocol_failed",
    "protocol_failure_type",
    "detector_call_count",
    "detector_rejection_count",
    "detector_error_count",
    "monitor_user_answers",
    "answer_scope_policy",
    "answer_scope_detector_call_count",
    "answer_scope_rejection_count",
    "answer_scope_error_count",
    "agent_question_turn_count",
    "agent_atomic_question_count",
    "agent_multi_question_turn_count",
    "answer_independent_fact_count",
    "answer_disclosed_hidden_slot_count",
    "answer_multi_hidden_slot_disclosure_count",
    "core_exact_restore",
    "core_slot_count",
    "core_unresolved_slot_count",
    "p2_slot_count",
    "p2_unresolved_slot_count",
    "all_slot_exact_restore",
    "all_slot_count",
    "choice_metrics",
    "choice_audit_metrics",
    "estimated_cost_usd",
    "run_dir",
)


def split_stat(stat: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    core = {key: stat[key] for key in CORE_STAT_KEYS if key in stat}
    health = {key: value for key, value in stat.items() if key not in CORE_STAT_KEYS}
    for link_key in ("case_id", "interaction_mode", "run_index", "run_dir"):
        if link_key in stat:
            health.setdefault(link_key, stat[link_key])
    return core, health


def write_stat_files(stat: dict[str, Any], run_dir: Path) -> None:
    core_stat, health_stat = split_stat(stat)
    (run_dir / "statistics.json").write_text(
        json.dumps(core_stat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "run_health.json").write_text(
        json.dumps(health_stat, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@dataclass
class ChatResult:
    content: str
    usage: dict[str, int]
    estimated_cost_usd: float


# [模块目标]：统一 Choice Pipeline 四个 LLM 角色的调用方式，并累计 token 与估算成本。
# [输入输出]：输入角色/profile 和温度；complete 接收消息列表并返回文本、用量和成本。
# [LLM 交互]：默认所有角色走 DeepSeek-compatible 接口；若配置 GENERIC_AGENT_*，
# 只有建模 Agent 会切到专用 endpoint，并在请求前遵守共享 RPM 限速。
class ChatClient:
    def __init__(self, profile_name: str, temperature: float):
        self.profile_name = profile_name
        self.model = resolve_model_name(profile_name)
        self.temperature = temperature
        self.total_usage: dict[str, int] = {}
        self.total_estimated_cost_usd = 0.0

    def _api_settings(self) -> tuple[str, str, bool]:
        return _resolve_api_settings(self.profile_name)

    def complete(self, messages: list[dict[str, str]], timeout: int = 180, max_retries: int = 10) -> ChatResult:
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
        base_url, api_key, uses_generic_agent_endpoint = self._api_settings()
        if uses_generic_agent_endpoint:
            _wait_for_generic_agent_rate_limit(base_url)
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
        independent_fact_count = int(result.get("independent_answered_fact_count", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Answer-scope detector independent_answered_fact_count must be an integer.") from exc
    if independent_fact_count < 0:
        raise ValueError("Answer-scope detector independent_answered_fact_count cannot be negative.")
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


def normalize_mc_agent_payload(result: dict[str, Any], interaction_mode: str) -> dict[str, Any]:
    action = str(result.get("action", "")).strip().upper()
    if action not in {"ASK", "READY_TO_MODEL"}:
        raise ValueError(f"Unsupported MC agent action: {action!r}")

    if action == "READY_TO_MODEL":
        if result.get("question") or result.get("options"):
            raise ValueError("READY_TO_MODEL must not include question or options.")
        return {
            "action": "READY_TO_MODEL",
            "summary": str(result.get("summary", "")).strip(),
        }

    question = str(result.get("question", "")).strip()
    if not question:
        raise ValueError("ASK must include a non-empty question.")
    options = result.get("options", [])
    if not isinstance(options, list):
        raise ValueError("ASK options must be a list.")
    if not (2 <= len(options) <= 4):
        raise ValueError("ASK must include between 2 and 4 options.")
    normalized_options: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("Each option must be an object.")
        option_id = str(option.get("id", "")).strip().upper()
        option_text = str(option.get("text", "")).strip()
        if option_id not in {"A", "B", "C"}:
            raise ValueError("Option ids must be A, B, or C only.")
        if option_id in seen_ids:
            raise ValueError(f"Duplicate option id: {option_id}")
        if not option_text:
            raise ValueError(f"Option {option_id} must have non-empty text.")
        seen_ids.add(option_id)
        normalized_options.append({"id": option_id, "text": option_text})
    if len(normalized_options) != 3 or seen_ids != {"A", "B", "C"}:
        raise ValueError("ASK must include exactly options A, B, and C.")

    allow_other = bool(result.get("allow_other", False))
    if interaction_mode == "mc" and allow_other:
        raise ValueError("Pure MC mode must set allow_other to false.")
    if interaction_mode == "mc_d" and not allow_other:
        raise ValueError("MC+D mode must set allow_other to true.")

    return {
        "action": "ASK",
        "question": question,
        "options": normalized_options,
        "allow_other": allow_other,
    }


def normalize_choice_detector_result(result: dict[str, Any]) -> dict[str, Any]:
    action = str(result.get("action", "")).strip().lower()
    if action not in {"ask", "ready_to_model", "invalid"}:
        raise ValueError(f"Unsupported choice detector action: {action!r}")
    try:
        question_count = int(result.get("question_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Choice detector question_count must be an integer.") from exc
    try:
        option_count = int(result.get("option_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Choice detector option_count must be an integer.") from exc
    option_ids = result.get("option_ids", [])
    if not isinstance(option_ids, list):
        raise ValueError("Choice detector option_ids must be a list.")
    option_ids = [str(item).strip().upper() for item in option_ids]
    is_valid_mc = bool(result.get("is_valid_mc", False))

    if action == "ask" and not (
        is_valid_mc
        and question_count >= 1
        and 2 <= option_count <= 4
        and len(option_ids) == option_count
        and len(set(option_ids)) == option_count
    ):
        raise ValueError("An ask action must describe a structurally valid multiple-choice question block.")
    if action == "ready_to_model" and not (
        is_valid_mc and question_count == 0 and option_count == 0 and not option_ids
    ):
        raise ValueError("A ready_to_model action must contain zero questions and options.")
    if action == "invalid" and is_valid_mc:
        raise ValueError("An invalid action cannot be marked valid.")

    return {
        "action": action,
        "question_count": question_count,
        "is_valid_mc": is_valid_mc,
        "question": str(result.get("question", "")).strip(),
        "option_count": option_count,
        "option_ids": option_ids,
        "allow_other": bool(result.get("allow_other", False)),
        "rationale": str(result.get("rationale", "")).strip(),
    }


# [模块目标]：校验 User Simulator 的选择、当场理由与选项匹配状态，形成完整但仅供审计的结构化结果。
# [输入输出]：输入 LLM JSON 与 mc/mc_d 模式；输出含 choice、rationale、match_status，D 时额外保留 comment。
# [LLM 交互]：强制理由由作答用户同时生成；禁止 best_available_option，避免把研究者事后判断混入用户原始想法。
def normalize_choice_simulator_result(result: dict[str, Any], interaction_mode: str) -> dict[str, Any]:
    if "best_available_option" in result:
        raise ValueError("Choice protocol does not allow best_available_option")
    choice = str(result.get("choice", "")).strip().upper()
    rationale = str(result.get("rationale", "")).strip()
    match_status = str(result.get("match_status", "")).strip().lower()
    if not rationale:
        raise ValueError("Choice response requires non-empty rationale")
    if match_status not in CHOICE_MATCH_STATUSES:
        raise ValueError(
            f"match_status must be one of {CHOICE_MATCH_STATUSES}, got {match_status!r}"
        )
    if interaction_mode == "mc":
        expected_keys = {"choice", "rationale", "match_status"}
        if set(result) != expected_keys:
            raise ValueError(f"mc choice response keys must be {sorted(expected_keys)}")
        if choice not in {"A", "B", "C"}:
            raise ValueError(f"mc mode choice must be A/B/C, got {choice!r}")
        return {"choice": choice, "rationale": rationale, "match_status": match_status}
    if interaction_mode == "mc_d":
        if choice in {"A", "B", "C"}:
            expected_keys = {"choice", "rationale", "match_status"}
            if set(result) != expected_keys:
                raise ValueError(f"mc_d A/B/C response keys must be {sorted(expected_keys)}")
            if match_status == "no_match":
                raise ValueError("mc_d must choose D when match_status is no_match")
            return {"choice": choice, "rationale": rationale, "match_status": match_status}
        if choice == "D":
            expected_keys = {"choice", "comment", "rationale", "match_status"}
            if set(result) != expected_keys:
                raise ValueError(f"mc_d D response keys must be {sorted(expected_keys)}")
            comment = str(result.get("comment", "")).strip()
            if not comment:
                raise ValueError("mc_d mode requires non-empty comment when choice is D")
            if match_status != "no_match":
                raise ValueError("mc_d choice D requires match_status=no_match")
            return {
                "choice": "D",
                "comment": comment,
                "rationale": rationale,
                "match_status": match_status,
            }
        raise ValueError(f"mc_d mode choice must be A/B/C/D, got {choice!r}")
    raise ValueError(f"Unsupported interaction_mode for choice simulator: {interaction_mode}")


# [模块目标]：从完整用户输出中剥离审计字段，得到唯一允许进入正式对话和 Agent 上下文的内容。
# [输入输出]：输入含选择理由与匹配状态的审计结果；输出仅含 choice，MC-D 选择 D 时保留既有 comment。
# [LLM 交互]：不调用 LLM；这是防止 rationale/match_status 泄露给 Agent 的硬边界。
def build_agent_visible_choice(sim_result: dict[str, Any]) -> dict[str, Any]:
    visible = {"choice": sim_result["choice"]}
    if sim_result["choice"] == "D":
        visible["comment"] = sim_result["comment"]
    return visible


def render_mc_question_display(payload: dict[str, Any], interaction_mode: str) -> str:
    lines = ["Question:", payload["question"], "", "Options:"]
    for option in payload["options"]:
        lines.append(f"{option['id']}. {option['text']}")
    if interaction_mode == "mc_d":
        lines.append(f"D. {FIXED_OPTION_D_TEXT}")
    return "\n".join(lines)


def render_choice_user_message(payload: dict[str, Any], interaction_mode: str) -> str:
    return (
        "The assistant asks a multiple-choice clarification question.\n\n"
        + render_mc_question_display(payload, interaction_mode)
    )


def render_choice_response_for_agent(sim_result: dict[str, Any]) -> str:
    choice = sim_result["choice"]
    if choice == "D":
        return f"Business user response: none of the offered options match. {sim_result['comment']}"
    return f"Business user selected option {choice}."


def should_check_answer_scope(interaction_mode: str, sim_result: dict[str, Any]) -> bool:
    return interaction_mode == "mc_d" and sim_result.get("choice") == "D" and bool(sim_result.get("comment", "").strip())


def build_minimal_agent_retry_feedback(interaction_mode: str, *, detector_error: bool = False) -> str:
    expected = (
        "Return exactly one valid JSON object with action ASK or READY_TO_MODEL."
        if interaction_mode != "open"
        else "Return exactly one valid QUESTION line or READY_TO_MODEL."
    )
    reason = "The previous response could not be validated." if detector_error else "The previous response violated the interaction protocol."
    return "\n".join(
        [
            "Protocol feedback from the interaction supervisor:",
            reason,
            expected,
            "Rewrite your previous response without adding extra explanation.",
        ]
    )


def build_agent_protocol_feedback(
    interaction_mode: str,
    *,
    detector_result: dict[str, Any] | None = None,
    detector_error: str | None = None,
) -> str:
    lines = [
        "Protocol feedback from the interaction supervisor:",
        "Your previous response did not pass the interaction protocol.",
    ]
    if detector_error:
        lines.append(f"Detector/validation error: {detector_error}")
    if detector_result:
        action = detector_result.get("action", "")
        rationale = detector_result.get("rationale", "")
        lines.append(f"Detected action: {action}")
        if "question_count" in detector_result:
            lines.append(f"Detected question count: {detector_result.get('question_count')}")
        if rationale:
            lines.append(f"Detector rationale: {rationale}")
    if interaction_mode == "open":
        lines.append("Rewrite as exactly one minimal clarification question, or READY_TO_MODEL.")
    else:
        lines.append("Rewrite as exactly one valid JSON ASK object, or READY_TO_MODEL.")
    return "\n".join(lines)


def build_minimal_user_format_retry_feedback() -> str:
    return "\n".join(
        [
            "Protocol feedback from the interaction supervisor:",
            "The previous response could not be parsed as the required choice JSON.",
            "Rewrite the JSON response. If choosing D, keep comment limited to the current question and why the offered options do not fit.",
        ]
    )


def build_user_format_feedback(detector_error: str) -> str:
    lines = [
        "Protocol feedback from the interaction supervisor:",
        "Your previous response could not be parsed as the required choice JSON.",
        f"Validation error: {detector_error}",
        "Rewrite the JSON response without adding extra prose.",
    ]
    return "\n".join(lines)


def render_hidden_slots_for_answer_audit(slots: list[dict[str, Any]]) -> str:
    lines = []
    for slot in slots:
        slot_id = str(slot.get("slot_id", "")).strip()
        answer = str(slot.get("simulator_answer", "")).strip()
        if slot_id and answer:
            lines.append(f"- {slot_id}: {answer}")
    return "\n".join(lines)


def render_d_comment_scope_message(
    agent_payload: dict[str, Any],
    sim_result: dict[str, Any],
    hidden_slots: list[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            "# Current assistant question",
            str(agent_payload.get("question", "")).strip(),
            "# Business user selected option",
            "D. " + FIXED_OPTION_D_TEXT,
            "# D-option free-text comment to check",
            str(sim_result.get("comment", "")).strip(),
            "# Hidden slots for passive audit",
            render_hidden_slots_for_answer_audit(hidden_slots),
        ]
    )


def load_case(path: Path) -> dict[str, Any]:
    case = tomllib.loads(path.read_text(encoding="utf-8"))
    case["_path"] = str(path)
    case["_case_id"] = case.get("metadata", {}).get("case_id", path.stem)
    return case


def case_sort_key(case: dict[str, Any]) -> str:
    return str(case["_case_id"])


# [模块目标]：把多个 OR-Clarify 案例批次合并成一次选择题实验输入，并在调用 LLM 前阻止重复 case。
# [输入输出]：输入单个目录或目录列表；输出按 case_id 排序的 case 列表。
def read_cases(toml_dirs: Path | list[Path] | tuple[Path, ...]) -> list[dict[str, Any]]:
    directories = [toml_dirs] if isinstance(toml_dirs, Path) else list(toml_dirs)
    paths: list[Path] = []
    for toml_dir in directories:
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
    return cases


def case_selector_aliases(selector: str) -> set[str]:
    raw = str(selector).strip()
    if not raw:
        return set()
    aliases = {raw}
    if raw.isdigit():
        normalized_number = str(int(raw))
        padded_number = normalized_number.zfill(3)
        aliases.update({normalized_number, padded_number, f"orclarify_{padded_number}"})
    case_id_match = re.fullmatch(r"(?P<prefix>[A-Za-z][A-Za-z0-9]*)_(?P<num>\d+)", raw)
    if case_id_match:
        number = str(int(case_id_match.group("num")))
        padded_number = number.zfill(3)
        prefix = case_id_match.group("prefix").lower()
        aliases.update({number, padded_number, f"{prefix}_{padded_number}"})
    return aliases


def case_lookup_keys(case: dict[str, Any]) -> set[str]:
    case_id = str(case.get("_case_id") or case.get("metadata", {}).get("case_id") or "").strip()
    return {key for key in case_selector_aliases(case_id) if key}


def resolve_case_selection(
    cases: list[dict[str, Any]], case_ids: list[str] | None
) -> tuple[list[dict[str, Any]], list[str]]:
    if case_ids:
        case_keys = [(case, case_lookup_keys(case)) for case in cases]
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for selector in case_ids:
            aliases = case_selector_aliases(selector)
            if not any(keys & aliases for _, keys in case_keys):
                missing.append(selector)
        for case, keys in case_keys:
            if any(keys & case_selector_aliases(selector) for selector in case_ids):
                selected.append(case)
        return selected, missing
    return cases, []


def load_cases(
    toml_dirs: Path | list[Path] | tuple[Path, ...],
    limit: int | None,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    cases = read_cases(toml_dirs)
    cases, _ = resolve_case_selection(cases, case_ids)
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
        if row.get("structured"):
            chunks.append("```json")
            chunks.append(json.dumps(row["structured"], ensure_ascii=False, indent=2))
            chunks.append("```")
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
    core_slots = [score for score in slot_scores if score.get("severity") in {"P0", "P1"}]
    p2_slots = [score for score in slot_scores if score.get("severity") == "P2"]
    all_slots = list(slot_scores)

    def is_exact(scores: list[dict[str, Any]]) -> bool:
        return bool(scores) and all(score.get("hit") == "yes" for score in scores)

    def unresolved(scores: list[dict[str, Any]]) -> list[str]:
        return [str(score.get("slot_id", "")) for score in scores if score.get("hit") != "yes"]

    return {
        "core_exact_restore": is_exact(core_slots),
        "core_slot_count": len(core_slots),
        "core_unresolved_slots": unresolved(core_slots),
        "p2_slot_count": len(p2_slots),
        "p2_unresolved_slots": unresolved(p2_slots),
        "all_slot_exact_restore": is_exact(all_slots),
        "all_slot_count": len(all_slots),
        "all_unresolved_slots": unresolved(all_slots),
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


def summarize_choice_metrics(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    d_count = 0
    ask_count = 0
    for row in transcript:
        structured = row.get("structured") or {}
        if row.get("speaker") == "generic_agent" and structured.get("action") == "ASK":
            ask_count += 1
        if row.get("speaker") == "user_simulator":
            choice = str(structured.get("choice", "")).upper()
            if choice == "D":
                d_count += 1
    return {
        "mc_question_turns": ask_count,
        "d_selection_count": d_count,
        "d_selection_rate": d_count / ask_count if ask_count else None,
    }


# [模块目标]：把 User Simulator 当场留下的选项匹配状态汇总成可用于论文分析的选项覆盖指标。
# [输入输出]：输入每轮用户审计事件；输出各状态计数、无匹配率，以及 MC 被迫三选一的次数与比例。
# [LLM 交互]：不调用 LLM；只聚合 User Simulator 已落盘的原始 choice/rationale/match_status。
def summarize_user_choice_audit(
    events: list[dict[str, Any]], interaction_mode: str
) -> dict[str, Any]:
    counts = {status: 0 for status in CHOICE_MATCH_STATUSES}
    for event in events:
        status = str(event.get("match_status", "")).strip().lower()
        if status in counts:
            counts[status] += 1
    total = sum(counts.values())
    no_match_count = counts["no_match"]
    forced_choice_count = no_match_count if interaction_mode == "mc" else 0
    return {
        "choice_audit_event_count": total,
        "choice_match_status_counts": counts,
        "choice_no_match_count": no_match_count,
        "choice_no_match_rate": no_match_count / total if total else None,
        "mc_forced_choice_count": forced_choice_count,
        "mc_forced_choice_rate": forced_choice_count / total if total else None,
    }


def summarize_question_audit(detector_events: list[dict[str, Any]]) -> dict[str, int]:
    question_turn_count = 0
    atomic_question_count = 0
    multi_question_turn_count = 0
    unknown_question_count_turns = 0
    for event in detector_events:
        parsed_agent = event.get("parsed_agent") or {}
        if not event.get("entered_transcript") or parsed_agent.get("action") != "ASK":
            continue
        question_turn_count += 1
        detector_result = event.get("detector_result") or {}
        question_count = detector_result.get("question_count")
        if not isinstance(question_count, int) or question_count < 1:
            unknown_question_count_turns += 1
            continue
        atomic_question_count += question_count
        if question_count > 1:
            multi_question_turn_count += 1
    return {
        "agent_question_turn_count": question_turn_count,
        "agent_atomic_question_count": atomic_question_count,
        "agent_multi_question_turn_count": multi_question_turn_count,
        "agent_question_count_unknown_turn_count": unknown_question_count_turns,
    }


@dataclass
class PromptBundle:
    agent_prompt: str
    primary_detector_prompt: str
    answer_detector_prompt: str
    simulator_prompt: str
    judge_prompt: str


def load_prompt_bundle(
    prompts_dir: Path,
    interaction_mode: str,
    judge_prompt_path: Path | None = None,
) -> PromptBundle:
    if interaction_mode not in INTERACTION_MODES:
        raise ValueError(f"Unsupported interaction_mode: {interaction_mode}")
    judge_path = judge_prompt_path or CANONICAL_JUDGE_PROMPT_PATH
    if not judge_path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {judge_path}")

    agent_file = "mc_d_agent_prompt.md" if interaction_mode == "mc_d" else "mc_agent_prompt.md"
    simulator_file = (
        "choice_user_simulator_mc_d.md"
        if interaction_mode == "mc_d"
        else "choice_user_simulator_mc.md"
    )
    return PromptBundle(
        agent_prompt=(prompts_dir / agent_file).read_text(encoding="utf-8").strip(),
        primary_detector_prompt=(prompts_dir / "choice_detector_prompt.md").read_text(encoding="utf-8").strip(),
        answer_detector_prompt=(prompts_dir / "answer_scope_detector_prompt.md").read_text(encoding="utf-8").strip(),
        simulator_prompt=(prompts_dir / simulator_file).read_text(encoding="utf-8").strip(),
        judge_prompt=judge_path.read_text(encoding="utf-8").strip(),
    )


# [模块目标]：完成一个 mc 或 mc_d case 的多轮选择题访谈、被动 Detector 审计和 Judge 评分。
# [输入输出]：输入 case、Prompt 和实验参数；输出单 run 统计，并写入 transcript、事件与评分证据。
# [LLM 交互]：Agent 生成 JSON 问题卡；Choice Detector 只计数并审计结构；Simulator 选择；可选 Answer-Scope 只审计 D comment；Judge 最终评分。
def run_interaction(
    case: dict[str, Any],
    agent_profile: str,
    detector_profile: str,
    user_profile: str,
    judge_profile: str,
    run_index: int,
    output_root: Path,
    prompts: PromptBundle,
    interaction_mode: str,
    agent_temperature: float,
    detector_temperature: float,
    user_temperature: float,
    judge_temperature: float,
    max_turns: int,
    detector_feedback_mode: str = "none",
    retry_limit_behavior: str = "pass_through",
    monitor_user_answers: bool = False,
    pipeline_mode: str | None = None,
) -> dict[str, Any]:
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
        pipeline_mode = (
            "choice_agent_protocol_intervention"
            if intervention_enabled
            else "choice_passive_question_audit"
        )
        if monitor_user_answers:
            pipeline_mode += "_with_answer_audit"

    case_id = case["_case_id"]
    run_dir = output_root / interaction_mode / agent_profile / f"run_{run_index:02d}" / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "statistics.json"
    health_path = run_dir / "run_health.json"
    judge_path = run_dir / "judge_result.json"
    if stats_path.exists() and judge_path.exists():
        stat = json.loads(stats_path.read_text(encoding="utf-8"))
        if health_path.exists():
            stat = {**json.loads(health_path.read_text(encoding="utf-8")), **stat}
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
        write_stat_files(stat, run_dir)
        return stat

    agent_client = ChatClient(agent_profile, agent_temperature)
    detector_client = ChatClient(detector_profile, detector_temperature)
    user_client = ChatClient(user_profile, user_temperature)
    judge_client = ChatClient(judge_profile, judge_temperature)

    first_user_message = (
        "Here is the user's initial request. Interview the user if clarification is needed.\n\n"
        + case["initial_brief"]["content"]
    )
    simulator_system_prompt = prompts.simulator_prompt + "\n\n" + render_user_simulator_case(case)
    agent_messages = [
        {"role": "system", "content": prompts.agent_prompt},
        {"role": "user", "content": first_user_message},
    ]
    simulator_messages = [{"role": "system", "content": simulator_system_prompt}]
    transcript: list[dict[str, Any]] = []
    detector_events: list[dict[str, Any]] = []
    answer_scope_events: list[dict[str, Any]] = []
    user_choice_audit_events: list[dict[str, Any]] = []
    completed = False
    protocol_failed = False
    protocol_failure_type = ""
    protocol_failure_reason = ""
    final_agent_text = ""
    detector_rejection_count = 0
    detector_error_count = 0
    answer_scope_rejection_count = 0
    answer_scope_error_count = 0
    answer_independent_fact_count = 0
    answer_disclosed_hidden_slot_count = 0
    answer_multi_hidden_slot_disclosure_count = 0
    agent_retry_total = 0
    user_retry_total = 0
    agent_max_retries_exceeded = False
    user_max_retries_exceeded = False
    answer_scope_detector_call_count = 0
    user_format_error_count = 0
    agent_retry_limit_pass_through_count = 0

    for turn in range(1, max_turns + 1):
        agent_retry = 0
        agent_retry_feedback: str | None = None
        agent_payload: dict[str, Any] | None = None
        agent_display = ""
        while True:
            agent_call_messages = agent_messages
            if detector_feedback_mode == "minimal" and agent_retry_feedback:
                agent_call_messages = agent_messages + [{"role": "user", "content": agent_retry_feedback}]
            agent_reply = agent_client.complete(agent_call_messages).content
            final_agent_text = agent_reply

            if interaction_mode == "open":
                detector_user_message = "Assistant latest response:\n\n" + agent_reply
                detector_messages = [
                    {"role": "system", "content": prompts.primary_detector_prompt},
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
                            agent_display = agent_reply
                        else:
                            agent_max_retries_exceeded = True
                        break
                    if detector_feedback_mode == "visible":
                        agent_messages.append({"role": "assistant", "content": agent_reply})
                        agent_messages.append(
                            {
                                "role": "user",
                                "content": build_agent_protocol_feedback(
                                    interaction_mode, detector_error=str(exc)
                                ),
                            }
                        )
                    elif detector_feedback_mode == "minimal":
                        agent_retry_feedback = build_minimal_agent_retry_feedback(
                            interaction_mode, detector_error=True
                        )
                    agent_retry_total += 1
                    continue

                detector_events.append(event)
                action = detector_result["action"]
                if action == "ready_to_model":
                    event["accepted"] = True
                    event["entered_transcript"] = True
                    transcript.append({"turn": turn, "speaker": "generic_agent", "content": agent_reply})
                    completed = True
                    break
                if action == "question":
                    event["accepted"] = True
                    event["entered_transcript"] = True
                    transcript.append(
                        {
                            "turn": turn,
                            "speaker": "generic_agent",
                            "content": agent_reply,
                            "detected_atomic_question": detector_result["atomic_questions"][0],
                        }
                    )
                    agent_messages.append({"role": "assistant", "content": agent_reply})
                    agent_display = agent_reply
                    break

                detector_rejection_count += 1
                agent_retry += 1
                event["accepted"] = False
                event["entered_transcript"] = False
                event["feedback_mode"] = detector_feedback_mode
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
                        agent_display = agent_reply
                    else:
                        agent_max_retries_exceeded = True
                    break
                if detector_feedback_mode == "visible":
                    agent_messages.append({"role": "assistant", "content": agent_reply})
                    agent_messages.append(
                        {
                            "role": "user",
                            "content": build_agent_protocol_feedback(
                                interaction_mode, detector_result=detector_result
                            ),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    agent_retry_feedback = build_minimal_agent_retry_feedback(interaction_mode)
                agent_retry_total += 1
                continue

            try:
                parsed = normalize_mc_agent_payload(extract_json_object(agent_reply), interaction_mode)
            except Exception as exc:
                detector_error_count += 1
                mc_event = {
                    "turn": turn,
                    "agent_response": agent_reply,
                    "detector_error": str(exc),
                    "retry_count": agent_retry,
                    "accepted": False,
                    "entered_transcript": False,
                    "feedback_mode": detector_feedback_mode,
                }
                detector_events.append(mc_event)
                agent_retry += 1
                if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                    mc_event["retry_limit_exceeded"] = True
                    mc_event["non_pass_through_reason"] = "Unparseable Agent JSON cannot drive a choice turn."
                    agent_max_retries_exceeded = True
                    protocol_failure_type = "agent_format_invalid"
                    break
                if detector_feedback_mode == "visible":
                    agent_messages.append({"role": "assistant", "content": agent_reply})
                    agent_messages.append(
                        {
                            "role": "user",
                            "content": build_agent_protocol_feedback(
                                interaction_mode, detector_error=str(exc)
                            ),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    agent_retry_feedback = build_minimal_agent_retry_feedback(
                        interaction_mode, detector_error=True
                    )
                agent_retry_total += 1
                continue

            detector_user_message = "Assistant latest response:\n\n" + agent_reply
            detector_messages = [
                {"role": "system", "content": prompts.primary_detector_prompt},
                {"role": "user", "content": detector_user_message},
            ]
            detector_raw = detector_client.complete(detector_messages).content
            mc_event = {
                "turn": turn,
                "agent_response": agent_reply,
                "detector_raw": detector_raw,
                "retry_count": agent_retry,
                "parsed_agent": parsed,
            }
            try:
                detector_result = normalize_choice_detector_result(extract_json_object(detector_raw))
                mc_event["detector_result"] = detector_result
            except Exception as exc:
                detector_error_count += 1
                mc_event["detector_error"] = str(exc)
                mc_event["accepted"] = False
                mc_event["entered_transcript"] = False
                mc_event["feedback_mode"] = detector_feedback_mode
                detector_events.append(mc_event)
                agent_retry += 1
                if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                    mc_event["retry_limit_exceeded"] = True
                    if retry_limit_behavior == "pass_through":
                        agent_retry_limit_pass_through_count += 1
                        mc_event["pass_through_after_retry_limit"] = True
                        mc_event["entered_transcript"] = True
                        if parsed["action"] == "READY_TO_MODEL":
                            transcript.append(
                                {
                                    "turn": turn,
                                    "speaker": "generic_agent",
                                    "content": parsed.get("summary") or agent_reply,
                                    "structured": parsed,
                                    "detector_pass_through_after_retry_limit": True,
                                    "detector_pass_through_reason": "detector_parse_error",
                                }
                            )
                            completed = True
                        else:
                            agent_payload = parsed
                            agent_display = render_mc_question_display(parsed, interaction_mode)
                            transcript.append(
                                {
                                    "turn": turn,
                                    "speaker": "generic_agent",
                                    "content": agent_display,
                                    "structured": parsed,
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
                            "content": build_agent_protocol_feedback(
                                interaction_mode, detector_error=str(exc)
                            ),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    agent_retry_feedback = build_minimal_agent_retry_feedback(
                        interaction_mode, detector_error=True
                    )
                agent_retry_total += 1
                continue

            detector_events.append(mc_event)
            if parsed["action"] == "READY_TO_MODEL":
                if detector_result["action"] != "ready_to_model":
                    detector_rejection_count += 1
                    agent_retry += 1
                    mc_event["accepted"] = False
                    mc_event["entered_transcript"] = False
                    mc_event["feedback_mode"] = detector_feedback_mode
                    if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                        mc_event["retry_limit_exceeded"] = True
                        if retry_limit_behavior == "pass_through":
                            agent_retry_limit_pass_through_count += 1
                            mc_event["pass_through_after_retry_limit"] = True
                            mc_event["entered_transcript"] = True
                            transcript.append(
                                {
                                    "turn": turn,
                                    "speaker": "generic_agent",
                                    "content": parsed.get("summary") or agent_reply,
                                    "structured": parsed,
                                    "detector_pass_through_after_retry_limit": True,
                                    "detector_pass_through_reason": "ready_to_model_detector_mismatch",
                                }
                            )
                            completed = True
                        else:
                            agent_max_retries_exceeded = True
                        break
                    if detector_feedback_mode == "visible":
                        agent_messages.append({"role": "assistant", "content": agent_reply})
                        agent_messages.append(
                            {
                                "role": "user",
                                "content": build_agent_protocol_feedback(
                                    interaction_mode, detector_result=detector_result
                                ),
                            }
                        )
                    elif detector_feedback_mode == "minimal":
                        agent_retry_feedback = build_minimal_agent_retry_feedback(interaction_mode)
                    agent_retry_total += 1
                    continue
                mc_event["accepted"] = True
                mc_event["entered_transcript"] = True
                transcript.append(
                    {
                        "turn": turn,
                        "speaker": "generic_agent",
                        "content": parsed.get("summary") or agent_reply,
                        "structured": parsed,
                    }
                )
                completed = True
                break

            if detector_result["action"] != "ask":
                detector_rejection_count += 1
                agent_retry += 1
                mc_event["accepted"] = False
                mc_event["entered_transcript"] = False
                mc_event["feedback_mode"] = detector_feedback_mode
                if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                    mc_event["retry_limit_exceeded"] = True
                    if retry_limit_behavior == "pass_through":
                        agent_retry_limit_pass_through_count += 1
                        mc_event["pass_through_after_retry_limit"] = True
                        mc_event["entered_transcript"] = True
                        transcript.append(
                            {
                                "turn": turn,
                                "speaker": "generic_agent",
                                "content": render_mc_question_display(parsed, interaction_mode),
                                "structured": parsed,
                                "detector_pass_through_after_retry_limit": True,
                                "detector_pass_through_reason": detector_result["action"],
                            }
                        )
                        agent_messages.append({"role": "assistant", "content": agent_reply})
                        agent_payload = parsed
                        agent_display = render_mc_question_display(parsed, interaction_mode)
                    else:
                        agent_max_retries_exceeded = True
                    break
                if detector_feedback_mode == "visible":
                    agent_messages.append({"role": "assistant", "content": agent_reply})
                    agent_messages.append(
                        {
                            "role": "user",
                            "content": build_agent_protocol_feedback(
                                interaction_mode, detector_result=detector_result
                            ),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    agent_retry_feedback = build_minimal_agent_retry_feedback(interaction_mode)
                agent_retry_total += 1
                continue

            agent_payload = parsed
            agent_display = render_mc_question_display(parsed, interaction_mode)
            mc_event["accepted"] = True
            mc_event["entered_transcript"] = True
            transcript.append(
                {
                    "turn": turn,
                    "speaker": "generic_agent",
                    "content": agent_display,
                    "structured": parsed,
                }
            )
            agent_messages.append({"role": "assistant", "content": agent_reply})
            break

        if agent_max_retries_exceeded:
            protocol_failed = True
            if not protocol_failure_type:
                protocol_failure_type = "agent_max_retries_exceeded"
            protocol_failure_reason = f"Exceeded {MAX_AGENT_RETRIES_PER_TURN} retries in turn {turn}"
            break
        if completed:
            break

        user_question_text = agent_display if interaction_mode != "open" else agent_reply
        simulator_messages.append(
            {
                "role": "user",
                "content": render_choice_user_message(agent_payload, interaction_mode)
                if interaction_mode != "open" and agent_payload
                else user_question_text,
            }
        )
        user_retry = 0
        user_retry_feedback: str | None = None
        while True:
            user_call_messages = simulator_messages
            if detector_feedback_mode == "minimal" and user_retry_feedback:
                user_call_messages = simulator_messages + [{"role": "user", "content": user_retry_feedback}]
            simulator_reply = user_client.complete(user_call_messages).content

            if interaction_mode == "open":
                simulator_messages.append({"role": "assistant", "content": simulator_reply})
                transcript.append({"turn": turn, "speaker": "user_simulator", "content": simulator_reply})
                agent_messages.append({"role": "user", "content": "Business user response:\n\n" + simulator_reply})
                break

            try:
                sim_structured = normalize_choice_simulator_result(
                    extract_json_object(simulator_reply), interaction_mode
                )
            except Exception as exc:
                user_format_error_count += 1
                answer_event = {
                    "event_type": "user_format_error",
                    "turn": turn,
                    "question": user_question_text,
                    "business_user_response": simulator_reply,
                    "detector_error": str(exc),
                    "retry_count": user_retry,
                    "accepted": False,
                    "entered_transcript": False,
                    "feedback_mode": detector_feedback_mode,
                }
                answer_scope_events.append(answer_event)
                user_retry += 1
                if user_retry > MAX_USER_RETRIES_PER_TURN:
                    answer_event["retry_limit_exceeded"] = True
                    answer_event["non_pass_through_reason"] = "Unparseable User JSON cannot drive a choice turn."
                    user_max_retries_exceeded = True
                    protocol_failure_type = "user_format_invalid"
                    break
                if detector_feedback_mode == "visible":
                    simulator_messages.append({"role": "assistant", "content": simulator_reply})
                    simulator_messages.append(
                        {
                            "role": "user",
                            "content": build_user_format_feedback(str(exc)),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    user_retry_feedback = build_minimal_user_format_retry_feedback()
                user_retry_total += 1
                continue

            agent_visible_choice = build_agent_visible_choice(sim_structured)
            natural_reply = render_choice_response_for_agent(agent_visible_choice)
            user_choice_audit_events.append(
                {
                    "event_type": "user_choice_rationale",
                    "turn": turn,
                    "interaction_mode": interaction_mode,
                    "question": (agent_payload or {}).get("question", user_question_text),
                    "options": list((agent_payload or {}).get("options", [])),
                    "choice": sim_structured["choice"],
                    "rationale": sim_structured["rationale"],
                    "match_status": sim_structured["match_status"],
                    "comment": sim_structured.get("comment", ""),
                    "agent_visible_structured": agent_visible_choice,
                    "agent_visible_response": natural_reply,
                    "audit_fields_entered_agent_context": False,
                    "raw_user_simulator_response": simulator_reply,
                }
            )
            should_audit_answer = monitor_user_answers and should_check_answer_scope(
                interaction_mode, sim_structured
            )
            if should_audit_answer:
                answer_detector_user_message = render_d_comment_scope_message(
                    agent_payload or {},
                    sim_structured,
                    case.get("hidden_slots", []),
                )
                answer_detector_messages = [
                    {"role": "system", "content": prompts.answer_detector_prompt},
                    {"role": "user", "content": answer_detector_user_message},
                ]
                answer_detector_raw = detector_client.complete(answer_detector_messages).content
                answer_scope_detector_call_count += 1
                answer_event = {
                    "event_type": "answer_scope_check",
                    "turn": turn,
                    "question": (agent_payload or {}).get("question", user_question_text),
                    "business_user_response": sim_structured["comment"],
                    "naturalized_response": natural_reply,
                    "structured_response": agent_visible_choice,
                    "detector_raw": answer_detector_raw,
                    "retry_count": 0,
                    "accepted": True,
                    "entered_transcript": True,
                    "passive_audit_only": True,
                }
                try:
                    answer_result = normalize_answer_scope_result(extract_json_object(answer_detector_raw))
                    answer_event["detector_result"] = answer_result
                    answer_event["scope_violation_count"] = answer_result["scope_violation_count"]
                    answer_event["scope_violations"] = answer_result["scope_violations"]
                    answer_event["independent_answered_fact_count"] = answer_result[
                        "independent_answered_fact_count"
                    ]
                    answer_event["disclosed_hidden_slot_ids"] = answer_result[
                        "disclosed_hidden_slot_ids"
                    ]
                    answer_event["disclosed_hidden_slot_count"] = answer_result[
                        "disclosed_hidden_slot_count"
                    ]
                    answer_independent_fact_count += answer_result["independent_answered_fact_count"]
                    disclosed_count = answer_result["disclosed_hidden_slot_count"]
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

            # rationale/match_status 只落审计文件，不回灌给后续 User Simulator，也不进入 Agent/Judge 对话。
            simulator_messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(agent_visible_choice, ensure_ascii=False),
                }
            )
            transcript.append(
                {
                    "turn": turn,
                    "speaker": "user_simulator",
                    "content": natural_reply,
                    "structured": agent_visible_choice,
                }
            )
            agent_messages.append({"role": "user", "content": "Business user response:\n\n" + natural_reply})
            break

        if user_max_retries_exceeded:
            protocol_failed = True
            if not protocol_failure_type:
                protocol_failure_type = "user_max_retries_exceeded"
            protocol_failure_reason = f"Exceeded {MAX_USER_RETRIES_PER_TURN} retries in turn {turn}"
            break

        time.sleep(0.2)

    judge_user_message = render_judge_case(case) + "\n\n# Full Transcript\n\n" + render_transcript(transcript)
    judge_messages = [
        {"role": "system", "content": prompts.judge_prompt},
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
                        "Your previous judge output was not valid JSON. Return only one valid JSON object "
                        "that follows the required schema, without markdown fences."
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
    choice_metrics = summarize_choice_metrics(transcript)
    choice_audit_metrics = summarize_user_choice_audit(
        user_choice_audit_events, interaction_mode
    )
    question_audit = summarize_question_audit(detector_events)

    (run_dir / "initial_brief.txt").write_text(case["initial_brief"]["content"], encoding="utf-8")
    (run_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (run_dir / "detector_events.json").write_text(
        json.dumps(detector_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "answer_scope_events.json").write_text(
        json.dumps(answer_scope_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / USER_CHOICE_AUDIT_FILENAME).write_text(
        json.dumps(user_choice_audit_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "judge_prompt_user_message.md").write_text(judge_user_message, encoding="utf-8")
    (run_dir / "judge_raw.txt").write_text(judge_raw, encoding="utf-8")
    judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")

    stat = {
        "pipeline_version": "choice",
        "pipeline_mode": pipeline_mode,
        "interaction_mode": interaction_mode,
        "detector_feedback_mode": detector_feedback_mode,
        "retry_limit_behavior": retry_limit_behavior,
        "monitor_user_answers": monitor_user_answers,
        "answer_scope_policy": "mc_d_d_comment_only_passive_opt_in",
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
        "answer_scope_detector_call_count": answer_scope_detector_call_count,
        "answer_scope_rejection_count": answer_scope_rejection_count,
        "answer_scope_error_count": answer_scope_error_count,
        "answer_independent_fact_count": answer_independent_fact_count,
        "answer_disclosed_hidden_slot_count": answer_disclosed_hidden_slot_count,
        "answer_multi_hidden_slot_disclosure_count": answer_multi_hidden_slot_disclosure_count,
        "user_format_error_count": user_format_error_count,
        **question_audit,
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
        "choice_metrics": choice_metrics,
        "choice_audit_metrics": choice_audit_metrics,
        "agent_usage": agent_client.total_usage,
        "detector_usage": detector_client.total_usage,
        "user_usage": user_client.total_usage,
        "judge_usage": judge_client.total_usage,
        "agent_estimated_cost_usd": agent_client.total_estimated_cost_usd,
        "detector_estimated_cost_usd": detector_client.total_estimated_cost_usd,
        "user_estimated_cost_usd": user_client.total_estimated_cost_usd,
        "judge_estimated_cost_usd": judge_client.total_estimated_cost_usd,
        "agent_retry_total": agent_retry_total,
        "user_retry_total": user_retry_total,
        "agent_max_retries_exceeded": agent_max_retries_exceeded,
        "user_max_retries_exceeded": user_max_retries_exceeded,
        "agent_retry_limit_pass_through_count": agent_retry_limit_pass_through_count,
        "rejected_attempt_in_transcript": bool(agent_retry_limit_pass_through_count),
        "run_dir": str(run_dir),
    }
    stat["estimated_cost_usd"] = (
        stat["agent_estimated_cost_usd"]
        + stat["detector_estimated_cost_usd"]
        + stat["user_estimated_cost_usd"]
        + stat["judge_estimated_cost_usd"]
    )
    write_stat_files(stat, run_dir)
    return stat


# [模块目标]：按 mc/mc_d 与 Agent profile 聚合质量、效率、Detector、泄露、还原率和成本指标。
# [输入输出]：输入全部 run 统计；输出机器可读 summary，并写入 summary_choice_eval.json。
def summarize(stats: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for stat in stats:
        key = f"{stat['interaction_mode']}::{stat['agent_profile']}"
        group = groups.setdefault(
            key,
            {
                "interaction_mode": stat["interaction_mode"],
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
                "answer_scope_detector_call_sum": 0,
                "answer_scope_rejection_sum": 0,
                "answer_scope_error_sum": 0,
                "answer_independent_fact_sum": 0,
                "answer_disclosed_hidden_slot_sum": 0,
                "answer_multi_hidden_slot_disclosure_sum": 0,
                "agent_question_turn_sum": 0,
                "agent_atomic_question_sum": 0,
                "agent_multi_question_turn_sum": 0,
                "agent_retry_total_sum": 0,
                "user_retry_total_sum": 0,
                "agent_retry_limit_pass_through_sum": 0,
                "d_selection_sum": 0,
                "choice_question_turn_sum": 0,
                "choice_audit_event_sum": 0,
                "choice_exact_match_sum": 0,
                "choice_acceptable_match_sum": 0,
                "choice_no_match_sum": 0,
                "choice_undetermined_sum": 0,
                "mc_forced_choice_sum": 0,
                "core_exact_restore_sum": 0,
                "core_eligible_run_count": 0,
                "all_slot_exact_restore_sum": 0,
                "core_slot_sum": 0,
                "p2_slot_sum": 0,
                "core_unresolved_slot_sum": 0,
                "p2_unresolved_slot_sum": 0,
                "silent_assumption_sum": 0,
                "stopping_status_mismatch_sum": 0,
                "estimated_cost_usd": 0.0,
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
        group["answer_scope_detector_call_sum"] += int(stat.get("answer_scope_detector_call_count", 0) or 0)
        group["answer_scope_rejection_sum"] += int(stat.get("answer_scope_rejection_count", 0) or 0)
        group["answer_scope_error_sum"] += int(stat.get("answer_scope_error_count", 0) or 0)
        group["answer_independent_fact_sum"] += int(stat.get("answer_independent_fact_count", 0) or 0)
        group["answer_disclosed_hidden_slot_sum"] += int(
            stat.get("answer_disclosed_hidden_slot_count", 0) or 0
        )
        group["answer_multi_hidden_slot_disclosure_sum"] += int(
            stat.get("answer_multi_hidden_slot_disclosure_count", 0) or 0
        )
        group["agent_question_turn_sum"] += int(stat.get("agent_question_turn_count", 0) or 0)
        group["agent_atomic_question_sum"] += int(stat.get("agent_atomic_question_count", 0) or 0)
        group["agent_multi_question_turn_sum"] += int(
            stat.get("agent_multi_question_turn_count", 0) or 0
        )
        group["agent_retry_total_sum"] += int(stat.get("agent_retry_total", 0) or 0)
        group["user_retry_total_sum"] += int(stat.get("user_retry_total", 0) or 0)
        group["agent_retry_limit_pass_through_sum"] += int(stat.get("agent_retry_limit_pass_through_count", 0) or 0)
        choice_metrics = stat.get("choice_metrics") or {}
        group["d_selection_sum"] += int(choice_metrics.get("d_selection_count", 0) or 0)
        group["choice_question_turn_sum"] += int(choice_metrics.get("mc_question_turns", 0) or 0)
        choice_audit_metrics = stat.get("choice_audit_metrics") or {}
        status_counts = choice_audit_metrics.get("choice_match_status_counts") or {}
        group["choice_audit_event_sum"] += int(
            choice_audit_metrics.get("choice_audit_event_count", 0) or 0
        )
        group["choice_exact_match_sum"] += int(status_counts.get("exact_match", 0) or 0)
        group["choice_acceptable_match_sum"] += int(
            status_counts.get("acceptable_match", 0) or 0
        )
        group["choice_no_match_sum"] += int(status_counts.get("no_match", 0) or 0)
        group["choice_undetermined_sum"] += int(status_counts.get("undetermined", 0) or 0)
        group["mc_forced_choice_sum"] += int(
            choice_audit_metrics.get("mc_forced_choice_count", 0) or 0
        )
        if int(stat.get("core_slot_count", 0) or 0) > 0:
            group["core_eligible_run_count"] += 1
        group["core_exact_restore_sum"] += 1 if stat.get("core_exact_restore") else 0
        group["all_slot_exact_restore_sum"] += 1 if stat.get("all_slot_exact_restore") else 0
        group["core_slot_sum"] += int(stat.get("core_slot_count", 0) or 0)
        group["p2_slot_sum"] += int(stat.get("p2_slot_count", 0) or 0)
        group["core_unresolved_slot_sum"] += int(stat.get("core_unresolved_slot_count", 0) or 0)
        group["p2_unresolved_slot_sum"] += int(stat.get("p2_unresolved_slot_count", 0) or 0)
        group["silent_assumption_sum"] += int(stat.get("silent_assumption_count", 0) or 0)
        group["stopping_status_mismatch_sum"] += 1 if stat.get("stopping_status_mismatch") else 0
        group["estimated_cost_usd"] += float(stat.get("estimated_cost_usd", 0) or 0)

    summary = {"profiles": {}, "total_estimated_cost_usd": 0.0}
    for key, group in groups.items():
        runs = max(1, group["runs"])
        core_eligible_runs = group["core_eligible_run_count"]
        score_count = max(1, group["weighted_slot_score_count"])
        question_turns = max(1, group["agent_question_turn_sum"])
        choice_audit_events = group["choice_audit_event_sum"]
        weighted_micro = (
            group["earned_weight_sum"] / group["total_weight_sum"]
            if group["total_weight_sum"]
            else None
        )
        summary["profiles"][key] = {
            "interaction_mode": group["interaction_mode"],
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
                "AnswerMultiHiddenSlotDisclosurePerRun": group[
                    "answer_multi_hidden_slot_disclosure_sum"
                ]
                / runs,
                "AvgAgentRetriesPerRun": group["agent_retry_total_sum"] / runs,
                "AvgUserRetriesPerRun": group["user_retry_total_sum"] / runs,
                "AgentRetryLimitPassThroughsPerRun": group["agent_retry_limit_pass_through_sum"] / runs,
                "DSelectionCount": group["d_selection_sum"],
                "DSelectionRate": (
                    group["d_selection_sum"] / group["choice_question_turn_sum"]
                    if group["choice_question_turn_sum"]
                    else None
                ),
                "ChoiceAuditEventCount": choice_audit_events,
                "ChoiceExactMatchRate": (
                    group["choice_exact_match_sum"] / choice_audit_events
                    if choice_audit_events
                    else None
                ),
                "ChoiceAcceptableMatchRate": (
                    group["choice_acceptable_match_sum"] / choice_audit_events
                    if choice_audit_events
                    else None
                ),
                "ChoiceNoMatchCount": group["choice_no_match_sum"],
                "ChoiceNoMatchRate": (
                    group["choice_no_match_sum"] / choice_audit_events
                    if choice_audit_events
                    else None
                ),
                "ChoiceUndeterminedRate": (
                    group["choice_undetermined_sum"] / choice_audit_events
                    if choice_audit_events
                    else None
                ),
                "MCForcedChoiceCount": group["mc_forced_choice_sum"],
                "MCForcedChoiceRate": (
                    group["mc_forced_choice_sum"] / choice_audit_events
                    if choice_audit_events and group["interaction_mode"] == "mc"
                    else None
                ),
                "SilentAssumptionsPerRun": group["silent_assumption_sum"] / runs,
                "StoppingStatusMismatchRate": group["stopping_status_mismatch_sum"] / runs,
            },
            "estimated_cost_usd": group["estimated_cost_usd"],
        }
        summary["total_estimated_cost_usd"] += group["estimated_cost_usd"]
    (output_root / "summary_choice_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_markdown_report(summary: dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Choice Pipeline Evaluation Report",
        "",
        "- pipeline version: `choice`",
        "- user choice audit: `choice + user-written rationale + match_status`; only Agent-visible fields enter transcript",
        f"- interaction modes: `{', '.join(args.interaction_modes)}`",
        f"- case ids: `{', '.join(args.case_ids) if args.case_ids else '(all in toml_dirs)'}`",
        f"- toml dirs: `{', '.join(args.toml_dirs)}`",
        f"- k: `{args.k}`",
        f"- max turns: `{args.max_turns}`",
        f"- detector feedback mode: `{args.detector_feedback_mode}`",
        f"- retry limit behavior: `{args.retry_limit_behavior}`",
        f"- answer-scope audit enabled: `{args.monitor_user_answers}`",
        "- answer-scope policy: `mc_d_d_comment_only_passive_opt_in`",
        "",
        "| 分组标签 (mode::profile) | 样本量 (runs) | 全要素还原率 (all restore) | 核心要素还原率 (core restore) | 加权要素分数 (WSS, weighted macro) | 平均提问轮数 (avg turns) | 正常收尾率 (ready) | 协议失败率 (protocol fail) | 平均原子问题数 (atomic Q/run) | 多问率 (multi-Q rate) | 回答范围审计调用次数 (scope calls) | 泄露的隐藏要素数 (disclosed slots) | 选项无匹配率 (no-match rate) | MC 强制选择率 (MC forced rate) | D 选项被选率（仅 mc_d 有意义） (D rate) | 估算美元成本 (est. cost) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["profiles"].items():
        m = item["metrics"]
        d_rate = f"{m['DSelectionRate']:.3f}" if m["DSelectionRate"] is not None else "n/a"
        no_match_rate = (
            f"{m['ChoiceNoMatchRate']:.3f}" if m["ChoiceNoMatchRate"] is not None else "n/a"
        )
        forced_rate = (
            f"{m['MCForcedChoiceRate']:.3f}" if m["MCForcedChoiceRate"] is not None else "n/a"
        )
        core_exact_rate = (
            f"{m['CoreExactRestoreRate']:.3f}"
            if m["CoreExactRestoreRate"] is not None
            else "n/a"
        )
        lines.append(
            f"| {key} | {item['runs']} | {m['AllSlotExactRestoreRate']:.3f} | "
            f"{core_exact_rate} | {m['WeightedSlotScore_macro']:.3f} | "
            f"{m['AverageTurns']:.2f} | {m['ReadyToModelRate']:.3f} | {m['ProtocolFailureRate']:.3f} | "
            f"{m['AtomicQuestionsPerRun']:.2f} | {m['MultiQuestionTurnRate']:.3f} | "
            f"{m['AnswerScopeDetectorCallsPerRun']:.2f} | "
            f"{m['AnswerDisclosedHiddenSlotsPerRun']:.2f} | "
            f"{no_match_rate} | {forced_rate} | "
            f"{d_rate} | "
            f"{item['estimated_cost_usd']:.4f} |"
        )
    lines.extend(["", f"Total estimated cost: `${summary['total_estimated_cost_usd']:.4f}`.", ""])
    (output_root / "choice_eval_report.md").write_text("\n".join(lines), encoding="utf-8")


# [模块目标]：集中定义 Choice Pipeline 命令行接口，保证脚本、测试和人工运行使用同一默认口径。
# [输入输出]：无业务输入；返回配置完成的 argparse 解析器。
def build_argument_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_prompts_dir = script_dir / "prompts"
    parser = argparse.ArgumentParser(
        description="Choice interaction pipeline (mc / mc_d + hidden user rationale audit)."
    )
    parser.add_argument(
        "--toml_dirs",
        "--toml_dir",
        dest="toml_dirs",
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
    parser.add_argument("--judge_prompt_path", default=str(CANONICAL_JUDGE_PROMPT_PATH))
    parser.add_argument(
        "--interaction_modes",
        nargs="+",
        default=["mc_d"],
        choices=list(INTERACTION_MODES),
    )
    parser.add_argument(
        "--case_ids",
        nargs="+",
        default=None,
        help=(
            "按题号过滤；在 --toml_dirs 的合并结果内做白名单匹配。"
            "支持单个编号（--case_ids 8 或 008）、多个编号（--case_ids 8 57 70），"
            "也支持完整 case_id（--case_ids orclarify_008）。"
            "或不传则跑 --toml_dirs 下全部 toml。"
        ),
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--agent_profiles", nargs="+", default=["generic_agent"])
    parser.add_argument("--detector_profile", default="detector")
    parser.add_argument("--user_profile", default="user_simulator")
    parser.add_argument("--judge_profile", default="judge")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--detector_temperature", type=float, default=0.0)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--prompts_dir", default=str(default_prompts_dir))
    parser.add_argument("--max_agent_retries", type=int, default=0)
    parser.add_argument("--max_user_retries", type=int, default=0)
    parser.add_argument(
        "--detector_feedback_mode",
        choices=["visible", "minimal", "none"],
        default="none",
        help="Feedback used only during an enabled format/protocol retry; none keeps retries silent.",
    )
    parser.add_argument(
        "--retry_limit_behavior",
        choices=["pass_through", "protocol_failed"],
        default="pass_through",
        help="pass_through continues after retry limit; protocol_failed stops the current case after retry limit.",
    )
    parser.add_argument(
        "--monitor_user_answers",
        action="store_true",
        help="Passively audit mc_d D comments for scope and hidden-slot disclosure; never rewrite the answer.",
    )
    return parser


def determine_pipeline_mode(args: argparse.Namespace) -> str:
    intervention_enabled = (
        args.max_agent_retries > 0
        or args.detector_feedback_mode != "none"
        or args.retry_limit_behavior == "protocol_failed"
    )
    mode = (
        "choice_agent_protocol_intervention"
        if intervention_enabled
        else "choice_passive_question_audit"
    )
    if args.monitor_user_answers:
        mode += "_with_answer_audit"
    return mode


def make_default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"choice_mc_mcd_{timestamp}"


def print_milestone(stage: str, status: str) -> None:
    print(f"| {stage} | {status} |", flush=True)


# [模块目标]：在一次 Choice 实验完成后重建固定审核页，使“最新结果”无需手工指定 output_dir 即可查看。
# [输入输出]：无业务输入；成功时返回 HTML 路径，失败时返回错误说明，但不推翻已经完成的实验结果。
# [LLM 交互]：不调用 LLM，只扫描本地 runs 产物并生成离线 HTML。
def refresh_choice_review_safely() -> tuple[Path | None, str]:
    try:
        import importlib.util

        builder_path = Path(__file__).resolve().parent / "html" / "build_choice_review.py"
        spec = importlib.util.spec_from_file_location("choice_review_builder", builder_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载审核页生成脚本: {builder_path}")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)

        output_path = builder.write_history_review(builder.DEFAULT_RUNS_ROOT, builder.DEFAULT_OUTPUT_PATH)
        return output_path, ""
    except Exception as exc:
        return None, str(exc)


# [模块目标]：加载 OR-Clarify cases，依次运行 mc/mc_d 两臂并生成统一审计报告。
# [输入输出]：输入来自命令行；输出写入时间戳目录，并按约 10% 粒度汇报进度。
def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    parser = build_argument_parser()
    args = parser.parse_args()
    args.pipeline_version = "choice"
    if args.max_agent_retries < 0 or args.max_user_retries < 0:
        parser.error("retry counts must be zero or greater")
    args.pipeline_mode = determine_pipeline_mode(args)

    global MAX_AGENT_RETRIES_PER_TURN, MAX_USER_RETRIES_PER_TURN
    MAX_AGENT_RETRIES_PER_TURN = args.max_agent_retries
    MAX_USER_RETRIES_PER_TURN = args.max_user_retries

    case_ids = args.case_ids
    cases, missing = resolve_case_selection(
        read_cases([Path(path) for path in args.toml_dirs]),
        case_ids,
    )
    if missing:
        raise SystemExit(
            f"--case_ids 指定的题号在 --toml_dirs 内未找到: {missing} "
            f"（支持 70、070、orclarify_070；toml_dirs={args.toml_dirs}）"
        )
    if args.limit is not None:
        cases = cases[: args.limit]
    output_root = Path(args.output_dir) if args.output_dir else make_default_output_dir()
    args.output_dir = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    total_runs = len(args.interaction_modes) * len(args.agent_profiles) * len(cases) * args.k
    print("| 里程碑 | 状态 |", flush=True)
    print("|---|---|", flush=True)
    print_milestone("数据加载完成", f"{len(cases)} 个 case，计划 {total_runs} 次运行")
    print_milestone("输出目录已创建", str(output_root))

    stats: list[dict[str, Any]] = []
    completed_runs = 0
    next_progress_percent = 10
    for interaction_mode in args.interaction_modes:
        prompts = load_prompt_bundle(Path(args.prompts_dir), interaction_mode, Path(args.judge_prompt_path))
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
                        interaction_mode=interaction_mode,
                        agent_temperature=args.agent_temperature,
                        detector_temperature=args.detector_temperature,
                        user_temperature=args.user_temperature,
                        judge_temperature=args.judge_temperature,
                        max_turns=args.max_turns,
                        detector_feedback_mode=args.detector_feedback_mode,
                        retry_limit_behavior=args.retry_limit_behavior,
                        monitor_user_answers=args.monitor_user_answers,
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
    print_milestone("评估汇总完成", f"报告：{output_root / 'choice_eval_report.md'}")
    print_milestone("估算总成本", f"${summary['total_estimated_cost_usd']:.4f}")
    review_path, review_error = refresh_choice_review_safely()
    if review_path:
        print_milestone("可视化审核页更新", str(review_path))
    else:
        print_milestone("可视化审核页更新跳过", review_error)


if __name__ == "__main__":
    main()
