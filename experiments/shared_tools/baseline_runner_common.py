from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# Locate either the anonymized artifact root or the original research repository root.
def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "PACKAGE_ROOT.marker").exists() or (
            (candidate / "AGENTS.md").exists() and (candidate / "doc-wiki.md").exists()
        ):
            return candidate
    raise RuntimeError("Could not find repo root from baseline_runner_common.py")


REPO_ROOT = find_repo_root()
PIPELINE_PATH = REPO_ROOT / "experiments" / "evaluation_protocol" / "run_pipeline.py"


# [模块目标]：复用已经稳定的本项目评测基础设施，避免外部 baseline 重写 case 读取、DeepSeek 调用和 judge 统计。
# [输入输出]：读取 evaluation_protocol 的 Python 文件；返回一个模块对象，供本文件调用其中的工具函数。
def load_reference_pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("aior_evaluation_protocol_reference", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reference pipeline from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REF = load_reference_pipeline()
DEFAULT_TOML_DIRS = (
    Path(os.environ.get("BENCHMARK_CASES_DIR", str(REPO_ROOT / "data"))),
)


@dataclass
class AgentAction:
    kind: str
    content: str
    protocol_note: str = ""


@dataclass
class ChatResult:
    content: str
    usage: dict[str, int]
    estimated_cost_usd: float
    recovered_from_reasoning_content: bool = False


# [模块目标]：为外部 baseline 提供只走 DeepSeek 的 LLM 调用器，并处理 dsv4pro 偶发 content 为空的问题。
# [输入输出]：输入角色名和温度；complete 输入 messages，输出模型文本、token 和估算成本。
# [LLM 交互]：所有请求都发往 DEEPSEEK_BASE_URL/chat/completions，使用 DEEPSEEK_API_KEY，不读取 OPENAI_*。
class BaselineChatClient:
    def __init__(self, profile_name: str, temperature: float):
        self.profile_name = profile_name
        self.model = REF.resolve_model_name(profile_name)
        self.temperature = temperature
        self.total_usage: dict[str, int] = {}
        self.total_estimated_cost_usd = 0.0
        self.reasoning_content_recovery_count = 0

    def _api_settings(self) -> tuple[str, str]:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Put it in repo root .env or the process environment.")
        return base_url, api_key

    def complete(self, messages: list[dict[str, str]], timeout: int = 180, max_retries: int = 12) -> ChatResult:
        retry = 0
        while True:
            try:
                return self._complete_once(messages, timeout)
            except Exception as exc:
                retry += 1
                if retry > max_retries:
                    raise RuntimeError(f"Maximum retries exceeded for {self.model}: {safe_error_message(exc)}") from exc
                delay = min(45.0, 1.0 * (2 ** (retry - 1)))
                print(
                    f"Error: {type(exc).__name__}: {safe_error_message(exc)}. Retrying in {delay:.2f}s (attempt {retry}/{max_retries})",
                    flush=True,
                )
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

        recovered = False
        if not content:
            reasoning_content = str(message.get("reasoning_content", "") or "").strip()
            content = extract_protocol_from_reasoning_content(reasoning_content)
            recovered = bool(content)
            if recovered:
                self.reasoning_content_recovery_count += 1
        if not content:
            raise RuntimeError("Empty content returned and no protocol line could be recovered.")

        usage = REF.normalize_usage(data.get("usage") or {})
        cost = REF.estimate_cost(self.model, usage)
        REF.add_usage(self.total_usage, usage)
        self.total_estimated_cost_usd += cost
        return ChatResult(content=content, usage=usage, estimated_cost_usd=cost, recovered_from_reasoning_content=recovered)


def safe_error_message(exc: Exception) -> str:
    return str(exc).encode("ascii", errors="backslashreplace").decode("ascii")


def extract_protocol_from_reasoning_content(reasoning_content: str) -> str:
    if not reasoning_content:
        return ""
    ready_idx = reasoning_content.upper().rfind("READY_TO_MODEL")
    if ready_idx >= 0:
        return reasoning_content[ready_idx:].strip()
    question_idx = reasoning_content.upper().rfind("QUESTION:")
    if question_idx >= 0:
        return reasoning_content[question_idx:].strip()
    return ""


# [模块目标]：把不同外部 baseline 的自然语言输出压到统一 ASK/READY 运行协议，但不额外引入 detector 或内部纠偏机制。
# [输入输出]：输入 agent 原始回复；输出 question、ready 或 invalid 三类动作，以及必要的协议备注。
def parse_agent_action(raw_text: str) -> AgentAction:
    text = raw_text.strip()
    if not text:
        return AgentAction("invalid", text, "empty_agent_response")

    try:
        parsed = REF.extract_json_object(text)
        action = str(parsed.get("action", "")).strip().upper()
        if action == "ASK":
            question = str(parsed.get("question", "")).strip()
            if question:
                return AgentAction("question", f"QUESTION: {question}", "json_ask_normalized")
        if action in {"READY", "READY_TO_MODEL"}:
            summary = str(parsed.get("summary", "")).strip()
            content = "READY_TO_MODEL" + (f"\n\n{summary}" if summary else "")
            return AgentAction("ready", content, "json_ready_normalized")
    except Exception:
        pass

    upper = text.upper()
    if upper.startswith("READY_TO_MODEL"):
        return AgentAction("ready", text)
    if "[INTERVIEW_COMPLETE]" in upper:
        cleaned = re.sub(r"\[INTERVIEW_COMPLETE\]", "", text, flags=re.IGNORECASE).strip()
        content = "READY_TO_MODEL" + (f"\n\n{cleaned}" if cleaned else "")
        return AgentAction("ready", content, "orpilot_interview_complete_normalized")
    if upper.startswith("QUESTION:"):
        return AgentAction("question", text)

    if "?" in text or "？" in text:
        return AgentAction("question", text, "question_mark_pass_through")

    return AgentAction("invalid", text, "unrecognized_agent_protocol")


# [模块目标]：把原方法 prompt、当前 case 和历史对话交给外部 baseline agent，并让统一 User Simulator 回答。
# [输入输出]：输入一个 TOML case、method 名称和运行配置；输出 statistics，并写出 transcript、judge_result 等四类产物。
# [LLM 交互]：Agent 使用各 baseline 自己的 prompt；User/Judge 使用本项目统一 prompt。三者都通过 DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY 调 deepseek-v4-pro。
def run_interaction(
    case: dict[str, Any],
    method_name: str,
    method_label: str,
    agent_prompt: str,
    user_prompt: str,
    judge_prompt: str,
    run_index: int,
    output_root: Path,
    agent_profile: str,
    user_profile: str,
    judge_profile: str,
    agent_temperature: float,
    user_temperature: float,
    judge_temperature: float,
    max_turns: int,
) -> dict[str, Any]:
    case_id = case["_case_id"]
    run_dir = output_root / method_name / f"run_{run_index:02d}" / case_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "statistics.json"
    judge_path = run_dir / "judge_result.json"
    if stats_path.exists() and judge_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))

    agent_client = BaselineChatClient(agent_profile, agent_temperature)
    user_client = BaselineChatClient(user_profile, user_temperature)
    judge_client = BaselineChatClient(judge_profile, judge_temperature)

    first_user_message = (
        "Here is the user's initial request. Interview the user if clarification is needed.\n\n"
        + case["initial_brief"]["content"]
    )
    simulator_system_prompt = user_prompt + "\n\n" + REF.render_user_simulator_case(case)

    agent_messages = [
        {"role": "system", "content": agent_prompt},
        {"role": "user", "content": first_user_message},
    ]
    simulator_messages = [{"role": "system", "content": simulator_system_prompt}]

    transcript: list[dict[str, Any]] = []
    method_events: list[dict[str, Any]] = []
    completed = False
    protocol_failed = False
    protocol_failure_type = ""
    protocol_failure_reason = ""
    final_agent_text = ""
    agent_question_turn_count = 0
    agent_atomic_question_count = 0
    agent_multi_question_turn_count = 0
    attempted_turn_count = 0

    for turn in range(1, max_turns + 1):
        attempted_turn_count = turn
        agent_reply = agent_client.complete(agent_messages).content
        final_agent_text = agent_reply
        parsed_action = parse_agent_action(agent_reply)
        method_events.append(
            {
                "turn": turn,
                "raw_agent_response": agent_reply,
                "normalized_kind": parsed_action.kind,
                "normalized_content": parsed_action.content,
                "protocol_note": parsed_action.protocol_note,
            }
        )

        if parsed_action.kind == "ready":
            transcript.append({"turn": turn, "speaker": "generic_agent", "content": parsed_action.content})
            completed = True
            final_agent_text = parsed_action.content
            break

        if parsed_action.kind == "invalid":
            transcript.append(
                {
                    "turn": turn,
                    "speaker": "generic_agent",
                    "content": parsed_action.content,
                    "protocol_note": parsed_action.protocol_note,
                }
            )
            protocol_failed = True
            protocol_failure_type = parsed_action.protocol_note or "agent_protocol_invalid"
            protocol_failure_reason = "Agent response was neither a clear question nor READY_TO_MODEL."
            break

        transcript.append(
            {
                "turn": turn,
                "speaker": "generic_agent",
                "content": parsed_action.content,
                "protocol_note": parsed_action.protocol_note,
            }
        )
        agent_question_turn_count += 1
        agent_atomic_question_count += 1
        agent_messages.append({"role": "assistant", "content": parsed_action.content})

        simulator_messages.append({"role": "user", "content": parsed_action.content})
        simulator_reply = user_client.complete(simulator_messages).content
        simulator_messages.append({"role": "assistant", "content": simulator_reply})
        transcript.append({"turn": turn, "speaker": "user_simulator", "content": simulator_reply})
        agent_messages.append({"role": "user", "content": "Business user response:\n\n" + simulator_reply})
        time.sleep(0.2)

    judge_user_message = REF.render_judge_case(case) + "\n\n# Full Transcript\n\n" + REF.render_transcript(transcript)
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
            judge_result = REF.extract_json_object(judge_raw)
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
                        "that follows the required schema. Do not include markdown fences."
                    ),
                },
            ]
    assert judge_result is not None
    if judge_parse_errors:
        judge_result["judge_parse_errors_before_success"] = judge_parse_errors
    weighted_summary = REF.calculate_weighted_slot_score(judge_result.get("slot_scores", []))
    restoration_summary = REF.calculate_restoration_summary(judge_result.get("slot_scores", []))
    stopping_audit = REF.audit_stopping_behavior(
        judge_result.get("slot_scores", []),
        judge_result.get("stopping_behavior") or {},
        completed,
    )
    judge_result["weighted_summary"] = weighted_summary
    judge_result["restoration_summary"] = restoration_summary
    judge_result["stopping_consistency_audit"] = stopping_audit

    (run_dir / "initial_brief.txt").write_text(case["initial_brief"]["content"], encoding="utf-8")
    (run_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "transcript.md").write_text(REF.render_transcript(transcript), encoding="utf-8")
    (run_dir / "method_events.json").write_text(json.dumps(method_events, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "judge_prompt_user_message.md").write_text(judge_user_message, encoding="utf-8")
    (run_dir / "judge_raw.txt").write_text(judge_raw, encoding="utf-8")
    judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")

    stat = {
        "pipeline_version": "external_baseline_adapter",
        "pipeline_mode": "external_baseline_no_detector",
        "method_name": method_name,
        "method_label": method_label,
        "case_id": case_id,
        "run_index": run_index,
        "agent_profile": agent_profile,
        "agent_model": agent_client.model,
        "user_profile": user_profile,
        "user_model": user_client.model,
        "judge_profile": judge_profile,
        "judge_model": judge_client.model,
        "turn_count": max((int(row["turn"]) for row in transcript), default=0),
        "attempted_turn_count": attempted_turn_count,
        "max_turns": max_turns,
        "max_turn_hit": attempted_turn_count == max_turns and not completed,
        "transcript_rows": len(transcript),
        "completed_ready_to_model": completed,
        "protocol_failed": protocol_failed,
        "protocol_failure_type": protocol_failure_type or "none",
        "protocol_failure_reason": protocol_failure_reason,
        "agent_question_turn_count": agent_question_turn_count,
        "agent_atomic_question_count": agent_atomic_question_count,
        "agent_multi_question_turn_count": agent_multi_question_turn_count,
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
        "user_usage": user_client.total_usage,
        "judge_usage": judge_client.total_usage,
        "agent_reasoning_content_recovery_count": agent_client.reasoning_content_recovery_count,
        "user_reasoning_content_recovery_count": user_client.reasoning_content_recovery_count,
        "judge_reasoning_content_recovery_count": judge_client.reasoning_content_recovery_count,
        "agent_estimated_cost_usd": agent_client.total_estimated_cost_usd,
        "user_estimated_cost_usd": user_client.total_estimated_cost_usd,
        "judge_estimated_cost_usd": judge_client.total_estimated_cost_usd,
        "run_dir": str(run_dir),
    }
    stat["estimated_cost_usd"] = (
        stat["agent_estimated_cost_usd"]
        + stat["user_estimated_cost_usd"]
        + stat["judge_estimated_cost_usd"]
    )
    stats_path.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    return stat


# [模块目标]：把每个 case-run 的 statistics 合成为论文/组会能读的主指标表。
# [输入输出]：输入一组 statistics；输出 summary_external_baseline.json，同时返回摘要字典。
def summarize(stats: list[dict[str, Any]], output_root: Path, method_name: str, method_label: str) -> dict[str, Any]:
    runs = max(1, len(stats))
    earned = sum(float(s.get("earned_weight", 0) or 0) for s in stats)
    total = sum(float(s.get("total_weight", 0) or 0) for s in stats)
    weighted_values = [float(s["weighted_slot_score"]) for s in stats if s.get("weighted_slot_score") is not None]
    status_counts: dict[str, int] = {}
    rule_status_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    agent_usage: dict[str, int] = {}
    user_usage: dict[str, int] = {}
    judge_usage: dict[str, int] = {}
    for stat in stats:
        status = stat.get("stopping_status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        rule_status = stat.get("rule_based_stopping_status") or "unknown"
        rule_status_counts[rule_status] = rule_status_counts.get(rule_status, 0) + 1
        failure = stat.get("protocol_failure_type") or "none"
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
        REF.add_usage(agent_usage, stat.get("agent_usage", {}))
        REF.add_usage(user_usage, stat.get("user_usage", {}))
        REF.add_usage(judge_usage, stat.get("judge_usage", {}))

    max_turn_hit_count = sum(
        1
        for s in stats
        if bool(s.get("max_turn_hit"))
        or (
            s.get("max_turns") is not None
            and int(s.get("attempted_turn_count", 0) or 0) == int(s.get("max_turns", 0) or 0)
            and not bool(s.get("completed_ready_to_model"))
        )
    )
    core_eligible_runs = sum(
        1 for s in stats if int(s.get("core_slot_count", 0) or 0) > 0
    )

    summary = {
        "method_name": method_name,
        "method_label": method_label,
        "runs": len(stats),
        "metrics": {
            "WeightedSlotScore_macro": sum(weighted_values) / max(1, len(weighted_values)),
            "WeightedSlotScore_micro": earned / total if total else None,
            "CoreExactRestoreRate": (
                sum(1 for s in stats if s.get("core_exact_restore")) / core_eligible_runs
                if core_eligible_runs
                else None
            ),
            "CoreEligibleRunCount": core_eligible_runs,
            "AllSlotExactRestoreRate": sum(1 for s in stats if s.get("all_slot_exact_restore")) / runs,
            "ReadyToModelRate": sum(1 for s in stats if s.get("completed_ready_to_model")) / runs,
            "SilentAssumptionsPerRun": sum(int(s.get("silent_assumption_count", 0) or 0) for s in stats) / runs,
            "StoppingStatusMismatchRate": sum(1 for s in stats if s.get("stopping_status_mismatch")) / runs,
            "ProtocolFailureRate": sum(1 for s in stats if s.get("protocol_failed")) / runs,
            "ProtocolFailureTypeCounts": failure_counts,
            "MaxTurnHitRate": max_turn_hit_count / runs,
            "MaxTurnHitCount": max_turn_hit_count,
            "AverageTurns": sum(int(s.get("turn_count", 0) or 0) for s in stats) / runs,
            "AverageAttemptedTurns": sum(int(s.get("attempted_turn_count", 0) or 0) for s in stats) / runs,
            "AgentQuestionTurnsPerRun": sum(int(s.get("agent_question_turn_count", 0) or 0) for s in stats) / runs,
            "StoppingStatusCounts": status_counts,
            "RuleBasedStoppingStatusCounts": rule_status_counts,
            "Cost USD": sum(float(s.get("estimated_cost_usd", 0) or 0) for s in stats),
        },
        "agent_usage": agent_usage,
        "user_usage": user_usage,
        "judge_usage": judge_usage,
        "run_dirs": [s.get("run_dir") for s in stats],
    }
    (output_root / "summary_external_baseline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_markdown_report(summary: dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    m = summary["metrics"]
    core_exact_rate = (
        f"{m['CoreExactRestoreRate']:.3f}"
        if m["CoreExactRestoreRate"] is not None
        else "n/a"
    )
    failure_types = ", ".join(f"{k}={v}" for k, v in sorted(m["ProtocolFailureTypeCounts"].items()))
    stop_counts = ", ".join(f"{k}={v}" for k, v in sorted(m["StoppingStatusCounts"].items()))
    lines = [
        f"# {summary['method_label']} Pilot Report",
        "",
        "This pilot adapts one external paper method into the local OR hidden-slot clarification interview framework.",
        "",
        "## Configuration",
        "",
        f"- Method: `{summary['method_name']}`",
        f"- TOML dirs: `{', '.join(args.toml_dirs)}`",
        f"- Case ids: `{', '.join(args.case_ids) if args.case_ids else '(all selected by limit)'}`",
        f"- K: `{args.k}`",
        f"- Max turns: `{args.max_turns}`",
        "- Max-turn hit: `attempted_turn_count == max_turns and completed_ready_to_model == false`",
        f"- Agent model/profile: `{args.agent_profile}`",
        f"- User simulator profile: `{args.user_profile}`",
        f"- Judge profile: `{args.judge_profile}`",
        "- Protocol detectors / MC-D / task-list memory / separate readiness gate: disabled",
        "",
        "## Metrics",
        "",
        "| runs | all-slot exact restore | core exact restore | weighted macro | weighted micro | ready rate | max-turn hit | avg turns | avg attempted turns | silent assumptions/run | stopping mismatch | protocol failure | failure types | cost USD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        f"| {summary['runs']} | {m['AllSlotExactRestoreRate']:.3f} | {core_exact_rate} | "
        f"{m['WeightedSlotScore_macro']:.3f} | {m['WeightedSlotScore_micro']:.3f} | "
        f"{m['ReadyToModelRate']:.3f} | {m['MaxTurnHitRate']:.3f} | "
        f"{m['AverageTurns']:.2f} | {m['AverageAttemptedTurns']:.2f} | "
        f"{m['SilentAssumptionsPerRun']:.2f} | {m['StoppingStatusMismatchRate']:.3f} | "
        f"{m['ProtocolFailureRate']:.3f} | {failure_types} | {m['Cost USD']:.4f} |",
        "",
        f"Stopping status counts: {stop_counts}",
        "",
        "Detailed per-run transcripts and judge JSON files are stored under the run directory.",
        "",
    ]
    (output_root / "external_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def normalize_case_token(value: str) -> set[str]:
    raw = value.strip()
    tokens = {raw, raw.lower()}
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        tokens.add(str(int(digits)))
        tokens.add(digits.zfill(3))
        tokens.add(f"orclarify_{digits.zfill(3)}")
    return tokens


def filter_cases_by_ids(cases: list[dict[str, Any]], case_ids: list[str]) -> list[dict[str, Any]]:
    if not case_ids:
        return cases
    wanted: set[str] = set()
    for case_id in case_ids:
        wanted.update(normalize_case_token(case_id))
    selected = []
    for case in cases:
        case_values = normalize_case_token(str(case.get("_case_id", "")))
        if wanted & case_values:
            selected.append(case)
    if not selected:
        raise ValueError(f"No selected case ids found: {case_ids}")
    return selected


def build_argument_parser(method_name: str, default_prompt_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {method_name} external baseline adapter.")
    parser.add_argument("--toml_dirs", nargs="+", default=[str(path) for path in DEFAULT_TOML_DIRS])
    parser.add_argument("--case_ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--agent_prompt", default=str(default_prompt_path))
    parser.add_argument("--user_prompt", default=str(REPO_ROOT / "experiments" / "evaluation_protocol" / "prompts" / "user_simulator_prompt.md"))
    parser.add_argument("--judge_prompt", default=str(REPO_ROOT / "experiments" / "evaluation_protocol" / "prompts" / "judge_prompt.md"))
    parser.add_argument("--agent_profile", default="generic_agent")
    parser.add_argument("--user_profile", default="user_simulator")
    parser.add_argument("--judge_profile", default="judge")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    return parser


def main(method_name: str, method_label: str, default_prompt_path: Path) -> None:
    parser = build_argument_parser(method_name, default_prompt_path)
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else REF.resolve_runs_root() / f"{method_name}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    agent_prompt = Path(args.agent_prompt).read_text(encoding="utf-8").strip()
    user_prompt = Path(args.user_prompt).read_text(encoding="utf-8").strip()
    judge_prompt = Path(args.judge_prompt).read_text(encoding="utf-8").strip()
    cases = REF.load_cases([Path(path) for path in args.toml_dirs], args.limit)
    cases = filter_cases_by_ids(cases, args.case_ids)
    if args.limit is not None and args.case_ids:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("No cases selected for baseline run.")

    print("| milestone | value |", flush=True)
    print("|---|---|", flush=True)
    print(f"| method | {method_name} |", flush=True)
    print(f"| selected cases | {len(cases)} |", flush=True)
    print(f"| k | {args.k} |", flush=True)
    print(f"| output | {output_root} |", flush=True)

    stats: list[dict[str, Any]] = []
    total_tasks = len(cases) * args.k
    done = 0
    for run_index in range(1, args.k + 1):
        for case in cases:
            stat = run_interaction(
                case=case,
                method_name=method_name,
                method_label=method_label,
                agent_prompt=agent_prompt,
                user_prompt=user_prompt,
                judge_prompt=judge_prompt,
                run_index=run_index,
                output_root=output_root,
                agent_profile=args.agent_profile,
                user_profile=args.user_profile,
                judge_profile=args.judge_profile,
                agent_temperature=args.agent_temperature,
                user_temperature=args.user_temperature,
                judge_temperature=args.judge_temperature,
                max_turns=args.max_turns,
            )
            stats.append(stat)
            done += 1
            print(
                f"| progress | {done}/{total_tasks} {case['_case_id']} score={stat.get('weighted_slot_score')} ready={stat.get('completed_ready_to_model')} |",
                flush=True,
            )

    summary = summarize(stats, output_root, method_name, method_label)
    write_markdown_report(summary, output_root, args)
    print(f"| summary | {output_root / 'summary_external_baseline.json'} |", flush=True)
