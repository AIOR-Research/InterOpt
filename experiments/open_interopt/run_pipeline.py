from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


MODEL_PROFILES = {
    "deepseek_v4_pro": {
        "provider": "deepseek",
        "model_version": "deepseek-v4-pro",
    },
    "gemini_3_1_pro": {
        "provider": "openrouter",
        "model_version": "google/gemini-3.1-pro-preview",
    },
    "gpt_5_5": {
        "provider": "openrouter",
        "model_version": "openai/gpt-5.5",
    },
    "opus_4_6": {
        "provider": "openrouter",
        "model_version": "anthropic/claude-opus-4.6",
    },
    "glm_5_1": {
        "provider": "openrouter",
        "model_version": "glm-5.1",
    },
}


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
OR_DIMENSION_IDS = {"T1", "T2", "T3", "T4", "T5"}
OR_DIMENSION_NAMES = {
    "T1": "objective_and_tradeoff",
    "T2": "decision_scope_and_granularity",
    "T3": "constraints_and_business_rules",
    "T4": "fulfillment_and_coverage_policy",
    "T5": "operational_logic_and_variable_domain",
}

# 重试配置（可改为从 argparse 传入）
MAX_AGENT_RETRIES_PER_TURN = 3
MAX_USER_RETRIES_PER_TURN = 3
PIPELINE_VERSION = "open_interopt"
GAP_SEARCH_EVENTS_FILENAME = "gap_search_events.json"
GAP_SEARCH_MAX_TOKENS = 2000


@dataclass
class ChatResult:
    content: str
    usage: dict[str, int]
    estimated_cost_usd: float


class ChatClient:
    def __init__(self, profile_name: str, temperature: float):
        if profile_name not in MODEL_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        self.profile_name = profile_name
        self.provider = MODEL_PROFILES[profile_name]["provider"]
        self.model = MODEL_PROFILES[profile_name]["model_version"]
        self.temperature = temperature
        self.total_usage: dict[str, int] = {}
        self.total_estimated_cost_usd = 0.0

    def _api_settings(self) -> tuple[str, str]:
        if self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if not api_key:
            raise RuntimeError(f"Missing API key for provider {self.provider}")
        return base_url, api_key

    def complete(
        self,
        messages: list[dict[str, str]],
        timeout: int = 180,
        max_retries: int = 5,
        max_tokens: int | None = None,
        thinking_type: str | None = None,
        response_format_json: bool = False,
    ) -> ChatResult:
        retry = 0
        while True:
            try:
                return self._complete_once(messages, timeout, max_tokens, thinking_type, response_format_json)
            except Exception as exc:
                retry += 1
                if retry > max_retries:
                    raise RuntimeError(f"Maximum retries exceeded for {self.model}: {exc}") from exc
                delay = min(60.0, 1.0 * (2 ** (retry - 1)))
                print(f"Error: {exc}. Retrying in {delay:.2f}s (attempt {retry}/{max_retries})", flush=True)
                time.sleep(delay)

    def _complete_once(
        self,
        messages: list[dict[str, str]],
        timeout: int,
        max_tokens: int | None,
        thinking_type: str | None,
        response_format_json: bool,
    ) -> ChatResult:
        base_url, api_key = self._api_settings()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking_type in {"enabled", "disabled"}:
            payload["thinking"] = {"type": thinking_type}
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "Anonymous-Clarification-Eval")
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
            reasoning_content = message.get("reasoning_content", "")
            if isinstance(reasoning_content, list):
                reasoning_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in reasoning_content
                )
            content = str(reasoning_content).strip()
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


def normalize_or_dimension_audit(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        dim_id = str(item.get("id", "")).strip().upper()
        status = str(item.get("status", "")).strip().lower()
        if status not in {"done", "open", "not_applicable"}:
            status = "unknown"
        normalized.append(
            {
                "id": dim_id,
                "dimension": str(item.get("dimension", "")).strip(),
                "status": status,
                "evidence": str(item.get("evidence", "")).strip(),
                "remaining_gap": str(item.get("remaining_gap", "")).strip(),
            }
        )
    return normalized


def normalize_candidate_questions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        score: float | None
        try:
            score = float(item.get("selection_score"))
        except (TypeError, ValueError):
            score = None
        normalized.append(
            {
                "id": str(item.get("id", "")).strip().upper(),
                "question": str(item.get("question", "")).strip(),
                "why_it_matters": str(item.get("why_it_matters", "")).strip(),
                "answerability": str(item.get("answerability", "")).strip(),
                "overask_risk": str(item.get("overask_risk", "")).strip(),
                "selection_score": score,
            }
        )
    return normalized


def candidate_questions_complete(candidates: list[dict[str, Any]], selected_question_id: str) -> bool:
    if len(candidates) != 3:
        return False
    ids = {str(item.get("id", "")).upper() for item in candidates}
    if selected_question_id and selected_question_id not in ids:
        return False
    return all(str(item.get("question", "")).strip() for item in candidates)


def audit_has_all_or_dimensions(audit: list[dict[str, Any]]) -> bool:
    return OR_DIMENSION_IDS.issubset({str(item.get("id", "")).upper() for item in audit})


def format_ready_public_text(
    *,
    confidence: float | None,
    confidence_rationale: str,
    summary: str,
) -> str:
    lines = ["READY_TO_MODEL"]
    if confidence is not None:
        lines.append(f"Formulatable confidence: {confidence:.3f}")
    if confidence_rationale:
        lines.append(f"Confidence rationale: {confidence_rationale}")
    if summary:
        lines.append(f"Summary: {summary}")
    return "\n\n".join(lines)


def parse_agent_output(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    parsed: dict[str, Any] = {
        "parse_ok": False,
        "parse_error": "",
        "action": "UNKNOWN",
        "public_text": raw_text,
        "agent_context_text": raw_text,
        "or_dimension_audit": [],
        "audit_complete": False,
        "selected_dimension_id": "",
        "candidate_questions": [],
        "candidate_questions_complete": False,
        "selected_question_id": "",
        "selection_rationale": "",
        "public_question": "",
        "formulatable_confidence": None,
        "confidence_parse_ok": False,
        "confidence_rationale": "",
        "summary": "",
    }
    try:
        obj = extract_json_object(raw_text)
    except Exception as exc:
        upper = raw_text.upper()
        if upper.startswith("READY_TO_MODEL"):
            parsed["action"] = "READY_TO_MODEL"
        elif upper.startswith("QUESTION:"):
            parsed["action"] = "ASK"
            parsed["public_question"] = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
        else:
            parsed["action"] = "UNKNOWN"
        parsed["parse_error"] = str(exc)
        return parsed

    action = str(obj.get("action", "")).strip().upper()
    audit = normalize_or_dimension_audit(obj.get("or_dimension_audit", []))
    candidates = normalize_candidate_questions(obj.get("candidate_questions", []))
    selected_question_id = str(obj.get("selected_question_id", "")).strip().upper()
    parsed.update(
        {
            "parse_ok": action in {"ASK", "READY_TO_MODEL"},
            "action": action if action else "UNKNOWN",
            "or_dimension_audit": audit,
            "audit_complete": audit_has_all_or_dimensions(audit),
            "selected_dimension_id": str(obj.get("selected_dimension_id", "")).strip().upper(),
            "candidate_questions": candidates,
            "candidate_questions_complete": candidate_questions_complete(candidates, selected_question_id),
            "selected_question_id": selected_question_id,
            "selection_rationale": str(obj.get("selection_rationale", "")).strip(),
        }
    )
    if action == "ASK":
        question = str(obj.get("public_question", "")).strip()
        parsed["public_question"] = question
        parsed["public_text"] = "QUESTION: " + question if question and not question.upper().startswith("QUESTION:") else question
        if not question:
            parsed["parse_ok"] = False
            parsed["parse_error"] = "ASK action missing public_question."
    elif action == "READY_TO_MODEL":
        summary = str(obj.get("summary", "")).strip()
        rationale = str(obj.get("confidence_rationale", "")).strip()
        confidence: float | None = None
        if "formulatable_confidence" in obj:
            try:
                confidence = float(obj.get("formulatable_confidence"))
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("confidence out of range")
                parsed["confidence_parse_ok"] = True
            except Exception as exc:
                parsed["confidence_parse_ok"] = False
                parsed["parse_error"] = str(exc)
                confidence = None
        parsed["formulatable_confidence"] = confidence
        parsed["confidence_rationale"] = rationale
        parsed["summary"] = summary
        parsed["public_text"] = format_ready_public_text(
            confidence=confidence,
            confidence_rationale=rationale,
            summary=summary,
        )
    else:
        parsed["parse_ok"] = False
        parsed["parse_error"] = f"Unsupported agent action: {action!r}"
    return parsed


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
        "Your previous response did not pass the question interaction protocol.",
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
            "Rewrite the response as a valid public clarification question, or output READY_TO_MODEL.",
            "Do not bundle unrelated business gaps into one turn.",
            "Do not ask for implementation details, solver choices, or output formatting.",
            "Do not infer or mention any hidden benchmark labels.",
            "",
            "Allowed output formats:",
            "QUESTION: <public clarification question>",
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


def build_user_scope_feedback(
    answer_result: dict[str, Any] | None = None,
    detector_error: str | None = None,
) -> str:
    lines = [
        "Protocol feedback from the interaction supervisor:",
        "",
        "Your previous user-simulator response did not pass the answer-scope protocol.",
    ]
    if detector_error:
        lines.extend(
            [
                f"Detector issue: {detector_error}",
                "Please answer again using only information that directly responds to the current question.",
            ]
        )
    elif answer_result:
        violations = answer_result.get("scope_violations", []) or []
        rationale = answer_result.get("rationale", "")
        if violations:
            lines.append("Out-of-scope content detected:")
            for idx, violation in enumerate(violations, start=1):
                lines.append(f"{idx}. {violation}")
        if rationale:
            lines.append(f"Detector rationale: {rationale}")
    lines.extend(
        [
            "",
            "Rewrite the response as a business user.",
            "Answer only the current question.",
            "Do not volunteer answers to other hidden slots.",
            "Do not mention the benchmark, detector, judge, slots, or protocol.",
        ]
    )
    return "\n".join(lines)


def build_minimal_agent_retry_feedback(*, detector_error: bool = False) -> str:
    if detector_error:
        return "Your previous output did not follow the required format. Please output QUESTION: <public clarification question>, or output READY_TO_MODEL."
    return "Your previous output was not a valid QUESTION or READY_TO_MODEL action. Please output a public clarification question about the next unresolved gap, or output READY_TO_MODEL."


def build_minimal_user_retry_feedback(*, detector_error: bool = False) -> str:
    if detector_error:
        return "Your previous answer did not follow the required format. Please answer only the current question."
    return "Your previous answer included information beyond the current question. Please answer only the current question."



def load_case(path: Path) -> dict[str, Any]:
    case = tomllib.loads(path.read_text(encoding="utf-8"))
    case["_path"] = str(path)
    case["_case_id"] = case.get("metadata", {}).get("case_id", path.stem)
    return case


def load_cases(toml_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    paths = sorted(toml_dir.glob("*.toml"))
    if limit is not None:
        paths = paths[:limit]
    return [load_case(path) for path in paths]


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


def render_atomic_question_map(rows: list[dict[str, Any]]) -> str:
    """Expose the detector-counted question indices to the final judge."""
    lines: list[str] = []
    atomic_index = 0
    for row in rows:
        if str(row.get("speaker")) != "generic_agent":
            continue
        for question in row.get("detected_atomic_questions") or []:
            text = str(question).strip()
            if not text:
                continue
            atomic_index += 1
            lines.append(f"Q{atomic_index} (turn {row.get('turn')}): {text}")
    return "\n".join(lines) or "(No detector-counted atomic questions.)"


def build_question_slot_hit_events(
    transcript: list[dict[str, Any]], slot_scores: list[dict[str, Any]]
) -> dict[str, Any]:
    """Persist the final judge's first full question hit for every hidden slot."""
    atomic_questions = []
    for row in transcript:
        if str(row.get("speaker")) != "generic_agent":
            continue
        for question in row.get("detected_atomic_questions") or []:
            text = str(question).strip()
            if text:
                atomic_questions.append(
                    {
                        "atomic_question": len(atomic_questions) + 1,
                        "turn": row.get("turn"),
                        "question": text,
                    }
                )
    by_index = {item["atomic_question"]: item for item in atomic_questions}
    events = []
    for score in slot_scores:
        hit = str(score.get("hit", "")).lower()
        first = score.get("first_yes_atomic_question")
        if hit != "yes" or not isinstance(first, int) or first not in by_index:
            first = None
        events.append(
            {
                "slot_id": str(score.get("slot_id", "")),
                "name": str(score.get("name", "")),
                "severity": str(score.get("severity", "")),
                "question_hit": hit if hit in HIT_VALUES else "no",
                "first_yes_atomic_question": first,
                "evidence_question": by_index[first]["question"] if first is not None else "",
                "final_judge_evidence_location": str(score.get("evidence_location", "")),
                "final_judge_evidence_quote": str(score.get("evidence_quote", "")),
            }
        )
    return {
        "schema_version": "question_slot_hit_events_v1",
        "metric": "first full semantic hit by detector-counted atomic question",
        "atomic_questions": atomic_questions,
        "slot_scores": events,
    }


def normalize_gap_search_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = str(result.get("search_summary", "")).strip()
    if "ready_to_model" not in result:
        raise ValueError("gap search missing ready_to_model")
    ready_to_model_raw = result["ready_to_model"]
    if not isinstance(ready_to_model_raw, bool):
        raise ValueError("gap search ready_to_model must be a boolean")
    raw_gaps = result.get("gaps", [])
    if not summary:
        raise ValueError("gap search missing search_summary")
    if not isinstance(raw_gaps, list) or len(raw_gaps) > 5:
        raise ValueError("gap search gaps must contain between 0 and 5 items")
    if ready_to_model_raw and raw_gaps:
        raise ValueError("gap search gaps must be empty when ready_to_model is true")
    if not ready_to_model_raw and not raw_gaps:
        raise ValueError("gap search must return gaps unless ready_to_model is true")
    gaps: list[dict[str, str]] = []
    for index, raw_gap in enumerate(raw_gaps, start=1):
        if not isinstance(raw_gap, dict):
            raise ValueError(f"gap {index} must be an object")
        gap = {
            "gap_id": str(raw_gap.get("gap_id", "")).strip(),
            "category": str(raw_gap.get("category", "")).strip(),
            "description": str(raw_gap.get("description", "")).strip(),
            "direct_question": str(raw_gap.get("direct_question", "")).strip(),
            "source_status": str(raw_gap.get("source_status", "")).strip(),
            "why_material": str(raw_gap.get("why_material", "")).strip(),
        }
        if not all(gap.values()):
            raise ValueError(f"gap {index} fields must all be non-empty")
        gaps.append(gap)
    return {
        "search_summary": summary,
        "ready_to_model": bool(ready_to_model_raw),
        "ready_to_model_raw": ready_to_model_raw,
        "gaps": gaps,
    }


def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def append_live_event(run_dir: Path, case_start_time: float, event: dict[str, Any]) -> None:
    payload = {"elapsed_sec": round(time.time() - case_start_time, 3)}
    payload.update(event)
    with (run_dir / "live_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_direct_gap_question(gap: dict[str, str]) -> str:
    question = collapse_whitespace(gap.get("direct_question") or gap.get("description", ""))
    if question.upper().startswith("QUESTION:"):
        question = question.split(":", 1)[1].strip()
    question = question.rstrip(". ")
    if question and not question.endswith("?"):
        question += "?"
    return question


def choose_first_gap_question(gaps: list[dict[str, str]]) -> tuple[dict[str, str], str] | None:
    if not gaps:
        return None
    top_gap = gaps[0]
    question = format_direct_gap_question(top_gap)
    if not question:
        return None
    return top_gap, question


def build_gap_direct_agent_output(gap_search_result: dict[str, Any]) -> str:
    gaps = gap_search_result.get("gaps", []) or []
    if gap_search_result.get("ready_to_model"):
        return json.dumps(
            {
                "action": "READY_TO_MODEL",
                "summary": "Stage1 declared the public request and dialogue ready for modeling.",
            },
            ensure_ascii=False,
        )

    chosen = choose_first_gap_question(gaps)
    if chosen is None:
        raise ValueError("Stage1 returned unresolved gaps, but the first gap has no usable direct_question.")

    top_gap, question = chosen
    return json.dumps(
        {
            "action": "ASK",
            "public_question": question,
            "gap_direct_policy": {
                "selected_gap_id": top_gap.get("gap_id", ""),
                "selected_gap_category": top_gap.get("category", ""),
                "selected_gap_description": top_gap.get("description", ""),
                "selected_direct_question": question,
                "source_status": top_gap.get("source_status", ""),
                "why_material": top_gap.get("why_material", ""),
            },
        },
        ensure_ascii=False,
    )


def detector_accepts_agent_action(detector_result: dict[str, Any]) -> bool:
    action = detector_result.get("action")
    if action == "ready_to_model":
        return True
    return action == "question"


def render_gap_search_user_message(case: dict[str, Any], transcript: list[dict[str, Any]]) -> str:
    public_transcript = render_transcript(transcript) or "(No public dialogue yet.)"
    return "\n".join(
        [
            "# Initial public request",
            "",
            str(case["initial_brief"]["content"]),
            "",
            "# Public conversation so far",
            "",
            public_transcript,
        ]
    )


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
        "core_exact_applicable": bool(core_slots),
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
        "rule": "Core exact uses P0/P1 only; all-slot exact uses P0/P1/P2. P2-only cases are not applicable to core exact.",
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


def load_prompt_files(prompts_dir: Path) -> tuple[str, str, str, str, str, str]:
    return (
        (prompts_dir / "generic_agent_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "question_detector_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "answer_scope_detector_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "user_simulator_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "judge_prompt.md").read_text(encoding="utf-8").strip(),
        (prompts_dir / "gap_search_prompt.md").read_text(encoding="utf-8").strip(),
    )


def run_interaction(
    case: dict[str, Any],
    agent_profile: str,
    detector_profile: str,
    user_profile: str,
    judge_profile: str,
    run_index: int,
    output_root: Path,
    prompts: tuple[str, str, str, str, str, str],
    agent_temperature: float,
    detector_temperature: float,
    user_temperature: float,
    judge_temperature: float,
    max_turns: int,
    monitor_user_answers: bool = False,
    detector_feedback_mode: str = "visible",
    retry_limit_behavior: str = "pass_through",
    pipeline_mode: str | None = None,
    case_timeout_sec: float | None = None,
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

    (
        generic_prompt,
        question_detector_prompt,
        answer_detector_prompt,
        simulator_prompt_base,
        judge_prompt,
        gap_search_prompt,
    ) = prompts
    agent_client = ChatClient(agent_profile, agent_temperature)
    gap_search_client = ChatClient(agent_profile, 0.0)
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
        if monitor_user_answers and detector_feedback_mode in {"minimal", "none"} and retry_limit_behavior == "protocol_failed":
            pipeline_mode = "fixed_both_sides_silent"
        elif monitor_user_answers:
            pipeline_mode = "fixed_both_sides"
        else:
            pipeline_mode = "fixed_question_only"

    transcript: list[dict[str, Any]] = []
    detector_events: list[dict[str, Any]] = []
    answer_scope_events: list[dict[str, Any]] = []
    or_dimension_audit_events: list[dict[str, Any]] = []
    gap_search_events: list[dict[str, Any]] = []
    completed = False
    protocol_failed = False
    protocol_failure_type = ""
    protocol_failure_reason = ""
    final_agent_text = ""
    final_agent_raw_text = ""
    confidence_parse_ok = False
    formulatable_confidence: float | None = None
    confidence_rationale = ""
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
    user_retry_total = 0
    agent_max_retries_exceeded = False
    user_max_retries_exceeded = False
    agent_retry_limit_pass_through_count = 0
    user_retry_limit_pass_through_count = 0
    case_start_time = time.time()
    (run_dir / "live_events.jsonl").write_text("", encoding="utf-8")

    def case_timed_out() -> bool:
        return case_timeout_sec is not None and (time.time() - case_start_time) >= case_timeout_sec

    for turn in range(1, max_turns + 1):
        if case_timed_out():
            protocol_failed = True
            protocol_failure_type = "case_timeout"
            protocol_failure_reason = f"Exceeded case timeout of {case_timeout_sec:.0f} seconds before turn {turn}"
            break
        # ----- Agent 重试循环 -----
        agent_retry = 0
        agent_retry_feedback: str | None = None
        gap_search_result: dict[str, Any] | None = None
        gap_search_guidance = ""
        gap_search_event: dict[str, Any] = {
            "turn": turn,
            "entered_agent_context": False,
        }
        try:
            gap_search_raw = gap_search_client.complete(
                [
                    {"role": "system", "content": gap_search_prompt},
                    {"role": "user", "content": render_gap_search_user_message(case, transcript)},
                ],
                max_tokens=GAP_SEARCH_MAX_TOKENS,
                thinking_type="disabled",
                response_format_json=True,
            ).content
            gap_search_result = normalize_gap_search_result(extract_json_object(gap_search_raw))
            gap_search_event.update(
                {
                    "status": "valid",
                    "raw_response": gap_search_raw,
                    "result": gap_search_result,
                    "entered_agent_context": True,
                    "visibility": {
                        "uses_only_public_dialogue": True,
                        "entered_user_context": False,
                        "entered_transcript": False,
                        "entered_judge_context": False,
                    },
                }
            )
            gap_search_guidance = (
                "Internal gap-search guidance. Use it only to prioritize your private candidate questions. "
                "Do not mention this stage, its JSON, or hidden evaluation information to the user.\n\n"
                + json.dumps(gap_search_result, ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            gap_search_event.update(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            )
        gap_search_events.append(gap_search_event)
        append_live_event(
            run_dir,
            case_start_time,
            {
                "event": "gap_search",
                "turn": turn,
                "status": gap_search_event.get("status"),
                "ready_to_model": gap_search_result.get("ready_to_model") if gap_search_result else None,
                "gap_count": len(gap_search_result.get("gaps", [])) if gap_search_result else None,
                "first_gap_id": (gap_search_result.get("gaps", [{}])[0].get("gap_id") if gap_search_result and gap_search_result.get("gaps") else None),
                "first_direct_question": (format_direct_gap_question(gap_search_result.get("gaps", [{}])[0]) if gap_search_result and gap_search_result.get("gaps") else None),
                "search_summary": gap_search_result.get("search_summary") if gap_search_result else None,
                "error": gap_search_event.get("error"),
            },
        )
        if gap_search_result is None:
            protocol_failed = True
            protocol_failure_type = "gap_search_error"
            protocol_failure_reason = str(gap_search_event.get("error", "gap search failed"))
            break
        while True:
            if case_timed_out():
                protocol_failed = True
                protocol_failure_type = "case_timeout"
                protocol_failure_reason = f"Exceeded case timeout of {case_timeout_sec:.0f} seconds during turn {turn}"
                break
            agent_call_messages = list(agent_messages)
            if gap_search_guidance:
                agent_call_messages.append({"role": "user", "content": gap_search_guidance})
            if detector_feedback_mode == "minimal" and agent_retry_feedback:
                agent_call_messages.append({"role": "user", "content": agent_retry_feedback})
            raw_agent_reply = agent_client.complete(agent_call_messages).content
            parsed_agent = parse_agent_output(raw_agent_reply)
            agent_reply = parsed_agent["public_text"]
            agent_context_reply = parsed_agent["agent_context_text"]
            final_agent_text = agent_reply
            final_agent_raw_text = raw_agent_reply
            append_live_event(
                run_dir,
                case_start_time,
                {
                    "event": "agent_output",
                    "turn": turn,
                    "retry_count": agent_retry,
                    "action": parsed_agent["action"],
                    "public_question": parsed_agent["public_question"],
                    "summary": parsed_agent["summary"],
                    "public_text": agent_reply,
                },
            )

            audit_event: dict[str, Any] = {
                "turn": turn,
                "retry_count": agent_retry,
                "raw_agent_response": raw_agent_reply,
                "parse_ok": parsed_agent["parse_ok"],
                "parse_error": parsed_agent["parse_error"],
                "action": parsed_agent["action"],
                "or_dimension_audit": parsed_agent["or_dimension_audit"],
                "audit_complete": parsed_agent["audit_complete"],
                "selected_dimension_id": parsed_agent["selected_dimension_id"],
                "candidate_questions": parsed_agent["candidate_questions"],
                "candidate_questions_complete": parsed_agent["candidate_questions_complete"],
                "selected_question_id": parsed_agent["selected_question_id"],
                "selection_rationale": parsed_agent["selection_rationale"],
                "public_question": parsed_agent["public_question"],
                "formulatable_confidence": parsed_agent["formulatable_confidence"],
                "confidence_parse_ok": parsed_agent["confidence_parse_ok"],
                "confidence_rationale": parsed_agent["confidence_rationale"],
                "summary": parsed_agent["summary"],
                "public_text": agent_reply,
                "entered_user_context": False,
                "entered_transcript": False,
                "audit_entered_user_context": False,
                "audit_entered_judge_context": False,
            }

            detector_user_message = "Assistant latest response:\n\n" + agent_reply
            detector_messages = [
                {"role": "system", "content": question_detector_prompt},
                {"role": "user", "content": detector_user_message},
            ]
            detector_raw = detector_client.complete(detector_messages).content
            event: dict[str, Any] = {
                "turn": turn,
                "agent_response": agent_reply,
                "raw_agent_response": raw_agent_reply,
                "agent_parse_ok": parsed_agent["parse_ok"],
                "agent_parse_error": parsed_agent["parse_error"],
                "detector_raw": detector_raw,
                "retry_count": agent_retry,
            }
            try:
                detector_result = normalize_detector_result(extract_json_object(detector_raw))
                event["detector_result"] = detector_result
            except Exception as exc:
                detector_error_count += 1
                agent_retry_total += 1
                event["detector_error"] = str(exc)
                event["accepted"] = False
                event["entered_transcript"] = False
                event["feedback_mode"] = detector_feedback_mode
                event["protocol_violation_type"] = "agent_detector_parse_error"
                event["detected_action"] = "parse_error"
                event["detected_question_count"] = None
                event["detected_atomic_questions"] = []
                audit_event["accepted"] = False
                audit_event["detector_error"] = str(exc)
                audit_event["detector_action"] = "parse_error"
                audit_event["detector_question_count"] = None
                audit_event["detector_atomic_questions"] = []
                or_dimension_audit_events.append(audit_event)
                detector_events.append(event)
                append_live_event(
                    run_dir,
                    case_start_time,
                    {
                        "event": "detector_error",
                        "turn": turn,
                        "retry_count": agent_retry,
                        "error": str(exc),
                        "agent_response": agent_reply,
                    },
                )
                agent_retry += 1
                if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                    event["retry_limit_exceeded"] = True
                    audit_event["retry_limit_exceeded"] = True
                    if retry_limit_behavior == "pass_through":
                        agent_retry_limit_pass_through_count += 1
                        event["pass_through_after_retry_limit"] = True
                        event["entered_transcript"] = True
                        audit_event["accepted"] = True
                        audit_event["entered_transcript"] = True
                        audit_event["entered_user_context"] = True
                        transcript.append(
                            {
                                "turn": turn,
                                "speaker": "generic_agent",
                                "content": agent_reply,
                                "detector_pass_through_after_retry_limit": True,
                                "detector_pass_through_reason": "detector_parse_error",
                            }
                        )
                        agent_messages.append({"role": "assistant", "content": agent_context_reply})
                    else:
                        agent_max_retries_exceeded = True
                    break
                if detector_feedback_mode == "visible":
                    agent_messages.append({"role": "assistant", "content": agent_context_reply})
                    agent_messages.append(
                        {
                            "role": "user",
                            "content": build_agent_protocol_feedback(detector_error=str(exc)),
                        }
                    )
                elif detector_feedback_mode == "minimal":
                    agent_retry_feedback = build_minimal_agent_retry_feedback(detector_error=True)
                continue

            detector_events.append(event)
            action = detector_result["action"]
            append_live_event(
                run_dir,
                case_start_time,
                {
                    "event": "detector_result",
                    "turn": turn,
                    "retry_count": agent_retry,
                    "accepted": detector_accepts_agent_action(detector_result),
                    "detector_action": action,
                    "question_count": detector_result.get("question_count"),
                    "atomic_questions": detector_result.get("atomic_questions", []),
                    "rationale": detector_result.get("rationale", ""),
                },
            )

            if detector_accepts_agent_action(detector_result):
                if action == "ready_to_model":
                    event["accepted"] = True
                    event["entered_transcript"] = True
                    audit_event["accepted"] = True
                    audit_event["entered_transcript"] = True
                    audit_event["detector_action"] = action
                    audit_event["detector_question_count"] = detector_result.get("question_count")
                    audit_event["detector_atomic_questions"] = detector_result.get("atomic_questions", [])
                    confidence_parse_ok = bool(parsed_agent["confidence_parse_ok"])
                    formulatable_confidence = parsed_agent["formulatable_confidence"]
                    confidence_rationale = parsed_agent["confidence_rationale"]
                    or_dimension_audit_events.append(audit_event)
                    transcript.append({"turn": turn, "speaker": "generic_agent", "content": agent_reply})
                    completed = True
                    break

                # action == "question"; one or more detector-counted atomic questions are allowed.
                event["accepted"] = True
                event["entered_transcript"] = True
                audit_event["accepted"] = True
                audit_event["entered_transcript"] = True
                audit_event["entered_user_context"] = True
                audit_event["detector_action"] = action
                audit_event["detector_question_count"] = detector_result.get("question_count")
                audit_event["detector_atomic_questions"] = detector_result.get("atomic_questions", [])
                or_dimension_audit_events.append(audit_event)
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
                agent_messages.append({"role": "assistant", "content": agent_context_reply})
                break

            detector_rejection_count += 1
            agent_retry_total += 1
            agent_retry += 1
            event["accepted"] = False
            event["entered_transcript"] = False
            event["feedback_mode"] = detector_feedback_mode
            event["protocol_violation_type"] = "agent_question_protocol"
            event["detected_action"] = action
            event["detected_question_count"] = detector_result.get("question_count")
            event["detected_atomic_questions"] = detector_result.get("atomic_questions", [])
            event["detector_rationale"] = detector_result.get("rationale", "")
            audit_event["accepted"] = False
            audit_event["detector_action"] = action
            audit_event["detector_question_count"] = detector_result.get("question_count")
            audit_event["detector_atomic_questions"] = detector_result.get("atomic_questions", [])
            audit_event["detector_rationale"] = detector_result.get("rationale", "")
            or_dimension_audit_events.append(audit_event)
            if agent_retry > MAX_AGENT_RETRIES_PER_TURN:
                event["retry_limit_exceeded"] = True
                audit_event["retry_limit_exceeded"] = True
                if retry_limit_behavior == "pass_through":
                    agent_retry_limit_pass_through_count += 1
                    event["pass_through_after_retry_limit"] = True
                    event["entered_transcript"] = True
                    audit_event["accepted"] = True
                    audit_event["entered_transcript"] = True
                    audit_event["entered_user_context"] = True
                    transcript.append(
                        {
                            "turn": turn,
                            "speaker": "generic_agent",
                            "content": agent_reply,
                            "detector_pass_through_after_retry_limit": True,
                            "detector_pass_through_reason": action,
                        }
                    )
                    agent_messages.append({"role": "assistant", "content": agent_context_reply})
                else:
                    agent_max_retries_exceeded = True
                break
            if detector_feedback_mode == "visible":
                agent_messages.append({"role": "assistant", "content": agent_context_reply})
                agent_messages.append(
                    {
                        "role": "user",
                        "content": build_agent_protocol_feedback(detector_result=detector_result),
                    }
                )
            elif detector_feedback_mode == "minimal":
                agent_retry_feedback = build_minimal_agent_retry_feedback()
            continue
        if protocol_failed:
            break

        if agent_max_retries_exceeded:
            protocol_failed = True
            protocol_failure_type = "agent_max_retries_exceeded"
            protocol_failure_reason = f"Exceeded {MAX_AGENT_RETRIES_PER_TURN} retries in turn {turn}"
            break

        if completed:
            break  # READY_TO_MODEL 结束

        # ----- User response loop -----
        if case_timed_out():
            protocol_failed = True
            protocol_failure_type = "case_timeout"
            protocol_failure_reason = f"Exceeded case timeout of {case_timeout_sec:.0f} seconds before user response in turn {turn}"
            break
        simulator_messages.append({"role": "user", "content": agent_reply})
        if not monitor_user_answers:
            simulator_reply = user_client.complete(simulator_messages).content
            simulator_messages.append({"role": "assistant", "content": simulator_reply})
            transcript.append({"turn": turn, "speaker": "user_simulator", "content": simulator_reply})
            agent_messages.append({"role": "user", "content": "Business user response:\n\n" + simulator_reply})
            append_live_event(
                run_dir,
                case_start_time,
                {
                    "event": "user_response",
                    "turn": turn,
                    "question": agent_reply,
                    "response": simulator_reply,
                    "answer_scope_monitored": False,
                },
            )
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
        append_live_event(
            run_dir,
            case_start_time,
            {
                "event": "user_response",
                "turn": turn,
                "question": agent_reply,
                "response": simulator_reply,
                "answer_scope_monitored": True,
                "scope_violation_count": answer_event.get("scope_violation_count"),
                "disclosed_hidden_slot_count": answer_event.get("disclosed_hidden_slot_count"),
                "disclosed_hidden_slot_ids": answer_event.get("disclosed_hidden_slot_ids", []),
                "detector_error": answer_event.get("detector_error"),
            },
        )

        time.sleep(0.2)

    # ----- Judge 与统计（同原逻辑，增加重试字段）-----
    append_live_event(
        run_dir,
        case_start_time,
        {
            "event": "judge_start",
            "completed_ready_to_model": completed,
            "protocol_failed": protocol_failed,
            "protocol_failure_type": protocol_failure_type,
            "turns_in_transcript": sum(1 for item in transcript if item.get("speaker") == "generic_agent"),
        },
    )
    judge_user_message = (
        render_judge_case(case)
        + "\n\n# Atomic Question Map\n\n"
        + render_atomic_question_map(transcript)
        + "\n\n# Full Transcript\n\n"
        + render_transcript(transcript)
    )
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
    question_slot_hit_events = build_question_slot_hit_events(
        transcript, judge_result.get("slot_scores", [])
    )
    todo_event_count = len(or_dimension_audit_events)
    todo_parse_ok_count = sum(1 for event in or_dimension_audit_events if event.get("parse_ok"))
    todo_audit_complete_count = sum(1 for event in or_dimension_audit_events if event.get("audit_complete"))
    ask_candidate_events = [
        event for event in or_dimension_audit_events if str(event.get("action", "")).upper() == "ASK"
    ]
    candidate_event_count = len(ask_candidate_events)
    candidate_parse_ok_count = sum(1 for event in ask_candidate_events if event.get("parse_ok"))
    candidate_complete_count = sum(1 for event in ask_candidate_events if event.get("candidate_questions_complete"))
    candidate_question_count_sum = sum(len(event.get("candidate_questions", []) or []) for event in ask_candidate_events)
    selected_candidate_scores: list[float] = []
    for event in ask_candidate_events:
        selected_id = str(event.get("selected_question_id", "")).upper()
        for candidate in event.get("candidate_questions", []) or []:
            if str(candidate.get("id", "")).upper() == selected_id and candidate.get("selection_score") is not None:
                selected_candidate_scores.append(float(candidate["selection_score"]))
                break
    gap_search_valid_count = sum(1 for event in gap_search_events if event.get("status") == "valid")
    gap_search_error_count = sum(1 for event in gap_search_events if event.get("status") == "error")
    gap_search_gap_count = sum(
        len((event.get("result") or {}).get("gaps", []))
        for event in gap_search_events
        if event.get("status") == "valid"
    )
    todo_selected_counts = {
        dim_id: sum(
            1
            for event in or_dimension_audit_events
            if str(event.get("selected_dimension_id", "")).upper() == dim_id
        )
        for dim_id in sorted(OR_DIMENSION_IDS)
    }
    true_formulatable = bool(completed and restoration_summary["core_exact_restore"])
    high_confidence_error = (
        bool(completed)
        and confidence_parse_ok
        and formulatable_confidence is not None
        and formulatable_confidence >= 0.8
        and not true_formulatable
    )

    (run_dir / "initial_brief.txt").write_text(case["initial_brief"]["content"], encoding="utf-8")
    (run_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (run_dir / "detector_events.json").write_text(
        json.dumps(detector_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "answer_scope_events.json").write_text(
        json.dumps(answer_scope_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "or_dimension_audit_events.json").write_text(
        json.dumps(or_dimension_audit_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "candidate_question_events.json").write_text(
        json.dumps(or_dimension_audit_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / GAP_SEARCH_EVENTS_FILENAME).write_text(
        json.dumps(gap_search_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "judge_prompt_user_message.md").write_text(judge_user_message, encoding="utf-8")
    (run_dir / "judge_raw.txt").write_text(judge_raw, encoding="utf-8")
    judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "question_slot_hit_events.json").write_text(
        json.dumps(question_slot_hit_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stat = {
        "pipeline_version": PIPELINE_VERSION,
        "mode": pipeline_mode,
        "pipeline_mode": pipeline_mode,
        "monitor_user_answers": monitor_user_answers,
        "detector_feedback_mode": detector_feedback_mode,
        "retry_limit_behavior": retry_limit_behavior,
        "case_timeout_sec": case_timeout_sec,
        "case_elapsed_sec": time.time() - case_start_time,
        "detector_rationale_visible": detector_feedback_mode == "visible",
        "detector_feedback_in_context": detector_feedback_mode == "visible",
        "rejected_attempt_in_transcript": False,
        "case_id": case_id,
        "agent_profile": agent_profile,
        "agent_model": MODEL_PROFILES[agent_profile]["model_version"],
        "gap_search_model": gap_search_client.model,
        "detector_profile": detector_profile,
        "detector_model": MODEL_PROFILES[detector_profile]["model_version"],
        "user_profile": user_profile,
        "user_model": MODEL_PROFILES[user_profile]["model_version"],
        "judge_profile": judge_profile,
        "judge_model": MODEL_PROFILES[judge_profile]["model_version"],
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
        "gap_search_call_count": len(gap_search_events),
        "gap_search_valid_count": gap_search_valid_count,
        "gap_search_error_count": gap_search_error_count,
        "gap_search_gap_count": gap_search_gap_count,
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
        "core_exact_applicable": restoration_summary["core_exact_applicable"],
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
        "or_dimension_audit_event_count": todo_event_count,
        "todo_audit_parse_ok_count": todo_parse_ok_count,
        "todo_audit_parse_ok_rate": todo_parse_ok_count / todo_event_count if todo_event_count else None,
        "todo_audit_complete_count": todo_audit_complete_count,
        "todo_audit_complete_rate": todo_audit_complete_count / todo_event_count if todo_event_count else None,
        "todo_selected_dimension_counts": todo_selected_counts,
        "candidate_question_event_count": candidate_event_count,
        "candidate_question_parse_ok_count": candidate_parse_ok_count,
        "candidate_question_parse_ok_rate": candidate_parse_ok_count / candidate_event_count if candidate_event_count else None,
        "candidate_question_complete_count": candidate_complete_count,
        "candidate_question_complete_rate": candidate_complete_count / candidate_event_count if candidate_event_count else None,
        "candidate_questions_generated_count": candidate_question_count_sum,
        "candidate_questions_per_ask": candidate_question_count_sum / candidate_event_count if candidate_event_count else None,
        "selected_candidate_score_mean": (
            sum(selected_candidate_scores) / len(selected_candidate_scores) if selected_candidate_scores else None
        ),
        "confidence_parse_ok": confidence_parse_ok,
        "formulatable_confidence": formulatable_confidence,
        "confidence_rationale": confidence_rationale,
        "true_formulatable": true_formulatable if completed else None,
        "high_confidence_error": high_confidence_error if completed else None,
        "final_agent_raw_text": final_agent_raw_text,
        "agent_usage": agent_client.total_usage,
        "gap_search_usage": gap_search_client.total_usage,
        "detector_usage": detector_client.total_usage,
        "user_usage": user_client.total_usage,
        "judge_usage": judge_client.total_usage,
        "agent_estimated_cost_usd": agent_client.total_estimated_cost_usd,
        "gap_search_estimated_cost_usd": gap_search_client.total_estimated_cost_usd,
        "detector_estimated_cost_usd": detector_client.total_estimated_cost_usd,
        "user_estimated_cost_usd": user_client.total_estimated_cost_usd,
        "judge_estimated_cost_usd": judge_client.total_estimated_cost_usd,
        # 新增重试统计
        "agent_retry_total": agent_retry_total,
        "user_retry_total": user_retry_total,
        "agent_max_retries_exceeded": agent_max_retries_exceeded,
        "user_max_retries_exceeded": user_max_retries_exceeded,
        "agent_retry_limit_pass_through_count": agent_retry_limit_pass_through_count,
        "user_retry_limit_pass_through_count": user_retry_limit_pass_through_count,
        "run_dir": str(run_dir),
    }
    stat["estimated_cost_usd"] = (
        stat["agent_estimated_cost_usd"]
        + stat["gap_search_estimated_cost_usd"]
        + stat["detector_estimated_cost_usd"]
        + stat["user_estimated_cost_usd"]
        + stat["judge_estimated_cost_usd"]
    )
    stats_path.write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    append_live_event(
        run_dir,
        case_start_time,
        {
            "event": "case_done",
            "completed_ready_to_model": completed,
            "protocol_failed": protocol_failed,
            "protocol_failure_type": protocol_failure_type,
            "turn_count": stat["turn_count"],
            "attempted_turn_count": stat["attempted_turn_count"],
            "agent_question_turn_count": stat["agent_question_turn_count"],
            "agent_atomic_question_count": stat["agent_atomic_question_count"],
            "no_stop_or_maxturn": (not completed and not protocol_failed),
        },
    )
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
                "gap_search_call_sum": 0,
                "gap_search_valid_sum": 0,
                "gap_search_error_sum": 0,
                "gap_search_gap_sum": 0,
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
                "todo_event_sum": 0,
                "todo_parse_ok_sum": 0,
                "todo_audit_complete_sum": 0,
                "todo_selected_dimension_counts": {dim_id: 0 for dim_id in sorted(OR_DIMENSION_IDS)},
                "candidate_question_event_sum": 0,
                "candidate_question_parse_ok_sum": 0,
                "candidate_question_complete_sum": 0,
                "candidate_questions_generated_sum": 0,
                "selected_candidate_score_sum": 0.0,
                "selected_candidate_score_count": 0,
                "ready_confidence_count": 0,
                "ready_confidence_parse_ok_sum": 0,
                "ready_confidence_sum": 0.0,
                "ready_confidence_sq_error_sum": 0.0,
                "ready_true_formulatable_sum": 0,
                "ready_high_confidence_error_sum": 0,
                "stopping_status_counts": {},
                "rule_based_stopping_status_counts": {},
                "stopping_status_mismatch_sum": 0,
                "agent_usage": {},
                "gap_search_usage": {},
                "detector_usage": {},
                "user_usage": {},
                "judge_usage": {},
                "estimated_cost_usd": 0.0,
                "core_exact_restore_sum": 0,
                "core_exact_restore_applicable_runs": 0,
                "all_slot_exact_restore_sum": 0,
                "core_slot_sum": 0,
                "p2_slot_sum": 0,
                "core_unresolved_slot_sum": 0,
                "p2_unresolved_slot_sum": 0,
                # 新增重试聚合
                "agent_retry_total_sum": 0,
                "user_retry_total_sum": 0,
                "agent_max_retries_exceeded_sum": 0,
                "user_max_retries_exceeded_sum": 0,
                "agent_retry_limit_pass_through_sum": 0,
                "user_retry_limit_pass_through_sum": 0,
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
        group["gap_search_call_sum"] += int(stat.get("gap_search_call_count", 0) or 0)
        group["gap_search_valid_sum"] += int(stat.get("gap_search_valid_count", 0) or 0)
        group["gap_search_error_sum"] += int(stat.get("gap_search_error_count", 0) or 0)
        group["gap_search_gap_sum"] += int(stat.get("gap_search_gap_count", 0) or 0)
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
        group["todo_event_sum"] += int(stat.get("or_dimension_audit_event_count", 0) or 0)
        group["todo_parse_ok_sum"] += int(stat.get("todo_audit_parse_ok_count", 0) or 0)
        group["todo_audit_complete_sum"] += int(stat.get("todo_audit_complete_count", 0) or 0)
        selected_counts = stat.get("todo_selected_dimension_counts", {}) or {}
        for dim_id in sorted(OR_DIMENSION_IDS):
            group["todo_selected_dimension_counts"][dim_id] += int(selected_counts.get(dim_id, 0) or 0)
        group["candidate_question_event_sum"] += int(stat.get("candidate_question_event_count", 0) or 0)
        group["candidate_question_parse_ok_sum"] += int(stat.get("candidate_question_parse_ok_count", 0) or 0)
        group["candidate_question_complete_sum"] += int(stat.get("candidate_question_complete_count", 0) or 0)
        group["candidate_questions_generated_sum"] += int(stat.get("candidate_questions_generated_count", 0) or 0)
        if stat.get("selected_candidate_score_mean") is not None and stat.get("candidate_question_event_count"):
            event_count = int(stat.get("candidate_question_event_count", 0) or 0)
            group["selected_candidate_score_sum"] += float(stat["selected_candidate_score_mean"]) * event_count
            group["selected_candidate_score_count"] += event_count
        if stat.get("completed_ready_to_model"):
            group["ready_confidence_count"] += 1
            if stat.get("confidence_parse_ok"):
                group["ready_confidence_parse_ok_sum"] += 1
                conf = float(stat.get("formulatable_confidence", 0.0) or 0.0)
                truth = 1.0 if stat.get("true_formulatable") else 0.0
                group["ready_confidence_sum"] += conf
                group["ready_confidence_sq_error_sum"] += (conf - truth) ** 2
                group["ready_true_formulatable_sum"] += int(truth)
                group["ready_high_confidence_error_sum"] += 1 if stat.get("high_confidence_error") else 0
        status = stat.get("stopping_status") or "unknown"
        group["stopping_status_counts"][status] = group["stopping_status_counts"].get(status, 0) + 1
        rule_status = stat.get("rule_based_stopping_status") or status
        group["rule_based_stopping_status_counts"][rule_status] = (
            group["rule_based_stopping_status_counts"].get(rule_status, 0) + 1
        )
        group["stopping_status_mismatch_sum"] += 1 if stat.get("stopping_status_mismatch") else 0
        add_usage(group["agent_usage"], stat.get("agent_usage", {}))
        add_usage(group["gap_search_usage"], stat.get("gap_search_usage", {}))
        add_usage(group["detector_usage"], stat.get("detector_usage", {}))
        add_usage(group["user_usage"], stat.get("user_usage", {}))
        add_usage(group["judge_usage"], stat.get("judge_usage", {}))
        group["estimated_cost_usd"] += float(stat.get("estimated_cost_usd", 0) or 0)
        if stat.get("core_exact_applicable", int(stat.get("core_slot_count", 0) or 0) > 0):
            group["core_exact_restore_applicable_runs"] += 1
            group["core_exact_restore_sum"] += 1 if stat.get("core_exact_restore") else 0
        group["all_slot_exact_restore_sum"] += 1 if stat.get("all_slot_exact_restore") else 0
        group["core_slot_sum"] += int(stat.get("core_slot_count", 0) or 0)
        group["p2_slot_sum"] += int(stat.get("p2_slot_count", 0) or 0)
        group["core_unresolved_slot_sum"] += int(stat.get("core_unresolved_slot_count", 0) or 0)
        group["p2_unresolved_slot_sum"] += int(stat.get("p2_unresolved_slot_count", 0) or 0)

        # 重试聚合
        group["agent_retry_total_sum"] += int(stat.get("agent_retry_total", 0))
        group["user_retry_total_sum"] += int(stat.get("user_retry_total", 0))
        group["agent_max_retries_exceeded_sum"] += 1 if stat.get("agent_max_retries_exceeded") else 0
        group["user_max_retries_exceeded_sum"] += 1 if stat.get("user_max_retries_exceeded") else 0
        group["agent_retry_limit_pass_through_sum"] += int(stat.get("agent_retry_limit_pass_through_count", 0) or 0)
        group["user_retry_limit_pass_through_sum"] += int(stat.get("user_retry_limit_pass_through_count", 0) or 0)

    summary = {"profiles": {}, "total_estimated_cost_usd": 0.0}
    for profile, group in groups.items():
        runs = max(1, group["runs"])
        core_exact_denominator = max(1, group["core_exact_restore_applicable_runs"])
        score_count = max(1, group["weighted_slot_score_count"])
        weighted_micro = (
            group["earned_weight_sum"] / group["total_weight_sum"] if group["total_weight_sum"] else None
        )
        question_turns = max(1, group["agent_question_turn_sum"])
        todo_events = max(1, group["todo_event_sum"])
        candidate_events = max(1, group["candidate_question_event_sum"])
        selected_score_count = max(1, group["selected_candidate_score_count"])
        ready_confidence_count = max(1, group["ready_confidence_count"])
        parsed_confidence_count = max(1, group["ready_confidence_parse_ok_sum"])
        summary["profiles"][profile] = {
            "agent_model": group["agent_model"],
            "runs": group["runs"],
            "metrics": {
                "WeightedSlotScore_macro": group["weighted_slot_score_sum"] / score_count,
                "WeightedSlotScore_micro": weighted_micro,
                "AverageTurns": group["turn_sum"] / runs,
                "AverageAttemptedTurns": group["attempted_turn_sum"] / runs,
                "ReadyToModelRate": group["ready_to_model_sum"] / runs,
                "CoreExactRestoreRate": group["core_exact_restore_sum"] / core_exact_denominator,
                "CoreExactRestoreCount": group["core_exact_restore_sum"],
                "CoreExactRestoreDenominator": group["core_exact_restore_applicable_runs"],
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
                "GapSearchCallsPerRun": group["gap_search_call_sum"] / runs,
                "GapSearchValidRate": group["gap_search_valid_sum"] / max(1, group["gap_search_call_sum"]),
                "GapSearchErrorsPerRun": group["gap_search_error_sum"] / runs,
                "GapSearchGapsPerRun": group["gap_search_gap_sum"] / runs,
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
                "TodoAuditParseOkRate": group["todo_parse_ok_sum"] / todo_events,
                "TodoAuditCompleteRate": group["todo_audit_complete_sum"] / todo_events,
                "TodoSelectedDimensionCounts": group["todo_selected_dimension_counts"],
                "CandidateQuestionEventsPerRun": group["candidate_question_event_sum"] / runs,
                "CandidateQuestionParseOkRate": group["candidate_question_parse_ok_sum"] / candidate_events,
                "CandidateQuestionCompleteRate": group["candidate_question_complete_sum"] / candidate_events,
                "CandidateQuestionsPerAsk": group["candidate_questions_generated_sum"] / candidate_events,
                "SelectedCandidateScoreMean": (
                    group["selected_candidate_score_sum"] / selected_score_count
                    if group["selected_candidate_score_count"]
                    else None
                ),
                "ReadyConfidenceCount": group["ready_confidence_count"],
                "ReadyConfidenceParseOkRate": group["ready_confidence_parse_ok_sum"] / ready_confidence_count,
                "MeanReadyFormulatableConfidence": (
                    group["ready_confidence_sum"] / parsed_confidence_count
                    if group["ready_confidence_parse_ok_sum"]
                    else None
                ),
                "ReadyTrueFormulatableRate": group["ready_true_formulatable_sum"] / parsed_confidence_count
                    if group["ready_confidence_parse_ok_sum"]
                    else None,
                "ReadyHighConfidenceErrorRate": group["ready_high_confidence_error_sum"] / parsed_confidence_count
                    if group["ready_confidence_parse_ok_sum"]
                    else None,
                "ReadyConfidenceBrier": group["ready_confidence_sq_error_sum"] / parsed_confidence_count
                    if group["ready_confidence_parse_ok_sum"]
                    else None,
                "StoppingStatusCounts": group["stopping_status_counts"],
                "RuleBasedStoppingStatusCounts": group["rule_based_stopping_status_counts"],
                "StoppingStatusMismatchRate": group["stopping_status_mismatch_sum"] / runs,
                # 新增重试指标
                "AvgAgentRetriesPerRun": group["agent_retry_total_sum"] / runs,
                "AgentMaxRetriesExceededRate": group["agent_max_retries_exceeded_sum"] / runs,
            },
            "agent_usage": group["agent_usage"],
            "gap_search_usage": group["gap_search_usage"],
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
        f"- TOML dir: `{args.toml_dir}`",
        f"- K: `{args.k}`",
        f"- max turns: `{args.max_turns}`",
        f"- agent profiles: `{', '.join(args.agent_profiles)}`",
        f"- detector profile: `{args.detector_profile}`",
        f"- user profile: `{args.user_profile}`",
        f"- judge profile: `{args.judge_profile}`",
        f"- prompts dir: `{args.prompts_dir}`",
        f"- Max agent retries per turn: `{MAX_AGENT_RETRIES_PER_TURN}`",
        f"- Max user retries per turn: `{MAX_USER_RETRIES_PER_TURN}`",
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
        lines.append(
            f"| {profile} | {item['runs']} | {m['AllSlotExactRestoreRate']:.3f} ({m['AllSlotExactRestoreCount']}) | "
            f"{m['CoreExactRestoreRate']:.3f} ({m['CoreExactRestoreCount']}) | "
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
    lines.extend(["", "## Candidate Question Selection and READY Confidence", ""])
    for profile, item in summary["profiles"].items():
        m = item["metrics"]
        mean_conf = m["MeanReadyFormulatableConfidence"]
        true_rate = m["ReadyTrueFormulatableRate"]
        high_conf_error = m["ReadyHighConfidenceErrorRate"]
        brier = m["ReadyConfidenceBrier"]
        lines.extend(
            [
                f"### {profile}",
                "",
                f"- candidate question events per run: `{m['CandidateQuestionEventsPerRun']:.3f}`",
                f"- candidate question parse rate: `{m['CandidateQuestionParseOkRate']:.3f}`",
                f"- candidate question complete rate: `{m['CandidateQuestionCompleteRate']:.3f}`",
                f"- candidate questions per ASK: `{m['CandidateQuestionsPerAsk']:.3f}`",
                f"- selected candidate score mean: `{m['SelectedCandidateScoreMean']:.3f}`" if m["SelectedCandidateScoreMean"] is not None else "- selected candidate score mean: `NA`",
                f"- READY confidence count: `{m['ReadyConfidenceCount']}`",
                f"- READY confidence parse rate: `{m['ReadyConfidenceParseOkRate']:.3f}`",
                f"- mean READY confidence: `{mean_conf:.3f}`" if mean_conf is not None else "- mean READY confidence: `NA`",
                f"- READY true formulatable rate: `{true_rate:.3f}`" if true_rate is not None else "- READY true formulatable rate: `NA`",
                f"- high-confidence error rate: `{high_conf_error:.3f}`" if high_conf_error is not None else "- high-confidence error rate: `NA`",
                f"- Brier score: `{brier:.3f}`" if brier is not None else "- Brier score: `NA`",
                "",
            ]
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


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_prompts_dir = script_dir / "prompts"

    parser = argparse.ArgumentParser()
    parser.add_argument("--toml_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--case_timeout_sec", type=float, default=None)
    parser.add_argument("--agent_profiles", nargs="+", default=["deepseek_v4_pro"])
    parser.add_argument("--detector_profile", default="deepseek_v4_pro")
    parser.add_argument("--user_profile", default="deepseek_v4_pro")
    parser.add_argument("--judge_profile", default="deepseek_v4_pro")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--detector_temperature", type=float, default=0.0)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--prompts_dir", default=str(default_prompts_dir))
    # 可选：允许从命令行覆盖重试次数
    parser.add_argument("--max_agent_retries", type=int, default=3)
    parser.add_argument("--max_user_retries", type=int, default=3)
    parser.add_argument(
        "--monitor_user_answers",
        action="store_true",
        help="Also use the independent detector to check whether the simulated user answers only the current question.",
    )
    parser.add_argument(
        "--detector_feedback_mode",
        choices=["visible", "minimal", "none"],
        default="visible",
        help="visible injects detailed feedback; minimal injects short feedback; none silently discards invalid attempts and resamples.",
    )
    parser.add_argument(
        "--retry_limit_behavior",
        choices=["pass_through", "protocol_failed"],
        default="pass_through",
        help="pass_through continues after retry limit; protocol_failed stops the current case after retry limit.",
    )
    parser.add_argument(
        "--fixed_both_sides_silent",
        action="store_true",
        help=(
            "Shortcut for monitor_user_answers + no detector feedback + protocol_failed retry behavior. "
            "This is the official fixed_both_sides_silent variant."
        ),
    )
    args = parser.parse_args()
    if args.fixed_both_sides_silent:
        args.monitor_user_answers = True
        args.detector_feedback_mode = "none"
        args.retry_limit_behavior = "protocol_failed"
        args.pipeline_mode = "fixed_both_sides_silent"
    elif args.monitor_user_answers:
        args.pipeline_mode = "fixed_both_sides"
    else:
        args.pipeline_mode = "fixed_question_only"

    # 使用命令行参数覆盖全局重试限制
    global MAX_AGENT_RETRIES_PER_TURN, MAX_USER_RETRIES_PER_TURN
    MAX_AGENT_RETRIES_PER_TURN = args.max_agent_retries
    MAX_USER_RETRIES_PER_TURN = args.max_user_retries

    prompts = load_prompt_files(Path(args.prompts_dir))
    cases = load_cases(Path(args.toml_dir), args.limit)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    stats: list[dict[str, Any]] = []
    for profile in args.agent_profiles:
        for case in cases:
            for run_index in range(1, args.k + 1):
                print(
                    f"STANDALONE agent={profile} case={case['_case_id']} run={run_index}/{args.k}",
                    flush=True,
                )
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
                    case_timeout_sec=args.case_timeout_sec,
                )
                stats.append(stat)
    summary = summarize(stats, output_root)
    write_markdown_report(summary, output_root, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
