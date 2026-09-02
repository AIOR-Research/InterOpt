"""Persistent gap-ledger control for the InterOPT w/o Stage 2 pipeline.

Design contract:

- Two states only: OPEN / ASKED. No LLM-judged RESOLVED/UNANSWERABLE states.
- The gap-search stage may only ADD new gaps (at most 3 per turn); state
  transitions to ASKED happen programmatically when a candidate question bound
  to a gap is actually asked in the public transcript.
- No counterfactual branches, no materiality verdicts, no formulation deltas:
  every OPEN gap is eligible.
"""
from __future__ import annotations

import copy
import re
from typing import Any

LEDGER_STATUSES = {"OPEN", "ASKED"}
LEDGER_CATEGORIES = {
    "objective",
    "decision_scope",
    "constraint_set",
    "time_boundary",
    "entity_relation",
    "hard_soft_policy",
}
LEDGER_DECISIONS = {"CONTINUE", "NO_NEW_GAP"}
MAX_LEDGER_ADDS_PER_TURN = 3


def normalize_description_text(text: str) -> str:
    """Cheap literal-level normalizer used only for duplicate counting.

    This is a conservative lower bound: it catches verbatim re-reports, not
    semantic paraphrases (those are tolerated and reviewed in smoke tests).
    """
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _required_text(raw: dict[str, Any], field: str) -> str:
    value = str(raw.get(field, "")).strip()
    if not value:
        raise ValueError(f"ledger update field {field!r} must be non-empty")
    return value


def normalize_ledger_search_result(
    result: dict[str, Any], existing_descriptions: set[str]
) -> dict[str, Any]:
    """Validate one gap-search response for the ledger pipeline.

    Only ADD operations exist. Duplicate descriptions (literal level, compared
    against existing_descriptions) are dropped and counted instead of raising.
    """
    decision = str(result.get("decision", "")).strip().upper()
    if decision not in LEDGER_DECISIONS:
        raise ValueError("ledger decision must be CONTINUE or NO_NEW_GAP")

    updates = result.get("updates", [])
    if not isinstance(updates, list):
        raise ValueError("ledger updates must be a list")
    if len(updates) > MAX_LEDGER_ADDS_PER_TURN:
        raise ValueError(
            f"ledger updates must contain at most {MAX_LEDGER_ADDS_PER_TURN} items"
        )

    normalized_adds: list[dict[str, Any]] = []
    duplicate_skipped = 0
    seen_this_turn: set[str] = set()
    for index, raw in enumerate(updates, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"ledger update {index} must be an object")
        operation = str(raw.get("operation", "")).strip().upper()
        if operation != "ADD":
            raise ValueError(
                f"ledger update {index} has invalid operation {operation!r}: only ADD is supported"
            )
        category = str(raw.get("category", "")).strip()
        if category not in LEDGER_CATEGORIES:
            raise ValueError(f"ledger update {index} has invalid category")
        description = _required_text(raw, "description")
        evidence_quote = _required_text(raw, "evidence_quote")
        normalized_desc = normalize_description_text(description)
        if (
            normalized_desc in existing_descriptions
            or normalized_desc in seen_this_turn
        ):
            duplicate_skipped += 1
            continue
        seen_this_turn.add(normalized_desc)
        normalized_adds.append(
            {
                "operation": "ADD",
                "category": category,
                "description": description,
                "evidence_quote": evidence_quote,
            }
        )

    if decision == "NO_NEW_GAP" and normalized_adds:
        raise ValueError("NO_NEW_GAP cannot introduce new gaps")

    return {
        "decision": decision,
        "search_summary": _required_text(result, "search_summary"),
        "updates": normalized_adds,
        "duplicate_skipped": duplicate_skipped,
    }


def _next_gap_id(frontier: list[dict[str, Any]]) -> str:
    numbers = []
    for gap in frontier:
        gap_id = str(gap.get("gap_id", ""))
        if gap_id.startswith("G") and gap_id[1:].isdigit():
            numbers.append(int(gap_id[1:]))
    return f"G{max(numbers, default=0) + 1:03d}"


def apply_ledger_adds(
    frontier: list[dict[str, Any]], adds: list[dict[str, Any]], turn: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append ADD rows to the persistent frontier. Returns (merged, applied)."""
    merged = copy.deepcopy(frontier)
    applied: list[dict[str, Any]] = []
    for add in adds:
        row = copy.deepcopy(add)
        row.pop("operation", None)
        gap_id = _next_gap_id(merged)
        row.update(
            {
                "gap_id": gap_id,
                "status": "OPEN",
                "created_turn": turn,
                "updated_turn": turn,
                "asked_turn": None,
            }
        )
        merged.append(row)
        applied.append({"operation": "ADD", "gap_id": gap_id, "status": "OPEN"})
    return merged, applied


def mark_gap_asked(
    frontier: list[dict[str, Any]], gap_id: str, turn: int
) -> list[dict[str, Any]]:
    """Programmatic consumption: an asked question consumes its OPEN gap."""
    merged = copy.deepcopy(frontier)
    target = str(gap_id).strip().upper()
    for gap in merged:
        if str(gap.get("gap_id")) == target:
            if gap.get("status") != "OPEN":
                raise ValueError(f"gap {target} is not OPEN (status={gap.get('status')})")
            gap["status"] = "ASKED"
            gap["asked_turn"] = turn
            gap["updated_turn"] = turn
            return merged
    raise ValueError(f"unknown gap id: {target}")


def open_gaps(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every OPEN gap is eligible for binding. No materiality gating."""
    return [gap for gap in frontier if gap.get("status") == "OPEN"]
