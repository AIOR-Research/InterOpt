from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier_control import (
    MAX_LEDGER_ADDS_PER_TURN,
    apply_ledger_adds,
    mark_gap_asked,
    normalize_description_text,
    normalize_ledger_search_result,
    open_gaps,
)


def add_row(category: str = "constraint_set", description: str = "Gap X", evidence: str = "ev") -> dict:
    return {
        "operation": "ADD",
        "category": category,
        "description": description,
        "evidence_quote": evidence,
    }


def result(updates: list, decision: str = "CONTINUE", summary: str = "s") -> dict:
    return {"decision": decision, "search_summary": summary, "updates": updates}


class NormalizeLedgerSearchResultTests(unittest.TestCase):
    def test_valid_single_add(self) -> None:
        out = normalize_ledger_search_result(result([add_row()]), set())
        self.assertEqual(len(out["updates"]), 1)
        self.assertEqual(out["duplicate_skipped"], 0)
        self.assertEqual(out["decision"], "CONTINUE")

    def test_invalid_decision_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result([], decision="STOP"), set())

    def test_updates_must_be_list(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(
                {"decision": "CONTINUE", "search_summary": "s", "updates": "x"}, set()
            )

    def test_at_most_three_adds(self) -> None:
        rows = [add_row(description=f"Gap {i}") for i in range(MAX_LEDGER_ADDS_PER_TURN + 1)]
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result(rows), set())

    def test_update_operation_rejected(self) -> None:
        row = add_row()
        row["operation"] = "UPDATE"
        row["target_gap_id"] = "G001"
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result([row]), set())

    def test_invalid_category_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result([add_row(category="solver_detail")]), set())

    def test_empty_description_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result([add_row(description="  ")]), set())

    def test_empty_evidence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result([add_row(evidence=" ")]), set())

    def test_empty_summary_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(result([], summary=" "), set())

    def test_duplicate_against_ledger_dropped(self) -> None:
        existing = {normalize_description_text("Whether weekends count.")}
        out = normalize_ledger_search_result(
            result([add_row(description="Whether weekends count.")]), existing
        )
        self.assertEqual(out["updates"], [])
        self.assertEqual(out["duplicate_skipped"], 1)

    def test_duplicate_within_same_turn_dropped(self) -> None:
        rows = [add_row(description="Same gap."), add_row(description="Same gap.")]
        out = normalize_ledger_search_result(result(rows), set())
        self.assertEqual(len(out["updates"]), 1)
        self.assertEqual(out["duplicate_skipped"], 1)

    def test_duplicate_case_and_whitespace_insensitive(self) -> None:
        existing = {normalize_description_text("weekend policy")}
        out = normalize_ledger_search_result(
            result([add_row(description="  Weekend   Policy ")]), existing
        )
        self.assertEqual(out["duplicate_skipped"], 1)

    def test_no_new_gap_with_adds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_ledger_search_result(
                result([add_row()], decision="NO_NEW_GAP"), set()
            )

    def test_no_new_gap_empty_updates_valid(self) -> None:
        out = normalize_ledger_search_result(result([], decision="NO_NEW_GAP"), set())
        self.assertEqual(out["decision"], "NO_NEW_GAP")
        self.assertEqual(out["updates"], [])

    def test_empty_updates_with_continue_valid(self) -> None:
        out = normalize_ledger_search_result(result([]), set())
        self.assertEqual(out["updates"], [])


class ApplyLedgerAddsTests(unittest.TestCase):
    def test_gap_ids_increment(self) -> None:
        merged, applied = apply_ledger_adds(
            [], [add_row(description="a"), add_row(description="b")], turn=1
        )
        self.assertEqual([g["gap_id"] for g in merged], ["G001", "G002"])
        self.assertEqual(len(applied), 2)

    def test_rows_carry_open_status_and_turns(self) -> None:
        merged, _ = apply_ledger_adds([], [add_row()], turn=3)
        row = merged[0]
        self.assertEqual(row["status"], "OPEN")
        self.assertEqual(row["created_turn"], 3)
        self.assertEqual(row["updated_turn"], 3)
        self.assertIsNone(row["asked_turn"])

    def test_input_frontier_not_mutated(self) -> None:
        original = [{"gap_id": "G001", "status": "OPEN", "description": "old"}]
        merged, _ = apply_ledger_adds(original, [add_row()], turn=1)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(merged), 2)

    def test_gap_id_continues_after_existing(self) -> None:
        existing = [{"gap_id": "G003", "status": "ASKED", "description": "old"}]
        merged, _ = apply_ledger_adds(existing, [add_row()], turn=1)
        self.assertEqual(merged[-1]["gap_id"], "G004")


class MarkGapAskedTests(unittest.TestCase):
    def _frontier(self) -> list:
        merged, _ = apply_ledger_adds([], [add_row(description="a")], turn=1)
        return merged

    def test_open_to_asked(self) -> None:
        merged = mark_gap_asked(self._frontier(), "G001", turn=2)
        self.assertEqual(merged[0]["status"], "ASKED")
        self.assertEqual(merged[0]["asked_turn"], 2)
        self.assertEqual(merged[0]["updated_turn"], 2)

    def test_double_consume_rejected(self) -> None:
        merged = mark_gap_asked(self._frontier(), "G001", turn=2)
        with self.assertRaises(ValueError):
            mark_gap_asked(merged, "G001", turn=3)

    def test_unknown_gap_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mark_gap_asked(self._frontier(), "G999", turn=1)


class OpenGapsTests(unittest.TestCase):
    def test_filters_asked(self) -> None:
        merged, _ = apply_ledger_adds(
            [], [add_row(description="a"), add_row(description="b")], turn=1
        )
        merged = mark_gap_asked(merged, "G001", turn=2)
        remaining = open_gaps(merged)
        self.assertEqual([g["gap_id"] for g in remaining], ["G002"])


class NormalizeDescriptionTextTests(unittest.TestCase):
    def test_normalization(self) -> None:
        self.assertEqual(normalize_description_text("  Hello   World "), "hello world")


if __name__ == "__main__":
    unittest.main()
