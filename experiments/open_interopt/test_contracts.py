import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline as pipeline  # noqa: E402


def gap(gap_id: str, question: str) -> dict[str, str]:
    return {
        "gap_id": gap_id,
        "category": "objective",
        "description": f"Unresolved business fact {gap_id}.",
        "direct_question": question,
        "source_status": "not stated",
        "why_material": "Different answers change the formulation.",
    }


class NoStage2ContractTests(unittest.TestCase):
    def test_ready_to_model_must_be_boolean(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.normalize_gap_search_result(
                {"search_summary": "done", "ready_to_model": "false", "gaps": []}
            )

    def test_ready_to_model_true_rejects_residual_gaps(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.normalize_gap_search_result(
                {
                    "search_summary": "Contradictory output.",
                    "ready_to_model": True,
                    "gaps": [gap("G1", "Which objective should be optimized?")],
                }
            )

    def test_ready_to_model_true_stops_without_fixed_confidence(self) -> None:
        result = pipeline.normalize_gap_search_result(
            {
                "search_summary": "No formulation-changing gap remains.",
                "ready_to_model": True,
                "gaps": [],
            }
        )
        raw = pipeline.build_gap_direct_agent_output(result)
        parsed = pipeline.parse_agent_output(raw)
        self.assertEqual(parsed["action"], "READY_TO_MODEL")
        self.assertIsNone(parsed["formulatable_confidence"])
        self.assertFalse(parsed["confidence_parse_ok"])
        self.assertNotIn("0.8", raw)

    def test_non_ready_with_gap_asks_instead_of_stopping(self) -> None:
        result = pipeline.normalize_gap_search_result(
            {
                "search_summary": "The objective remains unresolved.",
                "ready_to_model": False,
                "gaps": [gap("G1", "Which objective should be optimized?")],
            }
        )
        raw = pipeline.build_gap_direct_agent_output(result)
        parsed = pipeline.parse_agent_output(raw)
        self.assertEqual(parsed["action"], "ASK")
        self.assertEqual(parsed["public_question"], "Which objective should be optimized?")

    def test_code_fixed_selects_first_gap_when_multiple_gaps_exist(self) -> None:
        result = pipeline.normalize_gap_search_result(
            {
                "search_summary": "Two gaps remain.",
                "ready_to_model": False,
                "gaps": [
                    gap("G1", "Which objective should be optimized?"),
                    gap("G2", "Which capacity limit is binding?"),
                ],
            }
        )
        raw = pipeline.build_gap_direct_agent_output(result)
        parsed = pipeline.parse_agent_output(raw)
        self.assertEqual(parsed["action"], "ASK")
        self.assertEqual(parsed["public_question"], "Which objective should be optimized?")

    def test_detector_accepts_question_actions_and_ready(self) -> None:
        self.assertTrue(
            pipeline.detector_accepts_agent_action(
                {
                    "action": "question",
                    "question_count": 1,
                    "is_atomic": True,
                    "atomic_questions": ["Which objective should be optimized?"],
                }
            )
        )
        self.assertTrue(
            pipeline.detector_accepts_agent_action(
                {
                    "action": "question",
                    "question_count": 2,
                    "is_atomic": False,
                    "atomic_questions": ["Question one?", "Question two?"],
                }
            )
        )
        self.assertTrue(
            pipeline.detector_accepts_agent_action(
                {
                    "action": "ready_to_model",
                    "question_count": 0,
                    "is_atomic": True,
                    "atomic_questions": [],
                }
            )
        )
        self.assertFalse(
            pipeline.detector_accepts_agent_action(
                {
                    "action": "invalid",
                    "question_count": 0,
                    "is_atomic": False,
                    "atomic_questions": [],
                }
            )
        )

    def test_prompt_and_feedback_do_not_require_single_atomic_question(self) -> None:
        prompt = Path(__file__).resolve().parent.joinpath("prompts", "gap_search_prompt.md").read_text(
            encoding="utf-8"
        )
        protocol_feedback = pipeline.build_agent_protocol_feedback()
        minimal_feedback = pipeline.build_minimal_agent_retry_feedback()

        combined = "\n".join([prompt, protocol_feedback, minimal_feedback]).lower()
        self.assertNotIn("must be one atomic", combined)
        self.assertNotIn("exactly one minimal", combined)
        self.assertNotIn("one-question-per-turn", combined)


if __name__ == "__main__":
    unittest.main()
