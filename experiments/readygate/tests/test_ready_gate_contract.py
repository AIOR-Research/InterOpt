from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
GATE_SENTINEL = "GATE_SENTINEL_SHOULD_NOT_LEAK"
SPEC = importlib.util.spec_from_file_location("readygate", ROOT / "run_pipeline.py")
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


VALID_ASK = json.dumps(
    {
        "action": "ASK",
        "question": "Must every location be visited?",
        "options": [
            {"id": "A", "text": "Every location must be visited."},
            {"id": "B", "text": "Only selected locations need to be visited."},
            {"id": "C", "text": "Locations are optional with a penalty."},
        ],
        "allow_other": True,
    }
)
VALID_READY = json.dumps({"action": "READY_TO_MODEL", "summary": "Confirmed route requirements."})
VALID_CHOICE_DETECTOR = (
    '{"action":"ask","question_count":1,"is_valid_mc":true,'
    '"question":"Must every location be visited?",'
    '"option_count":3,"option_ids":["A","B","C"],"allow_other":true,'
    '"rationale":"one valid mc_d question"}'
)
READY_DETECTOR = (
    '{"action":"ready_to_model","question_count":0,"is_valid_mc":true,'
    '"question":"","option_count":0,"option_ids":[],"allow_other":false,'
    '"rationale":"stop"}'
)
USER_CHOICE = json.dumps(
    {
        "choice": "A",
        "rationale": "Option A matches the private route requirement.",
        "match_status": "exact_match",
    }
)
GATE_BLOCK = (
    "Decision: BLOCK\n"
    "Feedback: Clarify the visit coverage policy before stopping. "
    f"{GATE_SENTINEL}"
)
GATE_PASS = "Decision: PASS\nFeedback: The public dialogue is sufficient to stop."
JUDGE_RESULT = (
    '{"slot_scores":[{"slot_id":"H1","name":"visit_policy","severity":"P0",'
    '"hit":"yes","evidence_location":"turn 1","evidence_quote":"visited",'
    '"rationale":"recovered"}],"silent_assumptions":[],"stopping_behavior":'
    '{"status":"appropriate_stop","unresolved_p0_slots":[],"unresolved_p1_slots":[],'
    '"rationale":"complete"},"summary":"complete"}'
)


class FakeChatClient:
    created: list["FakeChatClient"] = []
    responses: list[list[str]] = []

    def __init__(self, profile_name: str, temperature: float):
        self.profile_name = profile_name
        self.model = profile_name
        self.index = len(self.created)
        self.queue = list(self.responses[self.index])
        self.seen_messages: list[list[dict[str, str]]] = []
        self.total_usage: dict[str, int] = {}
        self.total_estimated_cost_usd = 0.0
        self.created.append(self)

    def complete(self, messages: list[dict[str, str]], timeout: int = 180, max_retries: int = 6):
        self.seen_messages.append([dict(message) for message in messages])
        return PIPELINE.ChatResult(content=self.queue.pop(0), usage={}, estimated_cost_usd=0.0)


def mock_case() -> dict:
    return {
        "_case_id": "mock_case",
        "_path": "mock.toml",
        "initial_brief": {"content": "Plan a route.", "visible_unit_ids": ["U1"]},
        "problem_units": [{"id": "U1", "kind": "request", "content": "Plan a route."}],
        "hidden_slots": [
            {
                "slot_id": "H1",
                "name": "visit_policy",
                "severity": "P0",
                "severity_reason": "Changes the model.",
                "problem_unit_id": "U1",
                "semantic_hit_rule": "Ask whether every location must be visited.",
                "reference_acceptable_questions": ["Must every location be visited?"],
                "failure_modes": ["Assume optional visits."],
                "simulator_answer": "Every location must be visited.",
            }
        ],
        "simulator": {"business_role": "route manager"},
    }


class ReadyGateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeChatClient.created = []
        PIPELINE.MAX_AGENT_RETRIES_PER_TURN = 0
        PIPELINE.MAX_USER_RETRIES_PER_TURN = 0

    def test_defaults_are_experiment_2_mc_d_only(self) -> None:
        args = PIPELINE.build_argument_parser().parse_args([])

        self.assertEqual(args.interaction_modes, ["mc_d"])
        self.assertEqual(args.max_turns, 20)
        self.assertEqual(args.ready_gate_profile, "ready_gate")
        self.assertEqual(PIPELINE.determine_pipeline_mode(args), "readygate")
        self.assertTrue(PIPELINE.make_default_output_dir().name.startswith("readygate_"))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            PIPELINE.build_argument_parser().parse_args(["--interaction_modes", "mc"])

    def test_ready_gate_parser_accepts_plain_text_contract(self) -> None:
        parsed = PIPELINE.parse_ready_gate_response(GATE_BLOCK)

        self.assertEqual(parsed["decision"], "BLOCK")
        self.assertIn(GATE_SENTINEL, parsed["feedback"])
        with self.assertRaisesRegex(ValueError, "Decision"):
            PIPELINE.parse_ready_gate_response("Feedback: missing decision")

    def test_block_feedback_enters_only_agent_context_then_passes_later(self) -> None:
        FakeChatClient.responses = [
            [VALID_READY, VALID_ASK, VALID_READY],
            [READY_DETECTOR, VALID_CHOICE_DETECTOR, READY_DETECTOR],
            [GATE_BLOCK, GATE_PASS],
            [USER_CHOICE],
            [JUDGE_RESULT],
        ]
        prompts = PIPELINE.load_prompt_bundle(ROOT / "prompts", "mc_d")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PIPELINE, "ChatClient", FakeChatClient
        ):
            stat = PIPELINE.run_interaction(
                case=mock_case(),
                agent_profile="generic_agent",
                detector_profile="detector",
                ready_gate_profile="ready_gate",
                user_profile="user_simulator",
                judge_profile="judge",
                run_index=1,
                output_root=Path(temp_dir),
                prompts=prompts,
                interaction_mode="mc_d",
                agent_temperature=0.2,
                detector_temperature=0.0,
                ready_gate_temperature=0.0,
                user_temperature=0.0,
                judge_temperature=0.0,
                max_turns=5,
            )
            run_dir = Path(stat["run_dir"])
            transcript_text = (run_dir / "transcript.json").read_text(encoding="utf-8")
            judge_message = (run_dir / "judge_prompt_user_message.md").read_text(encoding="utf-8")
            gate_events = json.loads(
                (run_dir / PIPELINE.READY_GATE_AUDIT_FILENAME).read_text(encoding="utf-8")
            )

        self.assertTrue(stat["completed_ready_to_model"])
        self.assertEqual(stat["ready_gate_call_count"], 2)
        self.assertEqual(stat["ready_gate_block_count"], 1)
        self.assertEqual(stat["ready_gate_pass_count"], 1)
        self.assertEqual(stat["ready_gate_final_status"], "passed_ready")
        self.assertEqual([event["effective_decision"] for event in gate_events], ["BLOCK", "PASS"])
        self.assertFalse(gate_events[0]["entered_public_transcript"])
        self.assertTrue(gate_events[0]["entered_agent_context"])
        self.assertNotIn(GATE_SENTINEL, transcript_text)
        self.assertNotIn(GATE_SENTINEL, judge_message)

        agent_seen = json.dumps(FakeChatClient.created[0].seen_messages, ensure_ascii=False)
        detector_seen = json.dumps(FakeChatClient.created[1].seen_messages, ensure_ascii=False)
        user_seen = json.dumps(FakeChatClient.created[3].seen_messages, ensure_ascii=False)
        self.assertIn(GATE_SENTINEL, agent_seen)
        self.assertNotIn(GATE_SENTINEL, detector_seen)
        self.assertNotIn(GATE_SENTINEL, user_seen)

    def test_gate_parse_failure_defaults_to_pass_with_audit(self) -> None:
        FakeChatClient.responses = [
            [VALID_READY],
            [READY_DETECTOR],
            ["this is not the contract"],
            [],
            [JUDGE_RESULT],
        ]
        prompts = PIPELINE.load_prompt_bundle(ROOT / "prompts", "mc_d")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PIPELINE, "ChatClient", FakeChatClient
        ):
            stat = PIPELINE.run_interaction(
                case=mock_case(),
                agent_profile="generic_agent",
                detector_profile="detector",
                ready_gate_profile="ready_gate",
                user_profile="user_simulator",
                judge_profile="judge",
                run_index=1,
                output_root=Path(temp_dir),
                prompts=prompts,
                interaction_mode="mc_d",
                agent_temperature=0.2,
                detector_temperature=0.0,
                ready_gate_temperature=0.0,
                user_temperature=0.0,
                judge_temperature=0.0,
                max_turns=2,
            )
            gate_events = json.loads(
                (Path(stat["run_dir"]) / PIPELINE.READY_GATE_AUDIT_FILENAME).read_text(encoding="utf-8")
            )

        self.assertTrue(stat["completed_ready_to_model"])
        self.assertEqual(stat["ready_gate_error_count"], 1)
        self.assertEqual(stat["ready_gate_final_status"], "gate_error_pass_through")
        self.assertTrue(gate_events[0]["default_pass_due_to_error"])
        self.assertIn("parse_error", gate_events[0])

    def test_max_turns_exhaustion_after_repeated_blocks_is_not_protocol_failure(self) -> None:
        FakeChatClient.responses = [
            [VALID_READY, VALID_READY],
            [READY_DETECTOR, READY_DETECTOR],
            [GATE_BLOCK, GATE_BLOCK],
            [],
            [JUDGE_RESULT],
        ]
        prompts = PIPELINE.load_prompt_bundle(ROOT / "prompts", "mc_d")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PIPELINE, "ChatClient", FakeChatClient
        ):
            stat = PIPELINE.run_interaction(
                case=mock_case(),
                agent_profile="generic_agent",
                detector_profile="detector",
                ready_gate_profile="ready_gate",
                user_profile="user_simulator",
                judge_profile="judge",
                run_index=1,
                output_root=Path(temp_dir),
                prompts=prompts,
                interaction_mode="mc_d",
                agent_temperature=0.2,
                detector_temperature=0.0,
                ready_gate_temperature=0.0,
                user_temperature=0.0,
                judge_temperature=0.0,
                max_turns=2,
            )

        self.assertFalse(stat["completed_ready_to_model"])
        self.assertFalse(stat["protocol_failed"])
        self.assertEqual(stat["ready_gate_block_count"], 2)
        self.assertEqual(stat["ready_gate_final_status"], "max_turns_reached_without_pass")


if __name__ == "__main__":
    unittest.main()
