from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_pipeline as pipeline  # noqa: E402
from frontier_control import apply_ledger_adds, mark_gap_asked  # noqa: E402


def raise_value_error(message: str) -> None:
    raise ValueError(message)


def question(text: str = "Which policy should apply?") -> dict:
    return {
        "question": text,
        "options": [
            {"id": "A", "text": "Policy A applies."},
            {"id": "B", "text": "Policy B applies."},
            {"id": "C", "text": "Policy C applies."},
        ],
        "allow_other": True,
    }


def add_update(
    ref: str = "N1",
    description: str = "Whether weekend work is allowed.",
) -> dict:
    return {
        "local_ref": ref,
        "operation": "ADD",
        "category": "constraint_set",
        "description": description,
        "evidence_quote": "The public brief does not state this.",
    }


def ask_result(
    ref: str = "N1",
    updates: list[dict] | None = None,
    text: str = "Which policy should apply?",
) -> dict:
    return {
        "action": "ASK",
        "search_summary": "One business-policy gap remains.",
        "updates": [add_update()] if updates is None else updates,
        "selected_gap_ref": ref,
        "public_question": question(text),
    }


def ready_result(updates: list[dict] | None = None) -> dict:
    return {
        "action": "READY_TO_MODEL",
        "search_summary": "No further clarification is required.",
        "updates": [] if updates is None else updates,
    }


class DirectLedgerNormalizationTests(unittest.TestCase):
    def test_new_local_ref_maps_to_persistent_gap(self) -> None:
        normalized, frontier, applied = pipeline.normalize_direct_ledger_result(
            ask_result(), [], 1
        )
        self.assertEqual(normalized["selected_gap_id"], "G001")
        self.assertEqual(normalized["update_local_refs"], ["N1"])
        self.assertEqual(applied[0]["gap_id"], "G001")
        self.assertEqual(frontier[0]["status"], "OPEN")

    def test_existing_open_gap_can_be_selected(self) -> None:
        frontier, _ = apply_ledger_adds([], [add_update()], turn=1)
        normalized, merged, applied = pipeline.normalize_direct_ledger_result(
            ask_result(ref="G001", updates=[]), frontier, 2
        )
        self.assertEqual(normalized["selected_gap_id"], "G001")
        self.assertEqual(merged, frontier)
        self.assertEqual(applied, [])

    def test_asked_gap_is_rejected(self) -> None:
        frontier, _ = apply_ledger_adds([], [add_update()], turn=1)
        frontier = mark_gap_asked(frontier, "G001", turn=1)
        with self.assertRaisesRegex(ValueError, "not OPEN"):
            pipeline.normalize_direct_ledger_result(
                ask_result(ref="G001", updates=[]), frontier, 2
            )

    def test_unknown_gap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.normalize_direct_ledger_result(
                ask_result(ref="G999", updates=[]), [], 1
            )

    def test_none_is_allowed_only_with_empty_open_ledger(self) -> None:
        normalized, _, _ = pipeline.normalize_direct_ledger_result(
            ask_result(ref="NONE", updates=[]), [], 1
        )
        self.assertEqual(normalized["selected_gap_id"], "NONE")
        frontier, _ = apply_ledger_adds([], [add_update()], turn=1)
        with self.assertRaises(ValueError):
            pipeline.normalize_direct_ledger_result(
                ask_result(ref="NONE", updates=[]), frontier, 2
            )

    def test_duplicate_update_cannot_be_selected_by_local_ref(self) -> None:
        frontier, _ = apply_ledger_adds([], [add_update()], turn=1)
        with self.assertRaises(ValueError):
            pipeline.normalize_direct_ledger_result(
                ask_result(ref="N1"), frontier, 2
            )

    def test_ready_with_new_or_open_gap_is_valid(self) -> None:
        normalized, frontier, applied = pipeline.normalize_direct_ledger_result(
            ready_result([add_update()]), [], 1
        )
        self.assertEqual(normalized["action"], "READY_TO_MODEL")
        self.assertEqual(len(frontier), 1)
        self.assertEqual(len(applied), 1)

    def test_ready_rejects_public_question(self) -> None:
        raw = ready_result()
        raw["public_question"] = question()
        with self.assertRaises(ValueError):
            pipeline.normalize_direct_ledger_result(raw, [], 1)

    def test_ask_requires_valid_mcd_card(self) -> None:
        raw = ask_result()
        raw["public_question"]["allow_other"] = False
        with self.assertRaises(ValueError):
            pipeline.normalize_direct_ledger_result(raw, [], 1)

    def test_local_refs_must_be_unique(self) -> None:
        updates = [
            add_update("N1", "Gap one."),
            add_update("N1", "Gap two."),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate local_ref"):
            pipeline.normalize_direct_ledger_result(
                ask_result(ref="N1", updates=updates), [], 1
            )


class FrozenBoundaryTests(unittest.TestCase):
    def test_run_interaction_has_no_stage2_or_selector_call(self) -> None:
        source = inspect.getsource(pipeline.run_interaction)
        self.assertNotIn("agent_client.complete(", source)
        self.assertNotIn("selector_client.complete(", source)
        self.assertIn("gap_search_client.complete(", source)

    def test_bundle_contains_only_public_protocol_and_direct_stage(self) -> None:
        bundle = pipeline.load_prompt_bundle(
            ROOT / "prompts",
            "mc_d",
            ROOT / "prompts" / "judge_prompt.md",
        )
        self.assertIn("Persistent Gap Ledger Direct MC-D Stage", bundle.gap_search_prompt)
        self.assertFalse(hasattr(bundle, "agent_prompt"))
        self.assertFalse(hasattr(bundle, "selector_prompt"))

    def test_shared_prompt_hashes_match_ledger_baseline(self) -> None:
        baseline = ROOT.parent / "interopt" / "prompts"
        for name, expected in pipeline.FROZEN_PROMPT_SHA256.items():
            local = hashlib.sha256((ROOT / "prompts" / name).read_bytes()).hexdigest().upper()
            source = hashlib.sha256((baseline / name).read_bytes()).hexdigest().upper()
            self.assertEqual(local, expected)
            self.assertEqual(local, source)

    def test_direct_prompt_hash_is_frozen(self) -> None:
        actual = hashlib.sha256(
            (ROOT / "prompts" / pipeline.DIRECT_PROMPT_FILENAME).read_bytes()
        ).hexdigest().upper()
        self.assertEqual(actual, pipeline.DIRECT_PROMPT_SHA256)

    def test_external_case_directory_is_accepted(self) -> None:
        parser = pipeline.build_argument_parser()
        args = parser.parse_args(
            ["--toml_dirs", str(ROOT / "external_cases")]
        )
        with patch.object(parser, "error", side_effect=raise_value_error):
            pipeline.validate_ablation_contract(parser, args)

    def test_frontier_control_hash_drift_is_rejected(self) -> None:
        parser = pipeline.build_argument_parser()
        args = parser.parse_args([])
        with patch.object(
            pipeline, "FROZEN_FRONTIER_CONTROL_SHA256", "0" * 64
        ):
            with patch.object(parser, "error", side_effect=raise_value_error):
                with self.assertRaisesRegex(ValueError, "frontier_control"):
                    pipeline.validate_ablation_contract(parser, args)

    def test_existing_output_config_allows_only_k_growth_and_same_arm(self) -> None:
        parser = pipeline.build_argument_parser()
        args = parser.parse_args(["--k", "5"])
        args.pipeline_version = pipeline.PIPELINE_VERSION
        args.ablation_arm = pipeline.ABLATION_ARM
        args.gap_search_enabled = True
        args.stage2_enabled = False
        args.pipeline_mode = pipeline.determine_pipeline_mode(args)
        existing = vars(args).copy()
        existing["k"] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            (output_root / "run_config.json").write_text(
                json.dumps(existing, ensure_ascii=False),
                encoding="utf-8",
            )
            pipeline.validate_existing_output_config(parser, output_root, args)

            existing["toml_dirs"] = [str(ROOT / "wrong_dataset")]
            (output_root / "run_config.json").write_text(
                json.dumps(existing, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(parser, "error", side_effect=raise_value_error):
                with self.assertRaisesRegex(ValueError, "配置不兼容"):
                    pipeline.validate_existing_output_config(
                        parser, output_root, args
                    )

    def test_summary_proves_stage2_is_unused(self) -> None:
        summary = pipeline.summarize_deep_search_events(
            [
                {
                    "gap_search_attempts": [{"status": "valid"}],
                    "gap_search_result": {
                        "action": "ASK",
                        "updates": [add_update()],
                    },
                }
            ]
        )
        self.assertFalse(summary["stage2_enabled"])
        self.assertEqual(summary["stage2_agent_call_count"], 0)
        self.assertEqual(summary["candidate_questions_generated_count"], 0)
        self.assertEqual(summary["selector_call_count"], 0)
        self.assertEqual(summary["gap_search_gap_count"], 1)


class OfflineEndToEndTests(unittest.TestCase):
    def test_three_invalid_bindings_become_protocol_failure(self) -> None:
        invalid = ask_result(ref="G999", updates=[])
        queues = {
            "formulation_question_selector": [invalid, invalid, invalid],
            "detector": [],
            "user_simulator": [],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "premature_stop"},
                }
            ],
        }

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                del temperature
                self.profile_name = profile_name
                self.model = f"fake-{profile_name}"
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages, timeout=180, max_retries=10):
                del messages, timeout, max_retries
                payload = queues[self.profile_name].pop(0)
                return pipeline.ChatResult(
                    content=json.dumps(payload),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {"content": "Plan.", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = pipeline.PromptBundle(
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
            gap_search_prompt="direct-ledger",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(pipeline, "ChatClient", FakeChatClient):
                stat = pipeline.run_interaction(
                    case=case,
                    agent_profile="generic_agent",
                    detector_profile="detector",
                    selector_profile="formulation_question_selector",
                    user_profile="user_simulator",
                    judge_profile="judge",
                    run_index=1,
                    output_root=Path(temp_dir),
                    prompts=prompts,
                    interaction_mode="mc_d",
                    agent_temperature=0.2,
                    detector_temperature=0.0,
                    selector_temperature=0.0,
                    user_temperature=0.0,
                    judge_temperature=0.0,
                    max_turns=20,
                )
        self.assertTrue(stat["protocol_failed"])
        self.assertEqual(
            stat["protocol_failure_type"],
            "gap_search_format_invalid",
        )
        self.assertEqual(stat["gap_search_call_count"], 3)
        self.assertEqual(stat["binding_regen_count"], 3)
        self.assertEqual(stat["detector_call_count"], 0)
        self.assertEqual(stat["ledger_final_size"], 0)

    def test_two_asks_consume_ledger_and_keep_private_simulator_history(self) -> None:
        direct_one = ask_result(
            "N1",
            [add_update("N1", "Whether weekend work is allowed.")],
            "May work be scheduled on weekends?",
        )
        direct_two = ask_result(
            "N1",
            [add_update("N1", "Whether overtime is capped.")],
            "Is overtime capped?",
        )
        detector_ask_one = {
            "action": "ask",
            "question_count": 1,
            "is_valid_mc": True,
            "question": "May work be scheduled on weekends?",
            "option_count": 3,
            "option_ids": ["A", "B", "C"],
            "allow_other": True,
            "rationale": "valid",
        }
        detector_ask_two = {
            **detector_ask_one,
            "question": "Is overtime capped?",
        }
        detector_ready = {
            "action": "ready_to_model",
            "question_count": 0,
            "is_valid_mc": True,
            "question": "",
            "option_count": 0,
            "option_ids": [],
            "allow_other": False,
            "rationale": "ready",
        }
        queues = {
            "formulation_question_selector": [
                direct_one,
                direct_two,
                ready_result(),
            ],
            "detector": [detector_ask_one, detector_ask_two, detector_ready],
            "user_simulator": [
                {
                    "choice": "A",
                    "rationale": "Weekend work is allowed.",
                    "match_status": "exact_match",
                },
                {
                    "choice": "B",
                    "rationale": "Overtime has a fixed cap.",
                    "match_status": "exact_match",
                },
            ],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "appropriate_stop"},
                }
            ],
        }
        captured_calls: list[tuple[str, list[dict[str, str]]]] = []

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                del temperature
                self.profile_name = profile_name
                self.model = f"fake-{profile_name}"
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages, timeout=180, max_retries=10):
                del timeout, max_retries
                captured_calls.append(
                    (self.profile_name, json.loads(json.dumps(messages)))
                )
                payload = queues[self.profile_name].pop(0)
                return pipeline.ChatResult(
                    content=json.dumps(payload),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {
                "content": "Plan a tiny staffing model.",
                "visible_unit_ids": [],
            },
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = pipeline.PromptBundle(
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
            gap_search_prompt="direct-ledger",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(pipeline, "ChatClient", FakeChatClient):
                stat = pipeline.run_interaction(
                    case=case,
                    agent_profile="generic_agent",
                    detector_profile="detector",
                    selector_profile="formulation_question_selector",
                    user_profile="user_simulator",
                    judge_profile="judge",
                    run_index=1,
                    output_root=Path(temp_dir),
                    prompts=prompts,
                    interaction_mode="mc_d",
                    agent_temperature=0.2,
                    detector_temperature=0.0,
                    selector_temperature=0.0,
                    user_temperature=0.0,
                    judge_temperature=0.0,
                    max_turns=20,
                )
            run_dir = Path(stat["run_dir"])
            transcript = json.loads(
                (run_dir / "transcript.json").read_text(encoding="utf-8")
            )
            events = json.loads(
                (run_dir / pipeline.CLARIFICATION_STATE_AUDIT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(stat["completed_ready_to_model"])
        self.assertFalse(stat["protocol_failed"])
        self.assertEqual(stat["ledger_add_count"], 2)
        self.assertEqual(stat["ledger_asked_count"], 2)
        self.assertEqual(stat["ledger_final_open_count"], 0)
        self.assertEqual(stat["stage2_agent_call_count"], 0)
        self.assertEqual(stat["selector_call_count"], 0)
        self.assertEqual(stat["agent_usage"], {})
        self.assertEqual(stat["selector_usage"], {})
        self.assertEqual(events[0]["ledger_consumed"]["gap_id"], "G001")
        self.assertEqual(events[1]["ledger_consumed"]["gap_id"], "G002")

        public_blob = json.dumps(transcript, ensure_ascii=False)
        self.assertNotIn("G001", public_blob)
        self.assertNotIn("selected_gap_ref", public_blob)
        self.assertNotIn("evidence_quote", public_blob)

        simulator_calls = [
            messages
            for profile, messages in captured_calls
            if profile == "user_simulator"
        ]
        prior_private_replies = [
            json.loads(message["content"])
            for message in simulator_calls[1]
            if message["role"] == "assistant"
        ]
        self.assertEqual(
            prior_private_replies[0]["rationale"],
            "Weekend work is allowed.",
        )
        self.assertEqual(
            prior_private_replies[0]["match_status"],
            "exact_match",
        )

        non_stage1_calls = [
            messages
            for profile, messages in captured_calls
            if profile != "formulation_question_selector"
        ]
        non_stage1_blob = json.dumps(non_stage1_calls, ensure_ascii=False)
        self.assertNotIn("G001", non_stage1_blob)
        self.assertNotIn("selected_gap_ref", non_stage1_blob)

    def test_ready_with_open_gap_is_counted_not_blocked(self) -> None:
        direct_ready = ready_result(
            [add_update("N1", "Whether overtime is capped.")]
        )
        queues = {
            "formulation_question_selector": [direct_ready],
            "detector": [
                {
                    "action": "ready_to_model",
                    "question_count": 0,
                    "is_valid_mc": True,
                    "question": "",
                    "option_count": 0,
                    "option_ids": [],
                    "allow_other": False,
                    "rationale": "ready",
                }
            ],
            "user_simulator": [],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "premature_stop"},
                }
            ],
        }

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                del temperature
                self.profile_name = profile_name
                self.model = f"fake-{profile_name}"
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages, timeout=180, max_retries=10):
                del messages, timeout, max_retries
                payload = queues[self.profile_name].pop(0)
                return pipeline.ChatResult(
                    content=json.dumps(payload),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {"content": "Plan.", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = pipeline.PromptBundle(
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
            gap_search_prompt="direct-ledger",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(pipeline, "ChatClient", FakeChatClient):
                stat = pipeline.run_interaction(
                    case=case,
                    agent_profile="generic_agent",
                    detector_profile="detector",
                    selector_profile="formulation_question_selector",
                    user_profile="user_simulator",
                    judge_profile="judge",
                    run_index=1,
                    output_root=Path(temp_dir),
                    prompts=prompts,
                    interaction_mode="mc_d",
                    agent_temperature=0.2,
                    detector_temperature=0.0,
                    selector_temperature=0.0,
                    user_temperature=0.0,
                    judge_temperature=0.0,
                    max_turns=20,
                )
        self.assertTrue(stat["completed_ready_to_model"])
        self.assertEqual(stat["ready_with_open_gaps_count"], 1)
        self.assertEqual(stat["ledger_final_open_count"], 1)


if __name__ == "__main__":
    unittest.main()
