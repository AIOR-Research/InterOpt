from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "interopt_pipeline", ROOT / "run_pipeline.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


def clarification_state(label: str) -> dict[str, object]:
    return {
        "confirmed_business_goal": f"{label} confirmed business goal",
        "confirmed_decision_scope": [f"{label} confirmed decision scope"],
        "confirmed_constraints_and_rules": [f"{label} confirmed constraint or rule"],
        "known_inputs_entities_and_indices": [f"{label} known input or entity"],
        "unresolved_business_assumptions": [f"{label} unresolved business assumption"],
    }


def question(label: str) -> dict[str, object]:
    return {
        "question": f"{label}?",
        "options": [
            {"id": "A", "text": f"{label} option A"},
            {"id": "B", "text": f"{label} option B"},
            {"id": "C", "text": f"{label} option C"},
        ],
        "allow_other": True,
    }


class FormulationDeepSearchContractTests(unittest.TestCase):
    def test_argument_defaults_match_experiment_3_contract(self) -> None:
        parser = PIPELINE.build_argument_parser()
        args = parser.parse_args(["--interaction_modes", "mc_d", "--k", "1"])

        self.assertEqual(args.selector_profile, "formulation_question_selector")
        self.assertEqual(args.selector_temperature, 0.0)
        self.assertEqual(args.max_turns, 20)
        self.assertEqual(
            PIPELINE.determine_pipeline_mode(args),
            "interopt",
        )
        self.assertTrue(
            PIPELINE.make_default_output_dir().name.startswith(
                "interopt_pipeline_"
            )
        )

    def test_normalize_deep_search_ask_payload(self) -> None:
        payload = {
            "action": "ASK",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "Q1": question("Q1"),
            "Q2": question("Q2"),
            "Q3": question("Q3"),
            "deep_search_decision_evidence": "Q2 clarifies the highest-risk business assumption.",
        }

        normalized = PIPELINE.normalize_deep_search_agent_payload(payload)

        self.assertEqual(normalized["action"], "ASK")
        self.assertEqual(normalized["Q2"]["question"], "Q2?")
        self.assertTrue(normalized["Q2"]["allow_other"])
        self.assertNotIn("question", normalized)
        self.assertNotIn("options", normalized)

    def test_normalize_deep_search_ready_payload(self) -> None:
        payload = {
            "action": "READY_TO_MODEL",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "deep_search_decision_evidence": "No remaining business-structure assumption needs clarification.",
            "summary": "Ready to build the model.",
        }

        normalized = PIPELINE.normalize_deep_search_agent_payload(payload)

        self.assertEqual(normalized["action"], "READY_TO_MODEL")
        self.assertEqual(normalized["summary"], "Ready to build the model.")
        self.assertNotIn("Q1", normalized)

    def test_deep_search_raw_payload_rejects_public_question_fields(self) -> None:
        payload = {
            "action": "ASK",
            "question": "This should not be top-level.",
            "options": [{"id": "A", "text": "A"}],
            "allow_other": True,
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "Q1": question("Q1"),
            "Q2": question("Q2"),
            "Q3": question("Q3"),
            "deep_search_decision_evidence": "Evidence.",
        }

        with self.assertRaises(ValueError):
            PIPELINE.normalize_deep_search_agent_payload(payload)

    def test_selector_result_contract_and_event_summary(self) -> None:
        selected = PIPELINE.normalize_selector_result(
            {
                "evaluation_process": "Q3 best clarifies the most risky unresolved business assumption.",
                "selected_question_id": "q3",
            }
        )
        self.assertEqual(selected["selected_question_id"], "Q3")

        metrics = PIPELINE.summarize_deep_search_events(
            [
                {
                    "agent_attempts": [{"status": "error"}, {"status": "ok"}],
                    "selector_attempts": [{"status": "error"}, {"status": "ok"}],
                    "candidate_critic": {"status": "valid", "revised": True},
                    "gap_search": {"status": "valid", "result": {"gaps": [{}, {}]}},
                    "fallback_used": False,
                    "selected_question_id": "Q3",
                },
                {
                    "agent_attempts": [{"status": "ok"}],
                    "selector_attempts": [{"status": "error"}, {"status": "error"}, {"status": "error"}],
                    "fallback_used": True,
                    "selected_question_id": "Q1",
                },
            ]
        )

        self.assertEqual(metrics["deep_search_agent_retry_count"], 1)
        self.assertEqual(metrics["deep_search_agent_format_error_count"], 1)
        self.assertEqual(metrics["selector_call_count"], 5)
        self.assertEqual(metrics["selector_error_count"], 4)
        self.assertEqual(metrics["selector_fallback_count"], 1)
        self.assertEqual(metrics["selector_selected_question_counts"], {"Q1": 1, "Q2": 0, "Q3": 1})
        self.assertEqual(metrics["candidate_critic_call_count"], 1)
        self.assertEqual(metrics["candidate_critic_revision_count"], 1)
        self.assertEqual(metrics["gap_search_call_count"], 1)
        self.assertEqual(metrics["gap_search_gap_count"], 2)

    def test_candidate_critic_and_gap_search_contracts(self) -> None:
        critic = PIPELINE.normalize_candidate_critic_result(
            {
                "evaluation_process": "Replace one low-impact parameter question.",
                "Q1": question("Q1"),
                "Q2": question("Q2"),
                "Q3": question("Q3"),
            }
        )
        self.assertEqual(critic["Q2"]["question"], "Q2?")

        gap_search = PIPELINE.normalize_ledger_search_result(
            {
                "decision": "CONTINUE",
                "search_summary": "A business rule remains unclear.",
                "updates": [
                    {
                        "operation": "ADD",
                        "category": "constraint_set",
                        "description": "Whether all locations must be visited.",
                        "evidence_quote": "not stated in the brief",
                    }
                ],
            },
            set(),
        )
        self.assertEqual(gap_search["updates"][0]["category"], "constraint_set")
        self.assertEqual(gap_search["duplicate_skipped"], 0)

    def test_new_mechanism_metrics_stay_in_statistics(self) -> None:
        stat = {
            "case_id": "orclarify_070",
            "candidate_critic_call_count": 2,
            "candidate_critic_error_count": 0,
            "candidate_critic_revision_count": 1,
            "gap_search_call_count": 3,
            "gap_search_error_count": 0,
            "gap_search_gap_count": 7,
            "raw_agent_latency_seconds": 1.2,
        }

        core, health = PIPELINE.split_stat(stat)

        self.assertEqual(core["candidate_critic_call_count"], 2)
        self.assertEqual(core["gap_search_gap_count"], 7)
        self.assertNotIn("raw_agent_latency_seconds", core)
        self.assertEqual(health["raw_agent_latency_seconds"], 1.2)

    def test_loads_optional_programmatic_stage_prompts(self) -> None:
        bundle = PIPELINE.load_prompt_bundle(
            ROOT / "prompts",
            "mc_d",
            ROOT / "prompts" / "judge_prompt.md",
            candidate_critic_prompt_path=None,
            gap_search_prompt_path=ROOT / "prompts" / "prompt-ledger-gap-search.md",
        )
        self.assertIsNone(bundle.candidate_critic_prompt)
        self.assertIn("Persistent Gap Ledger Search", bundle.gap_search_prompt or "")

    def test_user_simulator_keeps_full_reply_while_agent_sees_minimal_answer(self) -> None:
        simulator_messages = [{"role": "system", "content": "simulator system prompt"}]
        agent_messages = [{"role": "system", "content": "agent system prompt"}]
        simulator_reply = json.dumps(
            {
                "choice": "B",
                "rationale": "B matches the private business fact.",
                "match_status": "exact_match",
            },
            ensure_ascii=False,
        )
        natural_reply = "Business user selected option B."

        PIPELINE.append_choice_response_to_contexts(
            simulator_messages,
            agent_messages,
            simulator_reply,
            natural_reply,
        )

        self.assertEqual(simulator_messages[-1]["content"], simulator_reply)
        self.assertIn("rationale", simulator_messages[-1]["content"])
        self.assertIn("match_status", simulator_messages[-1]["content"])
        self.assertEqual(
            agent_messages[-1]["content"],
            "Business user response:\n\nBusiness user selected option B.",
        )
        self.assertNotIn("rationale", agent_messages[-1]["content"])
        self.assertNotIn("match_status", agent_messages[-1]["content"])

    def test_run_interaction_writes_deep_search_audit_without_real_llm(self) -> None:
        ask_payload = {
            "action": "ASK",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "Q1": question("Q1"),
            "Q2": question("Q2"),
            "Q3": question("Q3"),
            "deep_search_decision_evidence": "Q2 has the highest business-structure information value.",
        }
        ready_payload = {
            "action": "READY_TO_MODEL",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "deep_search_decision_evidence": "The user answer resolves the active business assumption.",
            "summary": "Ready to model with the selected policy.",
        }
        detector_ask = {
            "action": "ask",
            "question_count": 1,
            "is_valid_mc": True,
            "question": "Q2?",
            "option_count": 3,
            "option_ids": ["A", "B", "C"],
            "allow_other": True,
            "rationale": "valid",
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
            "generic_agent": [ask_payload, ready_payload],
            "formulation_question_selector": [
                {
                    "evaluation_process": "Q2 best clarifies the business-structure ambiguity.",
                    "selected_question_id": "Q2",
                }
            ],
            "detector": [detector_ask, detector_ready],
            "user_simulator": [
                {
                    "choice": "B",
                    "rationale": "B matches the business preference.",
                    "match_status": "exact_match",
                }
            ],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "appropriate_stop"},
                }
            ],
        }

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                self.profile_name = profile_name
                self.temperature = temperature
                self.model = f"fake-{profile_name}"
                self.total_usage: dict[str, int] = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages: list[dict[str, str]], timeout: int = 180, max_retries: int = 6):
                del messages, timeout, max_retries
                payload = queues[self.profile_name].pop(0)
                return PIPELINE.ChatResult(
                    content=json.dumps(payload, ensure_ascii=False),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {"content": "Plan a tiny shipment model.", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = PIPELINE.PromptBundle(
            agent_prompt="agent",
            selector_prompt="selector",
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(PIPELINE, "ChatClient", FakeChatClient):
                stat = PIPELINE.run_interaction(
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
                    pipeline_mode="interopt",
                )

            run_dir = Path(stat["run_dir"])
            events = json.loads(
                (run_dir / PIPELINE.FORMULATION_DEEP_SEARCH_AUDIT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))

        self.assertTrue(stat["completed_ready_to_model"])
        self.assertEqual(stat["selector_call_count"], 1)
        self.assertEqual(stat["selector_selected_question_counts"], {"Q1": 0, "Q2": 1, "Q3": 0})
        self.assertEqual(events[0]["selected_question_id"], "Q2")
        self.assertEqual(events[0]["public_agent_payload"]["question"], "Q2?")
        self.assertNotIn("C1", transcript[0]["structured"])

    def test_prompt_supplements_are_appended_without_changing_frozen_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            supplement = Path(temp_dir) / "prompt-test-supplement.md"
            supplement.write_text("One isolated experimental rule.", encoding="utf-8")
            bundle = PIPELINE.load_prompt_bundle(
                ROOT / "prompts",
                "mc_d",
                ROOT / "prompts" / "judge_prompt.md",
                [supplement],
                [supplement],
            )

        self.assertIn("# Experiment Supplement: prompt-test-supplement.md", bundle.agent_prompt)
        self.assertIn("One isolated experimental rule.", bundle.selector_prompt)

    def test_ledger_add_bind_consume_then_ready_cycle(self) -> None:
        ask_payload = {
            "action": "ASK",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "Q1": {**question("Q1"), "frontier_gap_id": "G001"},
            "Q2": {**question("Q2"), "frontier_gap_id": "G001"},
            "Q3": {**question("Q3"), "frontier_gap_id": "G001"},
            "deep_search_decision_evidence": "Q2 clarifies the weekend policy gap.",
        }
        ready_payload = {
            "action": "READY_TO_MODEL",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "deep_search_decision_evidence": "The user answer resolves the weekend policy.",
            "summary": "Ready to model with the selected policy.",
        }
        detector_ask = {
            "action": "ask",
            "question_count": 1,
            "is_valid_mc": True,
            "question": "Q2?",
            "option_count": 3,
            "option_ids": ["A", "B", "C"],
            "allow_other": True,
            "rationale": "valid",
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
            "generic_agent": [ask_payload, ready_payload],
            "formulation_question_selector": [
                {
                    "decision": "CONTINUE",
                    "search_summary": "Weekend work policy is unstated.",
                    "updates": [
                        {
                            "operation": "ADD",
                            "category": "constraint_set",
                            "description": "Whether weekends count as working days.",
                            "evidence_quote": "not stated in the brief",
                        }
                    ],
                },
                {
                    "evaluation_process": "Q2 best clarifies the weekend policy.",
                    "selected_question_id": "Q2",
                },
                {
                    "decision": "NO_NEW_GAP",
                    "search_summary": "No new gap beyond the ledger.",
                    "updates": [],
                },
            ],
            "detector": [detector_ask, detector_ready],
            "user_simulator": [
                {
                    "choice": "B",
                    "rationale": "B matches the business preference.",
                    "match_status": "exact_match",
                }
            ],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "appropriate_stop"},
                }
            ],
        }

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                self.profile_name = profile_name
                self.model = f"fake-{profile_name}"
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages, timeout=180, max_retries=6):
                del messages, timeout, max_retries
                payload = queues[self.profile_name].pop(0)
                return PIPELINE.ChatResult(
                    content=json.dumps(payload),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {"content": "Plan a tiny model.", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = PIPELINE.PromptBundle(
            agent_prompt="agent",
            selector_prompt="selector",
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
            gap_search_prompt="ledger",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(PIPELINE, "ChatClient", FakeChatClient):
                stat = PIPELINE.run_interaction(
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
            events = json.loads(
                (run_dir / "ledger_events.json").read_text(encoding="utf-8")
            )
            transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))

        self.assertTrue(stat["completed_ready_to_model"])
        self.assertEqual(stat["ledger_add_count"], 1)
        self.assertEqual(stat["ledger_asked_count"], 1)
        self.assertEqual(stat["ledger_final_open_count"], 0)
        self.assertEqual(stat["ready_with_open_gaps_count"], 0)
        self.assertEqual(stat["turn_count"], 2)
        self.assertEqual(events[0]["ledger_consumed"]["gap_id"], "G001")
        self.assertEqual(events[0]["ledger_consumed"]["asked_turn"], 1)
        # leakage guard: ledger internals never enter the public transcript
        transcript_blob = json.dumps(transcript, ensure_ascii=False)
        self.assertNotIn("G001", transcript_blob)
        self.assertNotIn("frontier_gap_id", transcript_blob)
        self.assertNotIn("ledger", transcript_blob.lower())

    def test_ready_with_open_gaps_is_counted_not_blocked(self) -> None:
        ready_payload = {
            "action": "READY_TO_MODEL",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "deep_search_decision_evidence": "Stopping despite the open gap.",
            "summary": "Ready to model.",
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
            "generic_agent": [ready_payload],
            "formulation_question_selector": [
                {
                    "decision": "CONTINUE",
                    "search_summary": "One unresolved policy gap.",
                    "updates": [
                        {
                            "operation": "ADD",
                            "category": "hard_soft_policy",
                            "description": "Whether the lateness penalty is hard or soft.",
                            "evidence_quote": "not stated in the brief",
                        }
                    ],
                }
            ],
            "detector": [detector_ready],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "premature_stop"},
                }
            ],
        }

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                self.profile_name = profile_name
                self.model = f"fake-{profile_name}"
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages, timeout=180, max_retries=6):
                del messages, timeout, max_retries
                payload = queues[self.profile_name].pop(0)
                return PIPELINE.ChatResult(
                    content=json.dumps(payload),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {"content": "Plan a tiny model.", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = PIPELINE.PromptBundle(
            agent_prompt="agent",
            selector_prompt="selector",
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
            gap_search_prompt="ledger",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(PIPELINE, "ChatClient", FakeChatClient):
                stat = PIPELINE.run_interaction(
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

        # zero stop intervention: READY is accepted even with an OPEN gap, but counted
        self.assertTrue(stat["completed_ready_to_model"])
        self.assertEqual(stat["ledger_add_count"], 1)
        self.assertEqual(stat["ledger_asked_count"], 0)
        self.assertEqual(stat["ledger_final_open_count"], 1)
        self.assertEqual(stat["ready_with_open_gaps_count"], 1)
        self.assertEqual(stat["agent_question_turn_count"], 0)


if __name__ == "__main__":
    unittest.main()


class LedgerConsumeErrorContractTests(unittest.TestCase):
    def test_hallucinated_gap_id_does_not_crash_run(self) -> None:
        ask_payload = {
            "action": "ASK",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "Q1": {**question("Q1"), "frontier_gap_id": "G999"},
            "Q2": {**question("Q2"), "frontier_gap_id": "G999"},
            "Q3": {**question("Q3"), "frontier_gap_id": "G999"},
            "deep_search_decision_evidence": "Agent hallucinated a gap id with an empty ledger.",
        }
        ready_payload = {
            "action": "READY_TO_MODEL",
            "C1": clarification_state("C1"),
            "C2": clarification_state("C2"),
            "C3": clarification_state("C3"),
            "deep_search_decision_evidence": "Done.",
            "summary": "Ready to model.",
        }
        detector_ask = {
            "action": "ask",
            "question_count": 1,
            "is_valid_mc": True,
            "question": "Q1?",
            "option_count": 3,
            "option_ids": ["A", "B", "C"],
            "allow_other": True,
            "rationale": "valid",
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
            "generic_agent": [ask_payload, ready_payload],
            "formulation_question_selector": [
                {
                    "decision": "NO_NEW_GAP",
                    "search_summary": "Ledger empty, nothing new.",
                    "updates": [],
                },
                {
                    "evaluation_process": "Q1.",
                    "selected_question_id": "Q1",
                },
                {
                    "decision": "NO_NEW_GAP",
                    "search_summary": "Still nothing new.",
                    "updates": [],
                },
            ],
            "detector": [detector_ask, detector_ready],
            "user_simulator": [
                {
                    "choice": "A",
                    "rationale": "A.",
                    "match_status": "exact_match",
                }
            ],
            "judge": [
                {
                    "slot_scores": [],
                    "stopping_behavior": {"status": "appropriate_stop"},
                }
            ],
        }

        class FakeChatClient:
            def __init__(self, profile_name: str, temperature: float):
                self.profile_name = profile_name
                self.model = f"fake-{profile_name}"
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0

            def complete(self, messages, timeout=180, max_retries=6):
                del messages, timeout, max_retries
                payload = queues[self.profile_name].pop(0)
                return PIPELINE.ChatResult(
                    content=json.dumps(payload),
                    usage={},
                    estimated_cost_usd=0.0,
                )

        case = {
            "_case_id": "orclarify_001",
            "initial_brief": {"content": "Plan a tiny model.", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        prompts = PIPELINE.PromptBundle(
            agent_prompt="agent",
            selector_prompt="selector",
            primary_detector_prompt="detector",
            answer_detector_prompt="answer-detector",
            simulator_prompt="simulator",
            judge_prompt="judge",
            gap_search_prompt="ledger",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(PIPELINE, "ChatClient", FakeChatClient):
                stat = PIPELINE.run_interaction(
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
            events = json.loads(
                (run_dir / "ledger_events.json").read_text(encoding="utf-8")
            )

        self.assertTrue(stat["completed_ready_to_model"])
        self.assertFalse(stat["protocol_failed"])
        self.assertEqual(stat["ledger_consume_error_count"], 1)
        self.assertEqual(stat["ledger_asked_count"], 0)
        self.assertEqual(events[0]["ledger_consume_error"]["gap_id"], "G999")


if __name__ == "__main__":
    unittest.main()
