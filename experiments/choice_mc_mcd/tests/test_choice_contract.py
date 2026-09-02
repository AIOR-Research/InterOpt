from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("choice_mc_mcd", ROOT / "run_pipeline.py")
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


VALID_MC_ASK = json.dumps(
    {
        "action": "ASK",
        "question": "Must the route return to the starting city?",
        "options": [
            {"id": "A", "text": "Yes, return to the start city."},
            {"id": "B", "text": "No, end at the last city."},
            {"id": "C", "text": "Either is acceptable."},
        ],
        "allow_other": False,
    }
)
VALID_CHOICE_DETECTOR = (
    '{"action":"ask","question_count":1,"is_valid_mc":true,'
    '"question":"Must the route return to the starting city?",'
    '"option_count":3,"option_ids":["A","B","C"],"allow_other":false,'
    '"rationale":"one valid mc question"}'
)
READY_DETECTOR = (
    '{"action":"ready_to_model","question_count":0,"is_valid_mc":true,'
    '"question":"","option_count":0,"option_ids":[],"allow_other":false,'
    '"rationale":"stop"}'
)
USER_NO_MATCH = json.dumps(
    {
        "choice": "A",
        "rationale": "The true policy is not represented; A is only the forced closest choice.",
        "match_status": "no_match",
    }
)
JUDGE_RESULT = (
    '{"slot_scores":[{"slot_id":"H1","name":"return_policy","severity":"P0",'
    '"hit":"yes","evidence_location":"turn 1","evidence_quote":"return",'
    '"rationale":"recovered"}],"silent_assumptions":[],'
    '"stopping_behavior":{"status":"appropriate_stop","unresolved_p0_slots":[],'
    '"unresolved_p1_slots":[],"rationale":"complete"},"summary":"complete"}'
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

    def complete(self, messages: list[dict[str, str]], timeout: int = 180, max_retries: int = 5):
        self.seen_messages.append([dict(message) for message in messages])
        return PIPELINE.ChatResult(content=self.queue.pop(0), usage={}, estimated_cost_usd=0.0)


def mock_case() -> dict:
    return {
        "_case_id": "mock_case",
        "_path": "mock.toml",
        "initial_brief": {"content": "Plan routes.", "visible_unit_ids": ["U1"]},
        "problem_units": [{"id": "U1", "kind": "request", "content": "Plan routes."}],
        "hidden_slots": [
            {
                "slot_id": "H1",
                "name": "return_policy",
                "severity": "P0",
                "severity_reason": "Changes the model.",
                "problem_unit_id": "U1",
                "semantic_hit_rule": "Ask whether the route returns.",
                "reference_acceptable_questions": ["Return to start?"],
                "failure_modes": ["Assume an open route."],
                "simulator_answer": "The route must return to the start.",
            }
        ],
        "simulator": {"business_role": "route manager"},
    }


class ChoiceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeChatClient.created = []
        PIPELINE.MAX_AGENT_RETRIES_PER_TURN = 0
        PIPELINE.MAX_USER_RETRIES_PER_TURN = 0

    def test_script_is_standalone_and_documents_protocol_at_top(self) -> None:
        source = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")

        self.assertTrue(source.startswith('"""\nChoice MC and MC-D pipeline'))
        self.assertNotIn("from run_pipeline import", source)
        self.assertNotIn("import run_pipeline", source)
        self.assertIn("不引入 `best_available_option`", source[:2000])

    def test_defaults_to_benchmark_cases_and_method_output_prefix(self) -> None:
        args = PIPELINE.build_argument_parser().parse_args([])

        self.assertEqual(len(args.toml_dirs), 1)
        self.assertEqual(Path(args.toml_dirs[0]), PIPELINE.REPO_ROOT / "data")
        self.assertEqual(args.max_turns, 20)
        self.assertEqual(PIPELINE.determine_pipeline_mode(args), "choice_passive_question_audit")
        self.assertTrue(PIPELINE.make_default_output_dir().name.startswith("choice_mc_mcd_"))

    def test_generic_agent_endpoint_override_is_role_scoped(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GENERIC_AGENT_MODEL": "glm",
                "GENERIC_AGENT_BASE_URL": "https://models.example.edu/api/v1",
                "GENERIC_AGENT_API_KEY": "test-agent-key",
                "DEEPSEEK_BASE_URL": "https://aux.example.com",
                "DEEPSEEK_API_KEY": "test-aux-key",
            },
            clear=True,
        ):
            self.assertEqual(PIPELINE.resolve_model_name("generic_agent"), "glm")
            self.assertEqual(PIPELINE.resolve_model_name("user_simulator"), "deepseek-v4-pro")
            self.assertEqual(PIPELINE.resolve_model_name("detector"), "deepseek-v4-pro")
            self.assertEqual(PIPELINE.resolve_model_name("judge"), "deepseek-v4-pro")

            agent_base, agent_key, agent_is_dedicated = PIPELINE._resolve_api_settings("generic_agent")
            aux_base, aux_key, aux_is_dedicated = PIPELINE._resolve_api_settings("judge")

        self.assertEqual(agent_base, "https://models.example.edu/api/v1")
        self.assertEqual(agent_key, "test-agent-key")
        self.assertTrue(agent_is_dedicated)
        self.assertEqual(aux_base, "https://aux.example.com")
        self.assertEqual(aux_key, "test-aux-key")
        self.assertFalse(aux_is_dedicated)

    def test_normalizer_requires_user_rationale_and_match_status(self) -> None:
        result = PIPELINE.normalize_choice_simulator_result(
            {
                "choice": "B",
                "rationale": "B exactly matches our return policy.",
                "match_status": "exact_match",
            },
            "mc",
        )

        self.assertEqual(result["choice"], "B")
        self.assertEqual(result["match_status"], "exact_match")
        with self.assertRaisesRegex(ValueError, "rationale"):
            PIPELINE.normalize_choice_simulator_result(
                {"choice": "B", "match_status": "exact_match"}, "mc"
            )
        with self.assertRaisesRegex(ValueError, "best_available_option"):
            PIPELINE.normalize_choice_simulator_result(
                {
                    "choice": "B",
                    "rationale": "reason",
                    "match_status": "exact_match",
                    "best_available_option": "B",
                },
                "mc",
            )

    def test_uses_dedicated_user_simulator_prompts(self) -> None:
        mc = PIPELINE.load_prompt_bundle(ROOT / "prompts", "mc")
        mc_d = PIPELINE.load_prompt_bundle(ROOT / "prompts", "mc_d")

        self.assertIn("mc)", mc.simulator_prompt)
        self.assertIn('"rationale"', mc.simulator_prompt)
        self.assertIn('"match_status"', mc.simulator_prompt)
        self.assertNotIn('"best_available_option"', mc.simulator_prompt)
        self.assertIn("mc_d)", mc_d.simulator_prompt)
        self.assertIn("audit records", mc_d.simulator_prompt)

    def test_agent_visible_choice_strips_audit_fields(self) -> None:
        visible = PIPELINE.build_agent_visible_choice(
            {
                "choice": "A",
                "rationale": "private reason",
                "match_status": "no_match",
            }
        )

        self.assertEqual(visible, {"choice": "A"})

    def test_mc_run_writes_audit_but_hides_it_from_agent_transcript_and_judge(self) -> None:
        FakeChatClient.responses = [
            [VALID_MC_ASK, '{"action":"READY_TO_MODEL","summary":"Confirmed."}'],
            [VALID_CHOICE_DETECTOR, READY_DETECTOR],
            [USER_NO_MATCH],
            [JUDGE_RESULT],
        ]
        prompts = PIPELINE.load_prompt_bundle(ROOT / "prompts", "mc")
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PIPELINE, "ChatClient", FakeChatClient
        ):
            stat = PIPELINE.run_interaction(
                case=mock_case(),
                agent_profile="generic_agent",
                detector_profile="detector",
                user_profile="user_simulator",
                judge_profile="judge",
                run_index=1,
                output_root=Path(temp_dir),
                prompts=prompts,
                interaction_mode="mc",
                agent_temperature=0.2,
                detector_temperature=0.0,
                user_temperature=0.0,
                judge_temperature=0.0,
                max_turns=3,
            )
            run_dir = Path(stat["run_dir"])
            transcript_text = (run_dir / "transcript.json").read_text(encoding="utf-8")
            judge_message = (run_dir / "judge_prompt_user_message.md").read_text(encoding="utf-8")
            audit = json.loads(
                (run_dir / PIPELINE.USER_CHOICE_AUDIT_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(stat["pipeline_version"], "choice")
        self.assertEqual(stat["choice_audit_metrics"]["mc_forced_choice_count"], 1)
        self.assertEqual(audit[0]["match_status"], "no_match")
        self.assertIn("forced closest choice", audit[0]["rationale"])
        self.assertFalse(audit[0]["audit_fields_entered_agent_context"])
        self.assertNotIn("rationale", transcript_text)
        self.assertNotIn("match_status", transcript_text)
        self.assertNotIn("forced closest choice", judge_message)
        agent_seen = json.dumps(FakeChatClient.created[0].seen_messages, ensure_ascii=False)
        self.assertNotIn("forced closest choice", agent_seen)
        self.assertNotIn("match_status", agent_seen)

    def test_mc_d_d_keeps_comment_visible_but_hides_rationale(self) -> None:
        normalized = PIPELINE.normalize_choice_simulator_result(
            {
                "choice": "D",
                "comment": "The route must return to the start.",
                "rationale": "None of A/B/C states the required closed route.",
                "match_status": "no_match",
            },
            "mc_d",
        )
        visible = PIPELINE.build_agent_visible_choice(normalized)

        self.assertEqual(
            visible,
            {"choice": "D", "comment": "The route must return to the start."},
        )
        self.assertNotIn("rationale", visible)
        self.assertNotIn("match_status", visible)

    def test_summary_reports_no_match_and_mc_forced_choice_rates(self) -> None:
        stat = {
            "interaction_mode": "mc",
            "agent_profile": "generic_agent",
            "agent_model": "test-model",
            "weighted_slot_score": 1.0,
            "choice_metrics": {"mc_question_turns": 2, "d_selection_count": 0},
            "choice_audit_metrics": {
                "choice_audit_event_count": 2,
                "choice_match_status_counts": {
                    "exact_match": 1,
                    "acceptable_match": 0,
                    "no_match": 1,
                    "undetermined": 0,
                },
                "mc_forced_choice_count": 1,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = PIPELINE.summarize([stat], output_dir)
            metrics = summary["profiles"]["mc::generic_agent"]["metrics"]
            PIPELINE.write_markdown_report(
                summary,
                output_dir,
                PIPELINE.build_argument_parser().parse_args([]),
            )
            report = (output_dir / "choice_eval_report.md").read_text(encoding="utf-8")

        self.assertEqual(metrics["ChoiceAuditEventCount"], 2)
        self.assertEqual(metrics["ChoiceExactMatchRate"], 0.5)
        self.assertEqual(metrics["ChoiceNoMatchRate"], 0.5)
        self.assertEqual(metrics["MCForcedChoiceCount"], 1)
        self.assertEqual(metrics["MCForcedChoiceRate"], 0.5)
        self.assertIn("no-match rate", report)
        self.assertIn("MC forced rate", report)
        self.assertIn("| 0.500 | 0.500 |", report)

    def test_summary_excludes_p2_only_runs_from_core_exact_denominator(self) -> None:
        common = {
            "interaction_mode": "mc_d",
            "agent_profile": "generic_agent",
            "agent_model": "test-model",
            "weighted_slot_score": 1.0,
        }
        stats = [
            {
                **common,
                "case_id": "core_case",
                "core_slot_count": 1,
                "core_exact_restore": True,
            },
            {
                **common,
                "case_id": "p2_only_case",
                "core_slot_count": 0,
                "core_exact_restore": False,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = PIPELINE.summarize(stats, Path(temp_dir))
            metrics = summary["profiles"]["mc_d::generic_agent"]["metrics"]

        self.assertEqual(metrics["CoreEligibleRunCount"], 1)
        self.assertEqual(metrics["CoreExactRestoreCount"], 1)
        self.assertEqual(metrics["CoreExactRestoreRate"], 1.0)


if __name__ == "__main__":
    unittest.main()
