from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluation_protocol_local", ROOT / "run_pipeline.py")
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class PipelineInterfaceTests(unittest.TestCase):
    def test_cli_defaults_match_passive_local_contract(self) -> None:
        args = PIPELINE.build_argument_parser().parse_args([])

        self.assertEqual(args.toml_dirs, [str(path) for path in PIPELINE.DEFAULT_TOML_DIRS])
        self.assertIsNone(args.output_dir)
        self.assertEqual(args.agent_profiles, ["generic_agent"])
        self.assertEqual(args.detector_profile, "detector")
        self.assertEqual(args.user_profile, "user_simulator")
        self.assertEqual(args.judge_profile, "judge")
        self.assertEqual(args.agent_temperature, 0.2)
        self.assertEqual(args.max_agent_retries, 0)
        self.assertEqual(args.detector_feedback_mode, "none")
        self.assertEqual(args.retry_limit_behavior, "pass_through")
        self.assertFalse(args.monitor_user_answers)
        self.assertEqual(PIPELINE.determine_pipeline_mode(args), "passive_question_audit")

    def test_default_case_directory_is_packaged_data(self) -> None:
        default_path = Path(PIPELINE.DEFAULT_TOML_DIRS[0])

        self.assertEqual(default_path, PIPELINE.REPO_ROOT / "data")

    def test_manual_directories_are_sorted_and_duplicate_case_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "case_002.toml").write_text(
                '[metadata]\ncase_id = "case_002"\n', encoding="utf-8"
            )
            (second / "case_001.toml").write_text(
                '[metadata]\ncase_id = "case_001"\n', encoding="utf-8"
            )

            cases = PIPELINE.load_cases([first, second], limit=None)
            self.assertEqual([case["_case_id"] for case in cases], ["case_001", "case_002"])

            (second / "duplicate.toml").write_text(
                '[metadata]\ncase_id = "case_002"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
                PIPELINE.load_cases([first, second], limit=None)

    def test_role_models_and_credentials_use_repo_deepseek_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GENERIC_AGENT_MODEL": "agent-model",
                "USER_SIMULATOR_MODEL": "user-model",
                "DETECTOR_MODEL": "detector-model",
                "JUDGE_MODEL": "judge-model",
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
                "DEEPSEEK_API_KEY": "test-key",
            },
            clear=False,
        ):
            self.assertEqual(PIPELINE.resolve_model_name("generic_agent"), "agent-model")
            self.assertEqual(PIPELINE.resolve_model_name("user_simulator"), "user-model")
            self.assertEqual(PIPELINE.resolve_model_name("detector"), "detector-model")
            self.assertEqual(PIPELINE.resolve_model_name("judge"), "judge-model")
            client = PIPELINE.ChatClient("generic_agent", 0.2)
            self.assertEqual(client._api_settings(), ("https://deepseek.example/v1", "test-key"))

    def test_removed_interfaces_do_not_remain_in_source(self) -> None:
        source = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")

        self.assertNotIn("OPENROUTER", source)
        self.assertNotIn("max_user_retries", source)
        self.assertNotIn("build_user_scope_feedback", source)
        self.assertNotIn("build_minimal_user_retry_feedback", source)

    def test_default_structural_invalid_response_passes_through_without_retry(self) -> None:
        responses = {
            "generic_agent": ["STRUCTURALLY_INVALID_RESPONSE"],
            "detector": [
                '{"action":"invalid","question_count":0,"is_atomic":false,'
                '"atomic_questions":[],"rationale":"no question"}'
            ],
            "user_simulator": ["Business answer"],
            "judge": [
                '{"slot_scores":[],"stopping_behavior":{},"silent_assumptions":[]}'
            ],
        }
        clients = {}

        class FakeClient:
            def __init__(self, profile_name: str, temperature: float):
                self.profile_name = profile_name
                self.model = profile_name + "-model"
                self.temperature = temperature
                self.total_usage = {}
                self.total_estimated_cost_usd = 0.0
                self.calls = 0
                clients[profile_name] = self

            def complete(self, messages, timeout=180, max_retries=5):
                del messages, timeout, max_retries
                content = responses[self.profile_name][self.calls]
                self.calls += 1
                return SimpleNamespace(content=content)

        case = {
            "_case_id": "case_001",
            "initial_brief": {"content": "Initial request", "visible_unit_ids": []},
            "problem_units": [],
            "hidden_slots": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PIPELINE, "ChatClient", FakeClient
        ), patch.object(PIPELINE.time, "sleep", return_value=None):
            PIPELINE.MAX_AGENT_RETRIES_PER_TURN = 0
            stat = PIPELINE.run_interaction(
                case=case,
                agent_profile="generic_agent",
                detector_profile="detector",
                user_profile="user_simulator",
                judge_profile="judge",
                run_index=1,
                output_root=Path(temp_dir),
                prompts=("agent", "question detector", "answer detector", "simulator", "judge"),
                agent_temperature=0.2,
                detector_temperature=0.0,
                user_temperature=0.0,
                judge_temperature=0.0,
                max_turns=1,
                monitor_user_answers=False,
                detector_feedback_mode="none",
                retry_limit_behavior="pass_through",
            )

        self.assertEqual(clients["generic_agent"].calls, 1)
        self.assertEqual(stat["agent_retry_total"], 0)
        self.assertEqual(stat["agent_retry_limit_pass_through_count"], 1)
        self.assertFalse(stat["protocol_failed"])


if __name__ == "__main__":
    unittest.main()
