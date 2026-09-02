from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "readygate_parallel", ROOT / "run_parallel.py"
)
assert SPEC and SPEC.loader
PARALLEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PARALLEL
SPEC.loader.exec_module(PARALLEL)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mock_case(case_id: str) -> dict:
    return {"_case_id": case_id}


class ReadyGateParallelTests(unittest.TestCase):
    def test_task_list_keeps_same_identity_for_mc_d_only(self) -> None:
        args = PARALLEL.build_argument_parser().parse_args(["--interaction_modes", "mc_d", "--k", "2"])
        cases = [mock_case("orclarify_001"), mock_case("orclarify_002")]

        tasks = PARALLEL.build_tasks(cases, args)

        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            [(task.case_id, task.interaction_mode, task.run_index) for task in tasks],
            [
                ("orclarify_001", "mc_d", 1),
                ("orclarify_002", "mc_d", 1),
                ("orclarify_001", "mc_d", 2),
                ("orclarify_002", "mc_d", 2),
            ],
        )

    def test_completion_rule_requires_ready_gate_audit(self) -> None:
        args = PARALLEL.build_argument_parser().parse_args(["--interaction_modes", "mc_d", "--k", "1"])
        cases = [mock_case("orclarify_001"), mock_case("orclarify_002")]
        tasks = PARALLEL.build_tasks(cases, args)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            complete_dir = PARALLEL.run_dir_for_task(output_root, tasks[0])
            write_json(complete_dir / "statistics.json", {"case_id": tasks[0].case_id})
            write_json(complete_dir / "judge_result.json", {"slot_scores": []})
            write_json(complete_dir / PARALLEL.PIPELINE.READY_GATE_AUDIT_FILENAME, [])

            partial_dir = PARALLEL.run_dir_for_task(output_root, tasks[1])
            write_json(partial_dir / "statistics.json", {"case_id": tasks[1].case_id})
            write_json(partial_dir / "judge_result.json", {"slot_scores": []})

            completed, pending = PARALLEL.split_task_states(output_root, tasks)

        self.assertEqual([state.task for state in completed], [tasks[0]])
        self.assertIn(tasks[1], [state.task for state in pending])
        self.assertEqual(PARALLEL.count_by_mode(completed), {"mc_d": 1})
        self.assertEqual(PARALLEL.count_by_mode(pending), {"mc_d": 1})

    def test_run_pending_task_passes_ready_gate_args(self) -> None:
        args = PARALLEL.build_argument_parser().parse_args(
            ["--interaction_modes", "mc_d", "--k", "1", "--ready_gate_profile", "ready_gate_x"]
        )
        args.pipeline_mode = "readygate"
        args.ready_gate_temperature = 0.0
        task = PARALLEL.build_tasks([mock_case("orclarify_001")], args)[0]

        captured: dict[str, object] = {}

        def fake_run_interaction(**kwargs):
            captured.update(kwargs)
            return {"case_id": kwargs["case"]["_case_id"]}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PARALLEL.PIPELINE, "run_interaction", side_effect=fake_run_interaction
        ):
            state = PARALLEL.inspect_task_state(Path(temp_dir), task)
            PARALLEL.run_pending_task(
                state,
                {"orclarify_001": {"_case_id": "orclarify_001"}},
                {"mc_d": object()},
                args,
                Path(temp_dir),
            )

        self.assertEqual(captured["ready_gate_profile"], "ready_gate_x")
        self.assertEqual(captured["ready_gate_temperature"], 0.0)

    def test_concurrency_limit_rejects_values_above_official_limit(self) -> None:
        parser = PARALLEL.build_argument_parser()
        args = parser.parse_args(["--max_concurrency", "501"])

        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            PARALLEL.validate_parallel_args(parser, args)

    def test_execute_pending_tasks_records_failures_without_stopping_other_tasks(self) -> None:
        args = PARALLEL.build_argument_parser().parse_args(
            ["--interaction_modes", "mc_d", "--k", "1", "--max_concurrency", "2"]
        )
        args.pipeline_mode = "readygate"
        cases = [mock_case("orclarify_001"), mock_case("orclarify_002")]
        cases_by_id = {case["_case_id"]: case for case in cases}
        tasks = PARALLEL.build_tasks(cases, args)

        def fake_run_interaction(**kwargs):
            case_id = kwargs["case"]["_case_id"]
            if case_id == "orclarify_002":
                raise RuntimeError("simulated API failure")
            return {
                "case_id": case_id,
                "interaction_mode": kwargs["interaction_mode"],
                "agent_profile": kwargs["agent_profile"],
                "agent_model": "fake-model",
                "run_index": kwargs["run_index"],
                "weighted_slot_score": 1.0,
            }

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            PARALLEL.PIPELINE, "run_interaction", side_effect=fake_run_interaction
        ):
            output_root = Path(temp_dir)
            states = [PARALLEL.inspect_task_state(output_root, task) for task in tasks]
            with redirect_stdout(io.StringIO()):
                successes, failures = PARALLEL.execute_pending_tasks(
                    states,
                    cases_by_id,
                    {"mc_d": object()},
                    args,
                    output_root,
                )
            failure_log = json.loads(
                (output_root / PARALLEL.PARALLEL_FAILURES_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(len(successes), 1)
        self.assertEqual(successes[0]["case_id"], "orclarify_001")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["case_id"], "orclarify_002")
        self.assertEqual(failure_log[0]["error"], "simulated API failure")
        self.assertEqual(failure_log[0]["call_max_retries"], 10)


if __name__ == "__main__":
    unittest.main()
