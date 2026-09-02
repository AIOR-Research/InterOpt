from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "interopt_parallel", ROOT / "run_parallel.py"
)
PARALLEL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PARALLEL
SPEC.loader.exec_module(PARALLEL)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FormulationDeepSearchParallelTests(unittest.TestCase):
    def test_completion_rule_requires_deep_search_audit(self) -> None:
        task = PARALLEL.PipelineTask(
            sequence=1,
            case_id="orclarify_001",
            interaction_mode="mc_d",
            agent_profile="generic_agent",
            run_index=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            run_dir = PARALLEL.run_dir_for_task(output_root, task)
            write_json(run_dir / "statistics.json", {})
            write_json(run_dir / "judge_result.json", {})

            state = PARALLEL.inspect_task_state(output_root, task)
            self.assertFalse(state.is_complete)

            write_json(
                run_dir / PARALLEL.PIPELINE.FORMULATION_DEEP_SEARCH_AUDIT_FILENAME,
                [],
            )
            state = PARALLEL.inspect_task_state(output_root, task)
            self.assertTrue(state.is_complete)

    def test_run_pending_task_passes_selector_args(self) -> None:
        task = PARALLEL.PipelineTask(
            sequence=1,
            case_id="orclarify_001",
            interaction_mode="mc_d",
            agent_profile="generic_agent",
            run_index=1,
        )
        state = PARALLEL.TaskState(
            task=task,
            run_dir="unused",
            has_statistics=False,
            has_judge_result=False,
            has_deep_search_audit=False,
        )
        args = SimpleNamespace(
            detector_profile="detector",
            selector_profile="selector_x",
            user_profile="user_simulator",
            judge_profile="judge",
            agent_temperature=0.2,
            detector_temperature=0.0,
            selector_temperature=0.1,
            user_temperature=0.0,
            judge_temperature=0.0,
            max_turns=20,
            detector_feedback_mode="none",
            retry_limit_behavior="pass_through",
            monitor_user_answers=False,
            pipeline_mode="interopt",
        )
        prompts = {"mc_d": object()}
        captured: dict[str, object] = {}

        def fake_run_interaction(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"ok": True}

        with patch.object(PARALLEL.PIPELINE, "run_interaction", side_effect=fake_run_interaction):
            result = PARALLEL.run_pending_task(
                state,
                {"orclarify_001": {"_case_id": "orclarify_001"}},
                prompts,
                args,
                Path("out"),
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["selector_profile"], "selector_x")
        self.assertEqual(captured["selector_temperature"], 0.1)
        self.assertFalse(any(str(key).startswith("ready") for key in captured))


if __name__ == "__main__":
    unittest.main()
