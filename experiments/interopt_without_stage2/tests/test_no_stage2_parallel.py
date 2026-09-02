from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_parallel as parallel  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_valid_run(run_dir: Path, task: object) -> None:
    write_json(
        run_dir / "statistics.json",
        {
            "pipeline_version": parallel.PIPELINE.PIPELINE_VERSION,
            "ablation_arm": parallel.PIPELINE.ABLATION_ARM,
            "case_id": task.case_id,
            "run_index": task.run_index,
            "interaction_mode": task.interaction_mode,
            "agent_profile": task.agent_profile,
        },
    )
    write_json(
        run_dir / "judge_result.json",
        {
            "slot_scores": [],
            "stopping_behavior": {"status": "appropriate_stop"},
        },
    )
    write_json(
        run_dir / parallel.PIPELINE.FORMULATION_DEEP_SEARCH_AUDIT_FILENAME,
        [{"turn": 1}],
    )


class ParallelContractTests(unittest.TestCase):
    def test_concurrency_cap_is_500(self) -> None:
        self.assertEqual(parallel.MAX_DEEPSEEK_V4_PRO_CONCURRENCY, 500)

    def test_completion_contract_uses_direct_ledger_audit(self) -> None:
        self.assertEqual(
            parallel.PIPELINE.CLARIFICATION_STATE_AUDIT_FILENAME,
            "ledger_direct_question_events.json",
        )

    def test_task_completion_requires_all_three_artifacts(self) -> None:
        task = parallel.PipelineTask(
            sequence=1,
            case_id="orclarify_001",
            interaction_mode="mc_d",
            agent_profile="generic_agent",
            run_index=1,
        )
        state = parallel.TaskState(
            task=task,
            run_dir="unused",
            has_statistics=True,
            has_judge_result=True,
            has_deep_search_audit=False,
        )
        self.assertFalse(state.is_complete)

    def test_valid_identity_is_complete(self) -> None:
        task = parallel.PipelineTask(
            sequence=1,
            case_id="orclarify_001",
            interaction_mode="mc_d",
            agent_profile="generic_agent",
            run_index=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            run_dir = parallel.run_dir_for_task(output_root, task)
            write_valid_run(run_dir, task)
            self.assertTrue(parallel.inspect_task_state(output_root, task).is_complete)

    def test_empty_or_wrong_identity_is_pending(self) -> None:
        task = parallel.PipelineTask(
            sequence=1,
            case_id="orclarify_001",
            interaction_mode="mc_d",
            agent_profile="generic_agent",
            run_index=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            run_dir = parallel.run_dir_for_task(output_root, task)
            run_dir.mkdir(parents=True)
            for filename in (
                "statistics.json",
                "judge_result.json",
                parallel.PIPELINE.FORMULATION_DEEP_SEARCH_AUDIT_FILENAME,
            ):
                (run_dir / filename).write_text("", encoding="utf-8")
            self.assertFalse(parallel.inspect_task_state(output_root, task).is_complete)

            write_valid_run(run_dir, task)
            stat = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
            stat["ablation_arm"] = "different_arm"
            write_json(run_dir / "statistics.json", stat)
            self.assertFalse(parallel.inspect_task_state(output_root, task).is_complete)


if __name__ == "__main__":
    unittest.main()
