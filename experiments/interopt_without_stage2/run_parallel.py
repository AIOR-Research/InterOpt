"""
InterOPT w/o Stage 2 parallel runner.

This file is only a scheduler for `run_pipeline.py`: it keeps the same case
loader, prompts, model settings, run_interaction implementation, output shape,
summary writer, and clarification-state audit contract. The only intended difference is
that pending `case x mode x run_index` tasks can run concurrently.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import run_pipeline as PIPELINE  # noqa: E402


MAX_DEEPSEEK_V4_PRO_CONCURRENCY = 500
DEFAULT_PARALLEL_CONCURRENCY = 200
PARALLEL_FAILURES_FILENAME = "parallel_failures.json"
PARALLEL_MANIFEST_FILENAME = "parallel_manifest.json"
PARALLEL_CONFIG_FILENAME = "run_config_parallel.json"


@dataclass(frozen=True)
class PipelineTask:
    sequence: int
    case_id: str
    interaction_mode: str
    agent_profile: str
    run_index: int


@dataclass
class TaskState:
    task: PipelineTask
    run_dir: str
    has_statistics: bool
    has_judge_result: bool
    has_deep_search_audit: bool
    artifacts_valid: bool = False

    @property
    def is_complete(self) -> bool:
        return (
            self.has_statistics
            and self.has_judge_result
            and self.has_deep_search_audit
            and self.artifacts_valid
        )


# [模块目标]：为 InterOPT w/o Stage 2 提供并发、断点续跑和完成清点能力。
# [输入输出]：输入命令行参数；输出 argparse 解析器，沿用主脚本的全部实验参数。
# [LLM 交互]：本函数不调用 LLM；它只让用户设置并发数、dry run 和进度汇报节奏。
def build_argument_parser() -> argparse.ArgumentParser:
    parser = PIPELINE.build_argument_parser()
    parser.description = (
        "InterOPT w/o Stage 2 pipeline parallel scheduler. "
        "It calls run_pipeline.run_interaction without changing experiment logic."
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=DEFAULT_PARALLEL_CONCURRENCY,
        help=(
            "Maximum concurrent run_interaction tasks. "
            "The official deepseek-v4-pro account-level concurrency limit is 500; "
            "the default 200 leaves quota headroom."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only count completed and pending tasks; do not call any LLM.",
    )
    parser.add_argument(
        "--progress_interval_sec",
        type=float,
        default=30.0,
        help="Seconds between milestone progress reports while tasks are running.",
    )
    return parser


def validate_parallel_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_concurrency < 1:
        parser.error("--max_concurrency must be at least 1")
    if args.max_concurrency > MAX_DEEPSEEK_V4_PRO_CONCURRENCY:
        parser.error(
            "--max_concurrency must not exceed 500 for deepseek-v4-pro "
            "(DeepSeek official account-level concurrency limit)."
        )
    if args.progress_interval_sec <= 0:
        parser.error("--progress_interval_sec must be greater than 0")


# [模块目标]：枚举正式实验要跑的全部任务身份。
# [输入输出]：输入 case 列表和命令行参数；输出稳定的 `case x mode x run_index` 任务清单。
# [LLM 交互]：不调用 LLM；这里只决定“哪些 run 应该存在”，不决定每个 run 的实验内容。
def build_tasks(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[PipelineTask]:
    tasks: list[PipelineTask] = []
    sequence = 0
    for run_index in range(1, args.k + 1):
        for case in cases:
            case_id = str(case["_case_id"])
            for interaction_mode in args.interaction_modes:
                for agent_profile in args.agent_profiles:
                    sequence += 1
                    tasks.append(
                        PipelineTask(
                            sequence=sequence,
                            case_id=case_id,
                            interaction_mode=interaction_mode,
                            agent_profile=agent_profile,
                            run_index=run_index,
                        )
                    )
    return tasks


def run_dir_for_task(output_root: Path, task: PipelineTask) -> Path:
    return (
        output_root
        / task.interaction_mode
        / task.agent_profile
        / f"run_{task.run_index:02d}"
        / task.case_id
    )


def inspect_task_state(output_root: Path, task: PipelineTask) -> TaskState:
    run_dir = run_dir_for_task(output_root, task)
    completed_artifacts = PIPELINE.load_completed_run_artifacts(
        run_dir,
        case_id=task.case_id,
        run_index=task.run_index,
        interaction_mode=task.interaction_mode,
        agent_profile=task.agent_profile,
    )
    return TaskState(
        task=task,
        run_dir=str(run_dir),
        has_statistics=(run_dir / "statistics.json").exists(),
        has_judge_result=(run_dir / "judge_result.json").exists(),
        has_deep_search_audit=(run_dir / PIPELINE.FORMULATION_DEEP_SEARCH_AUDIT_FILENAME).exists(),
        artifacts_valid=completed_artifacts is not None,
    )


def split_task_states(
    output_root: Path, tasks: list[PipelineTask]
) -> tuple[list[TaskState], list[TaskState]]:
    states = [inspect_task_state(output_root, task) for task in tasks]
    completed = [state for state in states if state.is_complete]
    pending = [state for state in states if not state.is_complete]
    return completed, pending


def count_by_mode(states: list[TaskState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        mode = state.task.interaction_mode
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def load_existing_stat(run_dir: Path, task: PipelineTask) -> dict[str, Any]:
    completed_artifacts = PIPELINE.load_completed_run_artifacts(
        run_dir,
        case_id=task.case_id,
        run_index=task.run_index,
        interaction_mode=task.interaction_mode,
        agent_profile=task.agent_profile,
    )
    if completed_artifacts is None:
        raise ValueError(f"Run artifacts failed completion validation: {run_dir}")
    return completed_artifacts[0]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def task_to_failure_record(
    state: TaskState, exc: BaseException, *, call_max_retries: int = 10
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_id": state.task.case_id,
        "interaction_mode": state.task.interaction_mode,
        "agent_profile": state.task.agent_profile,
        "run_index": state.task.run_index,
        "run_dir": state.run_dir,
        "exception_type": type(exc).__name__,
        "error": str(exc),
        "call_max_retries": call_max_retries,
    }


def write_failure_log(output_root: Path, failures: list[dict[str, Any]]) -> None:
    write_json(output_root / PARALLEL_FAILURES_FILENAME, failures)


def write_parallel_manifest(
    output_root: Path,
    args: argparse.Namespace,
    *,
    total_tasks: int,
    completed_before: int,
    completed_after: int,
    pending_before: int,
    pending_after: int,
    failures: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runner": "run_parallel.py",
        "dry_run": dry_run,
        "max_concurrency": args.max_concurrency,
        "total_tasks": total_tasks,
        "completed_before": completed_before,
        "completed_after": completed_after,
        "pending_before": pending_before,
        "pending_after": pending_after,
        "failure_count": len(failures),
        "failures_file": PARALLEL_FAILURES_FILENAME,
        "completion_rule": f"statistics.json + judge_result.json + {PIPELINE.CLARIFICATION_STATE_AUDIT_FILENAME}",
        "clarification_state_audit_note": (
            f"{PIPELINE.CLARIFICATION_STATE_AUDIT_FILENAME} is part of the clarification-state contract; "
            "resume requires it so incomplete runs are not treated as complete."
        ),
    }
    write_json(output_root / PARALLEL_MANIFEST_FILENAME, payload)


def write_parallel_config(output_root: Path, args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config["parallel_runner"] = "run_parallel.py"
    config["completion_rule"] = f"statistics.json + judge_result.json + {PIPELINE.CLARIFICATION_STATE_AUDIT_FILENAME}"
    write_json(output_root / PARALLEL_CONFIG_FILENAME, config)
    run_config = output_root / "run_config.json"
    if not run_config.exists():
        write_json(run_config, config)


def print_plan_snapshot(
    *, cases: list[dict[str, Any]], tasks: list[PipelineTask], completed: list[TaskState], pending: list[TaskState]
) -> None:
    PIPELINE.print_milestone(
        "数据加载完成",
        f"{len(cases)} 个 case，任务 {len(tasks)} 个",
    )
    PIPELINE.print_milestone(
        "断点清点完成",
        (
            f"已完成 {len(completed)}，待补 {len(pending)}；"
            f"已完成分布 {count_by_mode(completed)}，待补分布 {count_by_mode(pending)}"
        ),
    )


def run_pending_task(
    state: TaskState,
    cases_by_id: dict[str, dict[str, Any]],
    prompts_by_mode: dict[str, PIPELINE.PromptBundle],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    task = state.task
    return PIPELINE.run_interaction(
        case=cases_by_id[task.case_id],
        agent_profile=task.agent_profile,
        detector_profile=args.detector_profile,
        selector_profile=args.selector_profile,
        user_profile=args.user_profile,
        judge_profile=args.judge_profile,
        run_index=task.run_index,
        output_root=output_root,
        prompts=prompts_by_mode[task.interaction_mode],
        interaction_mode=task.interaction_mode,
        agent_temperature=args.agent_temperature,
        detector_temperature=args.detector_temperature,
        selector_temperature=args.selector_temperature,
        user_temperature=args.user_temperature,
        judge_temperature=args.judge_temperature,
        max_turns=args.max_turns,
        detector_feedback_mode=args.detector_feedback_mode,
        retry_limit_behavior=args.retry_limit_behavior,
        monitor_user_answers=args.monitor_user_answers,
        pipeline_mode=args.pipeline_mode,
    )


# [模块目标]：并发执行所有未完成 run，并把成功/失败都落到可恢复的本地文件。
# [输入输出]：输入 pending 任务；输出成功 stat 列表和失败清单。
# [LLM 交互]：每个线程仍调用主脚本的 run_interaction；本层不改任何 Prompt 或角色消息。
def execute_pending_tasks(
    pending: list[TaskState],
    cases_by_id: dict[str, dict[str, Any]],
    prompts_by_mode: dict[str, PIPELINE.PromptBundle],
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pending:
        return [], []

    max_workers = min(args.max_concurrency, len(pending))
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    future_to_state: dict[concurrent.futures.Future[dict[str, Any]], TaskState] = {}
    started_at = time.monotonic()
    next_report_at = started_at + args.progress_interval_sec

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for state in pending:
            future = executor.submit(
                run_pending_task,
                state,
                cases_by_id,
                prompts_by_mode,
                args,
                output_root,
            )
            future_to_state[future] = state

        while future_to_state:
            done, _ = concurrent.futures.wait(
                future_to_state,
                timeout=max(0.1, min(args.progress_interval_sec, 1.0)),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            now = time.monotonic()
            if not done and now >= next_report_at:
                finished = len(successes) + len(failures)
                PIPELINE.print_milestone(
                    "LLM 并发处理中",
                    f"{finished}/{len(pending)} 已结束，成功 {len(successes)}，失败 {len(failures)}",
                )
                next_report_at = now + args.progress_interval_sec
                continue

            for future in done:
                state = future_to_state.pop(future)
                try:
                    successes.append(future.result())
                except Exception as exc:
                    failures.append(task_to_failure_record(state, exc))
                    write_failure_log(output_root, failures)

            finished = len(successes) + len(failures)
            if finished == len(pending) or now >= next_report_at:
                PIPELINE.print_milestone(
                    "LLM 并发处理中",
                    f"{finished}/{len(pending)} 已结束，成功 {len(successes)}，失败 {len(failures)}",
                )
                next_report_at = now + args.progress_interval_sec

    write_failure_log(output_root, failures)
    return successes, failures


def load_all_completed_stats(output_root: Path, tasks: list[PipelineTask]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for task in tasks:
        state = inspect_task_state(output_root, task)
        if state.is_complete:
            stats.append(load_existing_stat(Path(state.run_dir), task))
    return stats


def prepare_cases(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    cases, missing = PIPELINE.resolve_case_selection(
        PIPELINE.read_cases([Path(path) for path in args.toml_dirs]),
        args.case_ids,
    )
    if missing:
        raise SystemExit(
            f"--case_ids 指定的题号在 --toml_dirs 内未找到: {missing} "
            f"（支持 70、070、orclarify_070；toml_dirs={args.toml_dirs}）"
        )
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        parser.error("No cases selected.")
    return cases


def main() -> None:
    PIPELINE.load_env_file(PIPELINE.REPO_ROOT / ".env")
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_parallel_args(parser, args)
    args.pipeline_version = "interopt_without_stage2_parallel"
    args.ablation_arm = PIPELINE.ABLATION_ARM
    args.gap_search_enabled = True
    args.stage2_enabled = False
    if args.max_agent_retries < 0 or args.max_user_retries < 0:
        parser.error("retry counts must be zero or greater")
    PIPELINE.validate_ablation_contract(parser, args)
    args.pipeline_mode = PIPELINE.determine_pipeline_mode(args)

    PIPELINE.MAX_AGENT_RETRIES_PER_TURN = args.max_agent_retries
    PIPELINE.MAX_USER_RETRIES_PER_TURN = args.max_user_retries

    cases = prepare_cases(args, parser)
    output_root = Path(args.output_dir) if args.output_dir else PIPELINE.make_default_output_dir()
    args.output_dir = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    PIPELINE.validate_existing_output_config(parser, output_root, args)
    write_parallel_config(output_root, args)

    tasks = build_tasks(cases, args)
    completed_before, pending_before = split_task_states(output_root, tasks)

    print("| 里程碑 | 状态 |", flush=True)
    print("|---|---|", flush=True)
    PIPELINE.print_milestone("输出目录已确认", str(output_root))
    print_plan_snapshot(
        cases=cases,
        tasks=tasks,
        completed=completed_before,
        pending=pending_before,
    )

    if args.dry_run:
        write_parallel_manifest(
            output_root,
            args,
            total_tasks=len(tasks),
            completed_before=len(completed_before),
            completed_after=len(completed_before),
            pending_before=len(pending_before),
            pending_after=len(pending_before),
            failures=[],
            dry_run=True,
        )
        PIPELINE.print_milestone("Dry run 完成", "未调用 LLM；只写入并发清点 manifest")
        return

    cases_by_id = {str(case["_case_id"]): case for case in cases}
    prompts_by_mode = {
        interaction_mode: PIPELINE.load_prompt_bundle(
            Path(args.prompts_dir),
            interaction_mode,
            Path(args.judge_prompt_path),
            [Path(path) for path in args.agent_prompt_supplements],
            [Path(path) for path in args.selector_prompt_supplements],
            None,
            None,
        )
        for interaction_mode in args.interaction_modes
    }

    _, failures = execute_pending_tasks(
        pending_before,
        cases_by_id,
        prompts_by_mode,
        args,
        output_root,
    )

    completed_after, pending_after = split_task_states(output_root, tasks)
    stats = load_all_completed_stats(output_root, tasks)
    summary = PIPELINE.summarize(stats, output_root)
    PIPELINE.write_markdown_report(summary, output_root, args)
    write_parallel_manifest(
        output_root,
        args,
        total_tasks=len(tasks),
        completed_before=len(completed_before),
        completed_after=len(completed_after),
        pending_before=len(pending_before),
        pending_after=len(pending_after),
        failures=failures,
        dry_run=False,
    )

    PIPELINE.print_milestone(
        "评估汇总完成",
        (
            f"{len(completed_after)}/{len(tasks)} 完成；报告："
            f"{output_root / PIPELINE.REPORT_FILENAME}"
        ),
    )
    PIPELINE.print_milestone("估算总成本", f"${summary['total_estimated_cost_usd']:.4f}")
    if failures:
        PIPELINE.print_milestone(
            "失败留痕",
            f"{len(failures)} 个 run 失败，见 {output_root / PARALLEL_FAILURES_FILENAME}",
        )
    review_path, review_error = PIPELINE.refresh_choice_review_safely()
    if review_path:
        PIPELINE.print_milestone("可视化审核页更新", str(review_path))
    else:
        PIPELINE.print_milestone("可视化审核页更新跳过", review_error)


if __name__ == "__main__":
    main()
