"""
Parallel scheduler for the Open InterOPT w/o Stage 1 ablation pipeline.

This file only parallelizes independent case x run tasks. It reuses
run_pipeline.py for prompts, API calls, interaction logic, output files, and
summary generation.
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


PARALLEL_FAILURES_FILENAME = "parallel_failures.json"
PARALLEL_MANIFEST_FILENAME = "parallel_manifest.json"
PARALLEL_CONFIG_FILENAME = "run_config_parallel.json"


@dataclass(frozen=True)
class PipelineTask:
    sequence: int
    case_id: str
    agent_profile: str
    run_index: int


@dataclass
class TaskState:
    task: PipelineTask
    run_dir: Path
    has_statistics: bool
    has_judge_result: bool

    @property
    def is_complete(self) -> bool:
        return self.has_statistics and self.has_judge_result


def build_argument_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_prompts_dir = script_dir / "prompts"
    parser = argparse.ArgumentParser(
        description=(
            "Parallel scheduler for Open InterOPT without Stage 1. "
            "It calls run_pipeline.run_interaction without changing experiment logic."
        )
    )
    parser.add_argument("--toml_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--recursive_toml",
        action="store_true",
        help="Read *.toml recursively under toml_dir. Default matches run_pipeline.py and reads only the top-level directory.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument(
        "--case_timeout_sec",
        type=float,
        default=None,
        help="Soft timeout per case in seconds. The current API request is allowed to return, then the case is stopped.",
    )
    parser.add_argument("--agent_profiles", nargs="+", default=["deepseek_v4_pro"])
    parser.add_argument("--detector_profile", default="deepseek_v4_pro")
    parser.add_argument("--user_profile", default="deepseek_v4_pro")
    parser.add_argument("--judge_profile", default="deepseek_v4_pro")
    parser.add_argument("--agent_temperature", type=float, default=0.2)
    parser.add_argument("--detector_temperature", type=float, default=0.0)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--prompts_dir", default=str(default_prompts_dir))
    parser.add_argument("--max_agent_retries", type=int, default=3)
    parser.add_argument("--max_user_retries", type=int, default=3)
    parser.add_argument("--monitor_user_answers", action="store_true")
    parser.add_argument(
        "--detector_feedback_mode",
        choices=["visible", "minimal", "none"],
        default="visible",
    )
    parser.add_argument(
        "--retry_limit_behavior",
        choices=["pass_through", "protocol_failed"],
        default="pass_through",
    )
    parser.add_argument(
        "--fixed_both_sides_silent",
        action="store_true",
        help="Shortcut for monitor_user_answers + no detector feedback + protocol_failed retry behavior.",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=5,
        help="Maximum concurrent run_interaction tasks. Use 5 or 10 for smoke tests first.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--progress_interval_sec",
        type=float,
        default=20.0,
        help="Minimum seconds between heartbeat progress lines if no task completes.",
    )
    return parser


def normalize_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_concurrency < 1:
        parser.error("--max_concurrency must be at least 1")
    if args.progress_interval_sec <= 0:
        parser.error("--progress_interval_sec must be greater than 0")
    if args.fixed_both_sides_silent:
        args.monitor_user_answers = True
        args.detector_feedback_mode = "none"
        args.retry_limit_behavior = "protocol_failed"
        args.pipeline_mode = "fixed_both_sides_silent"
    elif args.monitor_user_answers:
        args.pipeline_mode = "fixed_both_sides"
    else:
        args.pipeline_mode = "fixed_question_only"


def run_dir_for_task(output_root: Path, task: PipelineTask) -> Path:
    return output_root / task.agent_profile / f"run_{task.run_index:02d}" / task.case_id


def inspect_task_state(output_root: Path, task: PipelineTask) -> TaskState:
    run_dir = run_dir_for_task(output_root, task)
    return TaskState(
        task=task,
        run_dir=run_dir,
        has_statistics=(run_dir / "statistics.json").exists(),
        has_judge_result=(run_dir / "judge_result.json").exists(),
    )


def build_tasks(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[PipelineTask]:
    tasks: list[PipelineTask] = []
    sequence = 0
    for profile in args.agent_profiles:
        for case in cases:
            for run_index in range(1, args.k + 1):
                sequence += 1
                tasks.append(
                    PipelineTask(
                        sequence=sequence,
                        case_id=str(case["_case_id"]),
                        agent_profile=profile,
                        run_index=run_index,
                    )
                )
    return tasks


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    toml_dir = Path(args.toml_dir)
    if not args.recursive_toml:
        return PIPELINE.load_cases(toml_dir, args.limit)
    paths = sorted(toml_dir.rglob("*.toml"))
    if args.limit is not None:
        paths = paths[: args.limit]
    return [PIPELINE.load_case(path) for path in paths]


def split_task_states(
    output_root: Path,
    tasks: list[PipelineTask],
) -> tuple[list[TaskState], list[TaskState]]:
    states = [inspect_task_state(output_root, task) for task in tasks]
    completed = [state for state in states if state.is_complete]
    pending = [state for state in states if not state.is_complete]
    return completed, pending


def load_existing_stat(state: TaskState) -> dict[str, Any]:
    return json.loads((state.run_dir / "statistics.json").read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def task_to_failure_record(state: TaskState, exc: BaseException) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_id": state.task.case_id,
        "agent_profile": state.task.agent_profile,
        "run_index": state.task.run_index,
        "run_dir": str(state.run_dir),
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def progress_line(done: int, total: int, start_time: float) -> str:
    elapsed = time.time() - start_time
    rate = done / elapsed * 60.0 if elapsed > 0 and done else 0.0
    remaining = total - done
    eta = remaining / (done / elapsed) if elapsed > 0 and done else None
    eta_text = format_duration(eta) if eta is not None else "--:--:--"
    return (
        f"[progress] {done}/{total} "
        f"({done / total * 100:.1f}%) | "
        f"{rate:.2f} runs/min | "
        f"elapsed {format_duration(elapsed)} | ETA {eta_text}"
    )


def run_one_task(
    state: TaskState,
    *,
    cases_by_id: dict[str, dict[str, Any]],
    output_root: Path,
    prompts: tuple[str, str, str, str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    case = cases_by_id[state.task.case_id]
    return PIPELINE.run_interaction(
        case=case,
        agent_profile=state.task.agent_profile,
        detector_profile=args.detector_profile,
        user_profile=args.user_profile,
        judge_profile=args.judge_profile,
        run_index=state.task.run_index,
        output_root=output_root,
        prompts=prompts,
        agent_temperature=args.agent_temperature,
        detector_temperature=args.detector_temperature,
        user_temperature=args.user_temperature,
        judge_temperature=args.judge_temperature,
        max_turns=args.max_turns,
        monitor_user_answers=args.monitor_user_answers,
        detector_feedback_mode=args.detector_feedback_mode,
        retry_limit_behavior=args.retry_limit_behavior,
        pipeline_mode=args.pipeline_mode,
        case_timeout_sec=args.case_timeout_sec,
    )


def write_manifest(
    output_root: Path,
    args: argparse.Namespace,
    *,
    total_tasks: int,
    completed_before: int,
    completed_after: int,
    pending_before: int,
    failure_count: int,
    dry_run: bool,
) -> None:
    write_json(
        output_root / PARALLEL_MANIFEST_FILENAME,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "runner": "run_pipeline_parallel.py",
            "dry_run": dry_run,
            "max_concurrency": args.max_concurrency,
            "total_tasks": total_tasks,
            "completed_before": completed_before,
            "completed_after": completed_after,
            "pending_before": pending_before,
            "pending_after": total_tasks - completed_after,
            "failure_count": failure_count,
            "completion_rule": "statistics.json + judge_result.json",
            "pipeline_version": PIPELINE.PIPELINE_VERSION,
            "ablation_arm": PIPELINE.ABLATION_ARM,
            "gap_search_enabled": False,
            "pipeline_mode": args.pipeline_mode,
            "agent_profiles": args.agent_profiles,
            "k": args.k,
        },
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    normalize_args(parser, args)
    args.pipeline_version = PIPELINE.PIPELINE_VERSION
    args.ablation_arm = PIPELINE.ABLATION_ARM
    args.gap_search_enabled = False

    PIPELINE.MAX_AGENT_RETRIES_PER_TURN = args.max_agent_retries
    PIPELINE.MAX_USER_RETRIES_PER_TURN = args.max_user_retries

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / PARALLEL_CONFIG_FILENAME, vars(args))

    cases = load_cases(args)
    cases_by_id = {str(case["_case_id"]): case for case in cases}
    prompts = PIPELINE.load_prompt_files(Path(args.prompts_dir))
    tasks = build_tasks(cases, args)
    completed_before, pending = split_task_states(output_root, tasks)

    print(f"Total tasks: {len(tasks)}", flush=True)
    print(f"Already complete: {len(completed_before)}", flush=True)
    print(f"Pending: {len(pending)}", flush=True)
    print(f"Max concurrency: {args.max_concurrency}", flush=True)

    if args.dry_run:
        write_manifest(
            output_root,
            args,
            total_tasks=len(tasks),
            completed_before=len(completed_before),
            completed_after=len(completed_before),
            pending_before=len(pending),
            failure_count=0,
            dry_run=True,
        )
        return

    stats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for state in completed_before:
        stats.append(load_existing_stat(state))

    start_time = time.time()
    done_count = len(completed_before)
    last_heartbeat = start_time
    total_tasks = len(tasks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        future_to_state = {
            executor.submit(
                run_one_task,
                state,
                cases_by_id=cases_by_id,
                output_root=output_root,
                prompts=prompts,
                args=args,
            ): state
            for state in pending
        }
        while future_to_state:
            done, _ = concurrent.futures.wait(
                future_to_state,
                timeout=args.progress_interval_sec,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                now = time.time()
                if now - last_heartbeat >= args.progress_interval_sec:
                    print(progress_line(done_count, total_tasks, start_time), flush=True)
                    last_heartbeat = now
                continue
            for future in done:
                state = future_to_state.pop(future)
                try:
                    stat = future.result()
                    stats.append(stat)
                    status = stat.get("rule_based_stopping_status") or stat.get("stopping_status")
                    score = stat.get("weighted_slot_score")
                    score_text = f"{float(score):.3f}" if score is not None else "NA"
                    done_count += 1
                    print(
                        f"[done {done_count}/{total_tasks}] "
                        f"{state.task.case_id} run_{state.task.run_index:02d} "
                        f"score={score_text} core={stat.get('core_exact_restore')} "
                        f"ready={stat.get('completed_ready_to_model')} stop={status} "
                        f"conf={stat.get('formulatable_confidence')} "
                        f"cost=${float(stat.get('estimated_cost_usd', 0) or 0):.4f}",
                        flush=True,
                    )
                except BaseException as exc:
                    done_count += 1
                    record = task_to_failure_record(state, exc)
                    failures.append(record)
                    print(
                        f"[failed {done_count}/{total_tasks}] "
                        f"{state.task.case_id} run_{state.task.run_index:02d}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

    write_json(output_root / PARALLEL_FAILURES_FILENAME, failures)
    summary = PIPELINE.summarize(stats, output_root)
    PIPELINE.write_markdown_report(summary, output_root, args)
    write_manifest(
        output_root,
        args,
        total_tasks=len(tasks),
        completed_before=len(completed_before),
        completed_after=len(stats),
        pending_before=len(pending),
        failure_count=len(failures),
        dry_run=False,
    )
    print(progress_line(len(stats), total_tasks, start_time), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
