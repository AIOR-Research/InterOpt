from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"


@dataclass
class BatchTask:
    method: str
    case_start: int
    case_end: int
    output_dir: str
    log_path: str
    expected_statistics: int
    attempts: int = 0
    status: str = "queued"
    return_code: int | None = None
    started_at: str = ""
    finished_at: str = ""


# [模块目标]：把 ID001-100 这类连续编号切成小批次，避免一次性启动过多 DeepSeek 调用。
# [输入输出]：输入起止 ID 与每块大小；输出每个分块的起止编号，例如 001-010。
def build_ranges(case_start: int, case_end: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current = case_start
    while current <= case_end:
        end = min(case_end, current + chunk_size - 1)
        ranges.append((current, end))
        current = end + 1
    return ranges


def case_ids_for_range(start: int, end: int) -> list[str]:
    return [f"{idx:03d}" for idx in range(start, end + 1)]


def count_statistics(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(1 for _ in output_dir.rglob("statistics.json"))


# [模块目标]：把当前批次控制器的进度写成 JSON，让主线程或监看子代理可以只读查看。
# [输入输出]：输入任务列表和运行状态；输出 controller_status.json。
def write_status(
    status_path: Path,
    tasks: list[BatchTask],
    running: dict[int, BatchTask],
    run_tag: str,
    max_parallel: int,
) -> None:
    rows = []
    for task in tasks:
        row = asdict(task)
        row["statistics_count"] = count_statistics(Path(task.output_dir))
        rows.append(row)
    summary: dict[str, object] = {
        "run_tag": run_tag,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "max_parallel": max_parallel,
        "total_tasks": len(tasks),
        "queued": sum(1 for task in tasks if task.status == "queued"),
        "running": len(running),
        "completed": sum(1 for task in tasks if task.status == "completed"),
        "failed": sum(1 for task in tasks if task.status == "failed"),
        "tasks": rows,
    }
    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_command(task: BatchTask, args: argparse.Namespace) -> list[str]:
    run_pipeline = EXPERIMENT_ROOT / task.method / "run_pipeline.py"
    return [
        sys.executable,
        str(run_pipeline),
        "--case_ids",
        *case_ids_for_range(task.case_start, task.case_end),
        "--k",
        str(args.k),
        "--max_turns",
        str(args.max_turns),
        "--output_dir",
        task.output_dir,
        "--agent_profile",
        args.agent_profile,
        "--user_profile",
        args.user_profile,
        "--judge_profile",
        args.judge_profile,
    ]


# [模块目标]：启动单个分块进程，并把输出写入独立日志，方便失败后定位。
# [输入输出]：输入一个 BatchTask；返回 Python 子进程对象。
def start_task(task: BatchTask, args: argparse.Namespace) -> subprocess.Popen:
    task.attempts += 1
    task.status = "running"
    task.return_code = None
    task.started_at = datetime.now().isoformat(timespec="seconds")
    task.finished_at = ""
    Path(task.output_dir).mkdir(parents=True, exist_ok=True)
    Path(task.log_path).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log_file = open(task.log_path, "ab")
    return subprocess.Popen(
        build_command(task, args),
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )


# [模块目标]：统一调度 ORPilot 与 Lets 的分块任务，限制并发并自动补跑失败分块。
# [输入输出]：读取命令行参数；输出各分块目录、日志和 controller_status.json。
def main() -> None:
    parser = argparse.ArgumentParser(description="Run external baseline batches with bounded concurrency.")
    parser.add_argument("--methods", nargs="+", default=["orpilot", "lets_tpp"])
    parser.add_argument("--case_start", type=int, default=1)
    parser.add_argument("--case_end", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--run_tag", default=f"k5_001_100_max20_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--runs_root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--agent_profile", default="generic_agent")
    parser.add_argument("--user_profile", default="user_simulator")
    parser.add_argument("--judge_profile", default="judge")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    controller_dir = runs_root / f"external_baselines_{args.run_tag}_controller"
    logs_dir = controller_dir / "logs"
    controller_dir.mkdir(parents=True, exist_ok=True)
    status_path = controller_dir / "controller_status.json"

    tasks: list[BatchTask] = []
    for method in args.methods:
        for start, end in build_ranges(args.case_start, args.case_end, args.chunk_size):
            output_dir = runs_root / f"{method}_{args.run_tag}_chunk{start:03d}_{end:03d}"
            log_path = logs_dir / f"{method}_chunk{start:03d}_{end:03d}.log"
            tasks.append(
                BatchTask(
                    method=method,
                    case_start=start,
                    case_end=end,
                    output_dir=str(output_dir),
                    log_path=str(log_path),
                    expected_statistics=(end - start + 1) * args.k,
                )
            )

    running: dict[int, BatchTask] = {}
    processes: dict[int, subprocess.Popen] = {}
    write_status(status_path, tasks, running, args.run_tag, args.max_parallel)

    while True:
        for pid, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            task = running.pop(pid)
            processes.pop(pid)
            task.return_code = return_code
            task.finished_at = datetime.now().isoformat(timespec="seconds")
            complete_count = count_statistics(Path(task.output_dir))
            if return_code == 0 and complete_count >= task.expected_statistics:
                task.status = "completed"
            elif task.attempts <= args.retries:
                task.status = "queued"
            else:
                task.status = "failed"

        while len(running) < args.max_parallel:
            next_task = next((task for task in tasks if task.status == "queued"), None)
            if next_task is None:
                break
            process = start_task(next_task, args)
            running[process.pid] = next_task
            processes[process.pid] = process

        write_status(status_path, tasks, running, args.run_tag, args.max_parallel)
        if not running and all(task.status in {"completed", "failed"} for task in tasks):
            break
        time.sleep(10)

    write_status(status_path, tasks, running, args.run_tag, args.max_parallel)
    failed = [task for task in tasks if task.status == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} batch chunks failed. See {status_path}")


if __name__ == "__main__":
    main()
