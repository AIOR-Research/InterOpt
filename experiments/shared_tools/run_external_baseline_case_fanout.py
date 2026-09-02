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
class CaseTask:
    method: str
    case_id: int
    output_dir: str
    log_path: str
    expected_statistics: int
    attempts: int = 0
    status: str = "queued"
    return_code: int | None = None
    started_at: str = ""
    finished_at: str = ""


# [模块目标]：把 001 这种数字编号转换为统一 case_id，避免调度器和 run_pipeline 的命名不一致。
# [输入输出]：输入整数 case id；输出三位字符串，例如 7 -> "007"。
def format_case_id(case_id: int) -> str:
    return f"{case_id:03d}"


# [模块目标]：保持落盘目录仍按 10 题一块组织，但调度粒度改成单题，方便高并发补齐未完成 run。
# [输入输出]：输入 case_id 和分组大小；输出该 case 所在分组的起止编号。
def output_group_for_case(case_id: int, group_size: int) -> tuple[int, int]:
    start = ((case_id - 1) // group_size) * group_size + 1
    end = start + group_size - 1
    return start, end


# [模块目标]：只统计某个具体 case 已经完成的 statistics.json，避免同一 chunk 中其他 case 干扰补跑判断。
# [输入输出]：输入一个 task；输出该 task 对应 case 已完成的 run 数。
def count_case_statistics(task: CaseTask) -> int:
    case_dir_name = f"orclarify_{format_case_id(task.case_id)}"
    output_dir = Path(task.output_dir)
    if not output_dir.exists():
        return 0
    pattern = f"{task.method}/run_*/{case_dir_name}/statistics.json"
    return sum(1 for _ in output_dir.glob(pattern))


# [模块目标]：把调度进度持续写入 JSON，让主线程和监看代理不用读海量日志也能知道完成/失败情况。
# [输入输出]：输入任务列表与当前运行进程；输出 controller_status.json。
def write_status(status_path: Path, tasks: list[CaseTask], running: dict[int, CaseTask], run_tag: str, max_parallel: int) -> None:
    rows = []
    for task in tasks:
        row = asdict(task)
        row["statistics_count"] = count_case_statistics(task)
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
        "completed_statistics": sum(count_case_statistics(task) for task in tasks),
        "expected_statistics": sum(task.expected_statistics for task in tasks),
        "tasks": rows,
    }
    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_command(task: CaseTask, args: argparse.Namespace) -> list[str]:
    run_pipeline = EXPERIMENT_ROOT / task.method / "run_pipeline.py"
    return [
        sys.executable,
        str(run_pipeline),
        "--case_ids",
        format_case_id(task.case_id),
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


# [模块目标]：启动一个单题补跑任务；run_pipeline 内部会跳过已存在 statistics+judge 的 run。
# [输入输出]：输入 task 与命令行配置；输出 Python 子进程对象。
def start_task(task: CaseTask, args: argparse.Namespace) -> subprocess.Popen:
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


# [模块目标]：用较高并发把两个外部 baseline 的未完成 case-run 补齐到 ID001-100、K=5、max_turns=20。
# [输入输出]：读取命令行参数；输出每个 case 的运行目录、日志和 controller_status.json。
def main() -> None:
    parser = argparse.ArgumentParser(description="Fan out external baseline runs by case with bounded concurrency.")
    parser.add_argument("--methods", nargs="+", default=["orpilot", "lets_tpp"])
    parser.add_argument("--case_start", type=int, default=1)
    parser.add_argument("--case_end", type=int, default=100)
    parser.add_argument("--output_group_size", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--max_parallel", type=int, default=40)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--run_tag", default=f"k5_001_100_max20_{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--runs_root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--agent_profile", default="generic_agent")
    parser.add_argument("--user_profile", default="user_simulator")
    parser.add_argument("--judge_profile", default="judge")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    controller_dir = runs_root / f"external_baselines_{args.run_tag}_case_fanout_controller"
    logs_dir = controller_dir / "logs"
    controller_dir.mkdir(parents=True, exist_ok=True)
    status_path = controller_dir / "controller_status.json"

    tasks: list[CaseTask] = []
    for case_id in range(args.case_start, args.case_end + 1):
        group_start, group_end = output_group_for_case(case_id, args.output_group_size)
        for method in args.methods:
            output_dir = runs_root / f"{method}_{args.run_tag}_chunk{group_start:03d}_{group_end:03d}"
            log_path = logs_dir / f"{method}_case{case_id:03d}.log"
            tasks.append(
                CaseTask(
                    method=method,
                    case_id=case_id,
                    output_dir=str(output_dir),
                    log_path=str(log_path),
                    expected_statistics=args.k,
                )
            )

    for task in tasks:
        if count_case_statistics(task) >= task.expected_statistics:
            task.status = "completed"

    running: dict[int, CaseTask] = {}
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
            complete_count = count_case_statistics(task)
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
            if count_case_statistics(next_task) >= next_task.expected_statistics:
                next_task.status = "completed"
                continue
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
        raise SystemExit(f"{len(failed)} case tasks failed. See {status_path}")


if __name__ == "__main__":
    main()
