#!/usr/bin/env python3
"""依次采集全部仿真任务，并安全处理 Ctrl+C。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CollectionTask:
    """一个公开任务配置及其默认数据集目录名。"""

    task_id: str
    config: str
    dataset_name: str
    episodes: int


TASKS = (
    # CollectionTask("l_joint", "configs/default.yaml", "l_joint_raw_v2", 1200),
    # CollectionTask("curve_plate", "configs/curve_plate.yaml", "curve_plate_raw_v2", 1000),
    # CollectionTask(
    #     "trihedral_vertical",
    #     "configs/trihedral_vertical.yaml",
    #     "trihedral_vertical_raw_v2",
    #     1000,
    # ),
    # CollectionTask(
    #     "trihedral_horizontal",
    #     "configs/trihedral_horizontal.yaml",
    #     "trihedral_horizontal_raw_v2",
    #     800
    # ),
    # CollectionTask("pipe_bottom", "configs/pipe_bottom.yaml", "pipe_bottom_raw_v2", 910),
    CollectionTask("pipe_top", "configs/pipe_top.yaml", "pipe_top_raw_v2", 890),
)


def start_collection(command: list[str]) -> subprocess.Popen[bytes]:
    """在独立进程组启动采集，便于一次释放主进程和全部 worker。"""
    return subprocess.Popen(command, start_new_session=True)


def process_group_exists(group_id: int) -> bool:
    """判断指定 POSIX 进程组是否仍有成员。"""
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    return True


def interrupt_collection(process: subprocess.Popen[bytes], timeout_s: float = 30) -> None:
    """转发 Ctrl+C，等待总结落盘，并兜底释放全部子进程。"""
    if process.poll() is not None and not process_group_exists(process.pid):
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    if process_group_exists(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while process_group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if process_group_exists(process.pid):
        os.killpg(process.pid, signal.SIGKILL)


def collection_command(
    task: CollectionTask,
    dataset_root: Path,
    # episodes: int,
    workers: int,
) -> list[str]:
    """构造与单任务入口完全一致的采集命令。"""
    return [
        sys.executable,
        "scripts/collect_simulation_data.py",
        f"--config_path={task.config}",
        f"--collection.episodes={task.episodes}",
        f"--collection.dataset_root={dataset_root / task.dataset_name}",
        f"--collection.workers={workers}",
        # f"--randomization.task_group_size={5}",
    ]


def print_summary(root: Path) -> None:
    """显示刚完成任务的成功率和累计状态分布。"""
    path = root / "dataset.json"
    if not path.exists():
        print(f"⚠️ 未找到总结文件: {path}")
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    attempts = int(summary.get("last_request_attempts", 0))
    valid = int(summary.get("last_request_collected_valid_episodes", 0))
    success_rate = valid / attempts if attempts else 0.0
    print(f"📊 本次成功率 {success_rate:.1%} ({valid}/{attempts}), 累计状态: {summary['status']}")
    joint1 = summary.get("last_request_initial_joint1_deg")
    tcp_span = summary.get("last_request_initial_tcp_span_m")
    if joint1:
        print(
            f"🎲 初始轴1 {joint1['min']:.1f}°-{joint1['max']:.1f}°"
            f" (跨度 {joint1['span']:.1f}°), TCP XYZ 跨度 {tcp_span} m"
        )


def run_collections(args: argparse.Namespace) -> int:
    """顺序执行所选任务；单个任务失败时继续，手动中断时立即停止。"""
    selected = [task for task in TASKS if not args.tasks or task.task_id in args.tasks]
    dataset_root = Path(args.dataset_root)
    for index, task in enumerate(selected, 1):
        root = dataset_root / task.dataset_name
        command = collection_command(
            task,
            dataset_root,
            # args.episodes,
            args.workers,
        )
        print(
            f"\n{'=' * 60}\n🚀 {index}/{len(selected)}: {task.task_id}\n{' '.join(command)}",
            flush=True,
        )
        started = time.monotonic()
        process = start_collection(command)
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            print("\n⚠️ 正在中断采集并等待 dataset.json 保存……")
            interrupt_collection(process)
            print_summary(root)
            return 130
        print_summary(root)
        elapsed = (time.monotonic() - started) / 60
        if return_code:
            print(f"❌ {task.task_id} 失败, 退出码 {return_code}, 耗时 {elapsed:.1f} 分钟")
        else:
            print(f"✅ {task.task_id} 完成, 耗时 {elapsed:.1f} 分钟")
    return 0


def parse_args() -> argparse.Namespace:
    """解析批量采集所需的少量公共覆盖参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", default="/run/media/mintinson/DataDiskD/welding_path_vla/datasets"
    )
    # parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks", nargs="*", choices=[task.task_id for task in TASKS])
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_collections(parse_args()))
