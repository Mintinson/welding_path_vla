#!/usr/bin/env python3
"""为 LeRobot 数据集生成按全局 frame index 对齐的 phase sidecar。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from welding_path_vla.core.config_files import compose_config
from welding_path_vla.dataset.phase_index import phase_values, write_phase_index


def source_task(source_id: str) -> str:
    """从导出清单的稳定 source ID 中读取 task ID。"""
    return source_id.split("|", 1)[0]


def build_phase_index(dataset_root: Path, raw_root: Path, output: Path) -> None:
    """按导出清单把 raw trajectory phase 对齐到 LeRobot 全局帧。"""
    manifest = json.loads(
        (dataset_root / "meta/welding_path_vla_export.json").read_text(encoding="utf-8")
    )
    metadata = LeRobotDatasetMetadata("local/phase-index", root=dataset_root)
    episodes_by_task: dict[str, list[dict]] = defaultdict(list)
    for episode in metadata.episodes:
        episodes_by_task[episode["tasks"][0]].append(episode)
    for episodes in episodes_by_task.values():
        episodes.sort(key=lambda item: int(item["episode_index"]))

    cursors: dict[str, int] = defaultdict(int)
    phases = np.full(metadata.total_frames, 255, dtype=np.uint8)
    for source_id, episode_names in manifest["sources"].items():
        task_id = source_task(source_id)
        task_config = compose_config(Path("configs/tasks") / f"{task_id}.yaml")
        instruction = str(task_config["task"]["instruction"])
        target_episodes = episodes_by_task.get(instruction, [])
        for episode_name in episode_names:
            cursor = cursors[task_id]
            if cursor >= len(target_episodes):
                raise ValueError(f"manifest has too many {task_id} episodes")
            target = target_episodes[cursor]
            cursors[task_id] += 1
            trajectory_path = raw_root / task_id / "episodes" / episode_name / "trajectory.npz"
            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                values = phase_values(trajectory)
            start, end = int(target["dataset_from_index"]), int(target["dataset_to_index"])
            if len(values) != end - start:
                raise ValueError(
                    f"phase length mismatch for {task_id}/{episode_name}: "
                    f"raw={len(values)}, lerobot={end - start}"
                )
            phases[start:end] = values

    if np.any(phases == 255):
        raise ValueError("phase index has unfilled LeRobot frames")
    write_phase_index(output, phases)
    print(f"wrote {output} ({len(phases)} frames)")


def main() -> None:
    """解析命令行并生成 sidecar。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_phase_index(args.dataset_root, args.raw_root, args.output)


if __name__ == "__main__":
    main()
