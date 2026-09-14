#!/usr/bin/env python3
"""从 LeRobot metadata 汇总数据集和各任务规模。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.data import (
    dataset_task_summaries,
    filtered_episode_indices,
    metadata,
)


@dataclass
class DatasetInfoArguments(AppConfig):
    """数据集统计配置。"""

    output: Path | None = None
    show_episode_indices: bool = False
    sort_by: str = "task_index"


@cli
def main(config: DatasetInfoArguments) -> None:
    """输出总体规模、任务名称以及筛选前后的 episode/帧数。"""
    meta = metadata(config.training)
    summaries = dataset_task_summaries(config.training)
    filtered = filtered_episode_indices(config.training)
    selected = set(filtered) if filtered is not None else {
        episode for summary in summaries for episode in summary.episode_indices
    }
    lengths = {
        int(episode["episode_index"]): int(episode["length"]) for episode in meta.episodes
    }
    tasks: list[dict[str, Any]] = []
    for summary in summaries:
        selected_indices = [index for index in summary.episode_indices if index in selected]
        item: dict[str, Any] = {
            "task_index": summary.task_index,
            "name": summary.name,
            "episodes": summary.episodes,
            "frames": summary.frames,
            "selected_episodes": len(selected_indices),
            "selected_frames": sum(lengths[index] for index in selected_indices),
        }
        if config.show_episode_indices:
            item["episode_indices"] = list(summary.episode_indices)
            item["selected_episode_indices"] = selected_indices
        tasks.append(item)
    sorters = {
        "task_index": lambda item: item["task_index"],
        "name": lambda item: item["name"],
        "episodes": lambda item: (-item["episodes"], item["task_index"]),
        "frames": lambda item: (-item["frames"], item["task_index"]),
    }
    if config.sort_by not in sorters:
        raise ValueError(f"sort_by must be one of {sorted(sorters)}")
    tasks.sort(key=sorters[config.sort_by])
    output_json(
        {
            "repo_id": config.training.dataset_repo_id,
            "root": config.training.dataset_root,
            "fps": meta.fps,
            "total_episodes": meta.total_episodes,
            "total_frames": meta.total_frames,
            "total_tasks": len(summaries),
            "include_tasks": config.training.include_tasks,
            "exclude_tasks": config.training.exclude_tasks,
            "selected_episodes": len(selected),
            "selected_frames": sum(lengths[index] for index in selected),
            "selected_tasks": sum(item["selected_episodes"] > 0 for item in tasks),
            "tasks": tasks,
        },
        config.output,
    )


if __name__ == "__main__":
    main()
