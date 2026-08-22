#!/usr/bin/env python3
"""从 LeRobot 数据集中移除带技术参数的旧版 pipe-bottom episode。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import cli, output_json

LEGACY_MARKER = "; start at a local circumferential angle"
MANIFEST = Path("meta/welding_path_vla_export.json")


@dataclass
class CleanupArguments:
    """旧数据清理参数。

    Attributes:
        source: 原始 LeRobot 数据集目录。
        destination: 清理后数据集目录；必须与 source 不同且尚不存在。
        repo_id: 数据集的稳定 Hugging Face 仓库标识。
        expected_episodes: 预期删除数量，防止错误匹配扩大删除范围。
    """

    source: Path
    destination: Path
    repo_id: str = "huayan/weldpath_relative_v1"
    expected_episodes: int = 10


def legacy_episode_indices(dataset: Any) -> list[int]:
    """返回 instruction 中含旧版技术描述的 episode 索引。"""
    episodes = dataset.meta.episodes
    return [
        int(episode["episode_index"])
        for episode in episodes
        if any(LEGACY_MARKER in task for task in episode["tasks"])
    ]


def filtered_manifest(source: Path, removed: set[int]) -> dict:
    """删除清单中与全局 episode 索引对应的源记录。"""
    document = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    offset = 0
    for name, episodes in document["sources"].items():
        document["sources"][name] = [
            episode for index, episode in enumerate(episodes, offset) if index not in removed
        ]
        offset += len(episodes)
    return document


def write_manifest(destination: Path, document: dict) -> None:
    """把项目动作表示清单写入 LeRobot 官方编辑后的数据集。"""
    path = destination / MANIFEST
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@cli
def main(config: CleanupArguments) -> None:
    """删除旧 episode，并验证任务表与项目清单保持一致。"""
    from lerobot.datasets import LeRobotDataset, delete_episodes

    source = config.source.resolve()
    destination = config.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    dataset = LeRobotDataset(config.repo_id, root=source)
    indices = legacy_episode_indices(dataset)
    if len(indices) != config.expected_episodes:
        raise ValueError(f"expected {config.expected_episodes} legacy episodes, found {indices}")

    manifest = filtered_manifest(source, set(indices))
    cleaned = delete_episodes(dataset, indices, output_dir=destination, repo_id=config.repo_id)
    write_manifest(destination, manifest)
    remaining = legacy_episode_indices(cleaned)
    if remaining:
        raise RuntimeError(f"legacy episodes remain after cleanup: {remaining}")
    manifest_episodes = sum(len(episodes) for episodes in manifest["sources"].values())
    if manifest_episodes != cleaned.meta.total_episodes:
        raise RuntimeError("export manifest and LeRobot episode count differ")
    output_json(
        {
            "source": str(source),
            "destination": str(destination),
            "removed_episode_indices": indices,
            "episodes": cleaned.meta.total_episodes,
            "tasks": cleaned.meta.total_tasks,
        }
    )


if __name__ == "__main__":
    main()
