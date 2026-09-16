"""为旧版原始 episode 回填可选的视频统计 sidecar。"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .raw_schema import RAW_DATASET_FORMAT_V1, RAW_VIDEO_STATS_FILE
from .video_stats import (
    compute_video_stats,
    read_raw_video_stats,
    video_image_shape,
    write_raw_video_stats,
)

CAMERAS = ("global", "wrist")


@dataclass(frozen=True)
class BackfillReport:
    """汇总回填结果，便于 CLI 和自动化任务消费。"""

    backfilled: int = 0
    dry_run: int = 0
    refused: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        """转换为稳定的 JSON 输出结构。"""
        return asdict(self)


def episode_paths(raw_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """收集 raw 根目录内的 episode；根目录本身也可为一个 episode。"""
    paths = []
    for root in raw_roots:
        candidates = (root,) if (root / "metadata.json").exists() else root.iterdir()
        paths.extend(
            path for path in candidates if path.is_dir() and (path / "metadata.json").exists()
        )
    return tuple(sorted(set(paths)))


def action_count(path: Path) -> int:
    """只读取轨迹动作数组的长度，不改写任何旧版源文件。"""
    with np.load(path / "trajectory.npz", allow_pickle=False) as trajectory:
        return int(trajectory["command_delta_pose_seam"].shape[0])


def backfill_episode(path: Path, apply: bool) -> str:
    """处理单个 v1 episode，返回其可汇总的迁移状态。"""
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_format", RAW_DATASET_FORMAT_V1) != RAW_DATASET_FORMAT_V1:
        return "skipped"
    count = action_count(path)
    sidecar = path / RAW_VIDEO_STATS_FILE
    if sidecar.exists():
        try:
            read_raw_video_stats(path, CAMERAS, count)
        except ValueError:
            return "refused"
        return "skipped"
    if not apply:
        return "dry_run"
    statistics = {
        name: compute_video_stats(path / f"{name}.mp4", count, terminal_frame=True)
        for name in CAMERAS
    }
    try:
        write_raw_video_stats(
            path,
            statistics,
            count,
            image_shapes={name: video_image_shape(path / f"{name}.mp4") for name in CAMERAS},
            overwrite=False,
        )
    except FileExistsError:
        return "refused"
    return "backfilled"


def backfill_raw_video_stats(
    raw_roots: tuple[str | Path, ...], workers: int, apply: bool
) -> BackfillReport:
    """并行回填 v1 视频统计；有效 sidecar 作为可恢复的完成标记。

    Args:
        raw_roots: 包含 episode 子目录的原始数据集根目录。
        workers: 解码进程数量，必须为正数。
        apply: 为 False 时仅预演，绝不写入文件。

    Returns:
        各处理结果的数量汇总。
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    paths = episode_paths(tuple(Path(root) for root in raw_roots))
    if workers == 1:
        states = [backfill_episode(path, apply) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            states = list(executor.map(backfill_episode, paths, [apply] * len(paths)))
    return BackfillReport(
        **{name: states.count(name) for name in BackfillReport.__dataclass_fields__}
    )


__all__ = ["BackfillReport", "backfill_raw_video_stats"]
