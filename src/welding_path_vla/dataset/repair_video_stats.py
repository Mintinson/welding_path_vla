"""增量修复原始 episode 的归一化视频统计检查点。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from welding_path_vla.dataset.raw_schema import (
    RAW_DATASET_FORMAT_V1,
    RAW_DATASET_FORMAT_V2,
    RAW_VIDEO_STATS_FILE,
    EpisodeReader,
)
from welding_path_vla.dataset.video_stats import compute_video_stats, read_raw_video_stats

CAMERAS = ("global", "wrist")


def normalize_statistics(statistics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """转换 raw 0--255 统计为既有 LeRobot 检查点量纲。"""
    return {
        name: value.copy() if name == "count" else (value / 255).reshape(3, 1, 1)
        for name, value in statistics.items()
    }


def episode_fingerprint(episode: EpisodeReader) -> dict[str, object]:
    """生成决定单个 episode repair 缓存有效性的文件指纹。"""
    paths = [episode.path / f"{name}.mp4" for name in CAMERAS]
    sidecar = episode.path / RAW_VIDEO_STATS_FILE
    if sidecar.exists():
        paths.append(sidecar)
    files = {
        str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    }
    algorithm = (
        json.loads(sidecar.read_text(encoding="utf-8")).get("algorithm")
        if sidecar.exists()
        else None
    )
    return {"files": files, "algorithm": algorithm}


def compute_episode(episode: EpisodeReader) -> dict[str, dict[str, np.ndarray]]:
    """优先复用 v2 raw sidecar，缺失的 v1 episode 解码回退。"""
    format_name = episode.metadata.get("dataset_format", RAW_DATASET_FORMAT_V1)
    sidecar = episode.path / RAW_VIDEO_STATS_FILE
    if format_name == RAW_DATASET_FORMAT_V2 or sidecar.exists():
        raw = read_raw_video_stats(episode.path, CAMERAS, episode.action_count)
    elif format_name == RAW_DATASET_FORMAT_V1:
        raw = {
            name: compute_video_stats(
                episode.path / f"{name}.mp4", episode.action_count, terminal_frame=True
            )
            for name in CAMERAS
        }
    else:
        raise ValueError(f"unsupported raw dataset format: {format_name}")
    return {name: normalize_statistics(statistics) for name, statistics in raw.items()}


def read_checkpoint(path: Path) -> dict[str, dict]:
    """读取已有 repair 检查点；不存在时返回空检查点。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_statistics(statistics: dict[str, dict[str, np.ndarray]]) -> dict:
    """把 NumPy 检查点统计转换为 JSON 可持久化结构。"""
    return {
        camera: {name: value.tolist() for name, value in values.items()}
        for camera, values in statistics.items()
    }


def deserialize_statistics(statistics: dict) -> dict[str, dict[str, np.ndarray]]:
    """从 JSON 检查点恢复 NumPy 统计，保持调用方原有接口。"""
    return {
        camera: {
            name: np.asarray(value, dtype=np.int64 if name == "count" else np.float64)
            for name, value in values.items()
        }
        for camera, values in statistics.items()
    }


def repair_video_stats(episode_paths: tuple[Path, ...], checkpoint_path: str | Path) -> dict:
    """增量生成归一化统计，只有指纹变化的 episode 才重新计算。"""
    path = Path(checkpoint_path)
    previous = read_checkpoint(path)
    checkpoint = {}
    for episode_path in episode_paths:
        episode = EpisodeReader(episode_path)
        key = str(episode.path)
        fingerprint = episode_fingerprint(episode)
        cached = previous.get(key)
        statistics = (
            deserialize_statistics(cached["statistics"])
            if cached and cached["fingerprint"] == fingerprint
            else compute_episode(episode)
        )
        checkpoint[key] = {"fingerprint": fingerprint, "statistics": statistics}
    path.write_text(
        json.dumps(
            {
                key: {
                    "fingerprint": value["fingerprint"],
                    "statistics": serialize_statistics(value["statistics"]),
                }
                for key, value in checkpoint.items()
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


__all__ = ["compute_episode", "episode_fingerprint", "repair_video_stats"]
