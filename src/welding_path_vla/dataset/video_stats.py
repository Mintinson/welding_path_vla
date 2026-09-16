"""原始 RGB 视频的动作对齐统计和完整性校验。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.compute_stats import RunningQuantileStats, auto_downsample_height_width

from .raw_schema import RAW_DATASET_FORMAT, RAW_VIDEO_STATS_FILE

ALGORITHM = "action_aligned_rgb_v1"
STAT_FIELDS = frozenset(("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"))


def image_samples(image: np.ndarray) -> np.ndarray:
    """验证并展平一帧原始 RGB 图像，保留 0--255 数值。"""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be RGB uint8")
    return image.reshape(-1, 3).astype(np.float64)


def statistic_samples(image: np.ndarray) -> np.ndarray:
    """将 RGB 图像按 LeRobot 规则下采样并展平为逐通道统计样本。"""
    image_samples(image)
    downsampled = auto_downsample_height_width(image.transpose(2, 0, 1))
    return downsampled.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64)


def validate_statistics(stats: dict) -> None:
    """验证 canonical 视频统计的完整字段、形状、dtype 与有限性。"""
    if set(stats) != STAT_FIELDS:
        raise ValueError("invalid statistic fields")
    for key, value in stats.items():
        array = np.asarray(value)
        if key == "count":
            if array.shape != (1,) or array.dtype != np.int64 or array[0] <= 0:
                raise ValueError("statistics must have shape or dtype (1,) int64 and be positive")
        elif array.shape != (3,) or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("statistics must have shape or dtype (3,) float64 and be finite")
    ordered = np.stack(
        [stats[key] for key in ("min", "q01", "q10", "q50", "q90", "q99", "max")]
    )
    if np.any(np.diff(ordered, axis=0) < 0):
        raise ValueError("statistics quantiles must be ordered")
    if np.any(stats["mean"] < stats["min"]) or np.any(stats["mean"] > stats["max"]):
        raise ValueError("statistics mean must be within min and max")
    if np.any(stats["std"] < 0):
        raise ValueError("statistics std must be nonnegative")


def compute_video_stats(
    path: str | Path, frame_count: int, terminal_frame: bool = False
) -> dict[str, np.ndarray]:
    """解码动作帧统计，保留 LeRobot 归一化前的 0--255 RGB 值。

    Args:
        path: 单路相机视频路径。
        frame_count: 与动作对齐的前 N 帧数量。
        terminal_frame: 兼容 repair 调用；统计始终只读取前 N 个动作帧。

    Returns:
        LeRobot writer 所需的逐通道原始统计。
    """
    capture = cv2.VideoCapture(str(path))
    statistics = RunningQuantileStats()
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video ended before action {index}: {path}")
            statistics.update(statistic_samples(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    result = statistics.get_statistics()
    # LeRobot 直方图边界可能让 q01 略小于 min，统一夹紧为 canonical 契约。
    fields = ("min", "q01", "q10", "q50", "q90", "q99", "max")
    values = np.stack([result[field] for field in fields])
    values = np.maximum.accumulate(np.clip(values, result["min"], result["max"]), axis=0)
    result.update(zip(fields, values, strict=True))
    return result


class ActionAlignedVideoStats:
    """以一帧延迟方式排除视频末尾的 terminal frame。"""

    def __init__(self, camera_names: tuple[str, ...]) -> None:
        self.camera_names = camera_names
        self.pending: dict[str, np.ndarray] | None = None
        self.statistics = {name: RunningQuantileStats() for name in camera_names}
        self.frame_count = 0

    def append(self, images: dict[str, np.ndarray]) -> None:
        """缓存当前帧，并将上一帧加入动作统计样本。"""
        if set(images) != set(self.camera_names):
            raise ValueError("images must contain exactly the configured cameras")
        for frame in images.values():
            image_samples(frame)
        if self.pending is not None:
            for name, frame in self.pending.items():
                self.statistics[name].update(statistic_samples(frame))
        self.pending = {name: images[name].copy() for name in self.camera_names}
        self.frame_count += 1

    def finish(self, action_count: int) -> dict[str, dict[str, np.ndarray]]:
        """返回每个相机的完整 canonical 逐通道统计。"""
        if self.frame_count != action_count + 1:
            raise ValueError("expected N action frames and one terminal frame")
        return {
            name: statistics.get_statistics()
            for name, statistics in self.statistics.items()
        }


def write_raw_video_stats(root: str | Path, stats: dict, action_count: int) -> Path:
    """原子写入 v2 sidecar，并拒绝非有限统计值。"""
    for camera in stats.values():
        validate_statistics(camera)
    root = Path(root)
    path = root / RAW_VIDEO_STATS_FILE
    payload = {"format": RAW_DATASET_FORMAT, "algorithm": ALGORITHM,
               "action_count": action_count, "state_count": action_count + 1,
               "cameras": list(stats), "dtype": "float64",
               "stats": {name: {key: np.asarray(value).tolist() for key, value in camera.items()}
                         for name, camera in stats.items()}}
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_raw_video_stats(root: str | Path, camera_names: tuple[str, ...], action_count: int) -> dict:
    """读取并校验 sidecar 的格式、相机集合、计数和数值完整性。"""
    try:
        payload = json.loads((Path(root) / RAW_VIDEO_STATS_FILE).read_text(encoding="utf-8"))
        if payload.get("format") != RAW_DATASET_FORMAT or payload.get("algorithm") != ALGORITHM:
            raise ValueError("invalid format or algorithm")
        if payload.get("dtype") != "float64":
            raise ValueError("invalid statistics dtype")
        if (payload.get("action_count") != action_count
                or payload.get("state_count") != action_count + 1
                or set(payload.get("cameras", ())) != set(camera_names)):
            raise ValueError("invalid action count or camera set")
        result = {}
        for name in camera_names:
            camera = payload["stats"][name]
            result[name] = {key: np.asarray(value) for key, value in camera.items()}
            validate_statistics(result[name])
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("invalid raw video statistics sidecar") from error


__all__ = ["ActionAlignedVideoStats", "compute_video_stats", "image_samples", "read_raw_video_stats", "write_raw_video_stats"]
