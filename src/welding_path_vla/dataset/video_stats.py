"""原始 RGB 视频的动作对齐统计和完整性校验。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.compute_stats import RunningQuantileStats, auto_downsample_height_width

from .raw_schema import RAW_DATASET_FORMAT, RAW_VIDEO_STATS_FILE

ALGORITHM = "action_aligned_rgb_v1"
COLOR_SPACE = "RGB"
FRAME_SELECTION = "first_action_count_frames_excluding_terminal_state"
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


def canonicalize_statistics(stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """夹紧并单调化直方图分位数，保持原始 RGB 统计量纲。"""
    fields = ("min", "q01", "q10", "q50", "q90", "q99", "max")
    values = np.stack([stats[field] for field in fields])
    values = np.maximum.accumulate(np.clip(values, stats["min"], stats["max"]), axis=0)
    stats.update(zip(fields, values, strict=True))
    return stats


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
    ordered = np.stack([stats[key] for key in ("min", "q01", "q10", "q50", "q90", "q99", "max")])
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
    return canonicalize_statistics(statistics.get_statistics())


def video_image_shape(path: str | Path) -> tuple[int, int, int]:
    """读取视频流的 HWC 图像形状，不解码像素。"""
    capture = cv2.VideoCapture(str(path))
    try:
        shape = (
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3,
        )
    finally:
        capture.release()
    validate_image_shapes({"camera": shape}, ("camera",))
    return shape


def validate_image_shapes(image_shapes: dict, camera_names: tuple[str, ...]) -> None:
    """验证逐相机 HWC 图像形状与相机集合。"""
    if not isinstance(image_shapes, dict) or set(image_shapes) != set(camera_names):
        raise ValueError("invalid image metadata")
    for shape in image_shapes.values():
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[2] != 3
        ):
            raise ValueError("invalid image metadata")


class ActionAlignedVideoStats:
    """以一帧延迟方式排除视频末尾的 terminal frame。"""

    def __init__(self, camera_names: tuple[str, ...]) -> None:
        self.camera_names = camera_names
        self.pending: dict[str, np.ndarray] | None = None
        self.statistics = {name: RunningQuantileStats() for name in camera_names}
        self.frame_count = 0
        self.image_shapes: dict[str, tuple[int, int, int]] = {}

    def append(self, images: dict[str, np.ndarray]) -> None:
        """缓存当前帧，并将上一帧加入动作统计样本。"""
        if set(images) != set(self.camera_names):
            raise ValueError("images must contain exactly the configured cameras")
        for frame in images.values():
            image_samples(frame)
        shapes = {name: images[name].shape for name in self.camera_names}
        if not self.image_shapes:
            self.image_shapes = shapes
        elif shapes != self.image_shapes:
            raise ValueError("image shapes must remain constant")
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
            name: canonicalize_statistics(statistics.get_statistics())
            for name, statistics in self.statistics.items()
        }


def write_raw_video_stats(
    root: str | Path,
    stats: dict,
    action_count: int,
    image_shapes: dict[str, tuple[int, int, int]],
    overwrite: bool = True,
) -> Path:
    """原子写入 v2 sidecar，并拒绝非有限统计值。

    Args:
        root: sidecar 所属 episode 目录。
        stats: 每路相机的 canonical 统计。
        action_count: 与视频帧对齐的动作数量。
        overwrite: 为 False 时拒绝覆盖并发出现的目标 sidecar。
    """
    for camera in stats.values():
        validate_statistics(camera)
    validate_image_shapes(image_shapes, tuple(stats))
    root = Path(root)
    path = root / RAW_VIDEO_STATS_FILE
    payload = {
        "format": RAW_DATASET_FORMAT,
        "algorithm": ALGORITHM,
        "color_space": COLOR_SPACE,
        "frame_selection": FRAME_SELECTION,
        "action_count": action_count,
        "state_count": action_count + 1,
        "cameras": list(stats),
        "image_shapes": {name: list(shape) for name, shape in image_shapes.items()},
        "dtype": "float64",
        "stats": {
            name: {key: np.asarray(value).tolist() for key, value in camera.items()}
            for name, camera in stats.items()
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    if not overwrite:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=root, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    if overwrite:
        os.replace(temporary, path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def read_raw_video_stats(
    root: str | Path, camera_names: tuple[str, ...], action_count: int
) -> dict:
    """读取并校验 sidecar 的格式、相机集合、计数和数值完整性。"""
    try:
        payload = json.loads((Path(root) / RAW_VIDEO_STATS_FILE).read_text(encoding="utf-8"))
        if payload.get("format") != RAW_DATASET_FORMAT or payload.get("algorithm") != ALGORITHM:
            raise ValueError("invalid format or algorithm")
        if payload.get("dtype") != "float64":
            raise ValueError("invalid statistics dtype")
        if (
            payload.get("color_space") != COLOR_SPACE
            or payload.get("frame_selection") != FRAME_SELECTION
        ):
            raise ValueError("invalid image metadata")
        if (
            payload.get("action_count") != action_count
            or payload.get("state_count") != action_count + 1
            or set(payload.get("cameras", ())) != set(camera_names)
        ):
            raise ValueError("invalid action count or camera set")
        validate_image_shapes(payload.get("image_shapes"), camera_names)
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


__all__ = [
    "ActionAlignedVideoStats",
    "compute_video_stats",
    "image_samples",
    "read_raw_video_stats",
    "write_raw_video_stats",
]
