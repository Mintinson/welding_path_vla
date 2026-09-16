"""原始 RGB 视频的动作对齐统计和完整性校验。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .raw_schema import RAW_DATASET_FORMAT, RAW_VIDEO_STATS_FILE

ALGORITHM = "action_aligned_rgb_v1"


def image_samples(image: np.ndarray) -> np.ndarray:
    """验证并展平一帧原始 RGB 图像，保留 0--255 数值。"""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be RGB uint8")
    return image.reshape(-1, 3).astype(np.float64)


class ActionAlignedVideoStats:
    """以一帧延迟方式排除视频末尾的 terminal frame。"""

    def __init__(self, camera_names: tuple[str, ...]) -> None:
        self.camera_names = camera_names
        self.pending: dict[str, np.ndarray] | None = None
        self.frames: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
        self.frame_count = 0

    def append(self, images: dict[str, np.ndarray]) -> None:
        """缓存当前帧，并将上一帧加入动作统计样本。"""
        if set(images) != set(self.camera_names):
            raise ValueError("images must contain exactly the configured cameras")
        for frame in images.values():
            image_samples(frame)
        if self.pending is not None:
            for name, frame in self.pending.items():
                self.frames[name].append(image_samples(frame))
        self.pending = {name: images[name].copy() for name in self.camera_names}
        self.frame_count += 1

    def finish(self, action_count: int) -> dict[str, dict[str, np.ndarray]]:
        """返回每个相机的逐通道均值、最小值和最大值。"""
        if self.frame_count != action_count + 1:
            raise ValueError("expected N action frames and one terminal frame")
        return {
            name: {key: getattr(np, reducer)(np.concatenate(self.frames[name]), axis=0)
                   for key, reducer in (("mean", "mean"), ("min", "min"), ("max", "max"))}
            for name in self.camera_names
        }


def write_raw_video_stats(root: str | Path, stats: dict, action_count: int) -> Path:
    """原子写入 v2 sidecar，并拒绝非有限统计值。"""
    for camera in stats.values():
        for value in camera.values():
            array = np.asarray(value)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError("statistics must be finite arrays with shape (3,)")
    root = Path(root)
    path = root / RAW_VIDEO_STATS_FILE
    payload = {"format": RAW_DATASET_FORMAT, "algorithm": ALGORITHM,
               "action_count": action_count, "cameras": list(stats),
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
        if payload.get("action_count") != action_count or set(payload.get("cameras", ())) != set(camera_names):
            raise ValueError("invalid action count or camera set")
        result = {}
        for name in camera_names:
            camera = payload["stats"][name]
            result[name] = {key: np.asarray(value) for key, value in camera.items()}
            if set(result[name]) != {"mean", "min", "max"}:
                raise ValueError("invalid statistics fields")
            for value in result[name].values():
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError("statistics must be finite arrays with shape (3,)")
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("invalid raw video statistics sidecar") from error


__all__ = ["ActionAlignedVideoStats", "image_samples", "read_raw_video_stats", "write_raw_video_stats"]
