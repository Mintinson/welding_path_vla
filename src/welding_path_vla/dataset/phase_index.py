"""Raw trajectory phase 与 LeRobot 全局 frame 的 sidecar 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PHASE_CODES = {"approach": 0, "track": 1, "retreat": 2}
PHASE_NAMES = tuple(PHASE_CODES)
PHASE_INDEX_NAME = "phase_index.npz"


def phase_values(trajectory: Any) -> np.ndarray:
    """把 raw trajectory 中的字符串 phase 编码为 uint8。"""
    if "phase" not in trajectory:
        raise ValueError("raw trajectory is missing required phase array")
    values = np.asarray(trajectory["phase"])
    if values.ndim != 1:
        raise ValueError(f"raw phase must be one-dimensional, got shape={values.shape}")
    decoded = [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    unknown = sorted(set(decoded) - PHASE_CODES.keys())
    if unknown:
        raise ValueError(f"raw trajectory contains unknown phase values: {unknown}")
    return np.asarray([PHASE_CODES[value] for value in decoded], dtype=np.uint8)


def phase_index_path(dataset_root: str | Path) -> Path:
    """返回 LeRobot 数据集的 phase sidecar 路径。"""
    return Path(dataset_root) / "meta" / PHASE_INDEX_NAME


def load_phase_index(path: str | Path) -> np.ndarray:
    """读取并校验 frame 顺序的 phase 编码。"""
    with np.load(path, allow_pickle=False) as sidecar:
        phase = np.asarray(sidecar["phase"], dtype=np.uint8)
    if phase.ndim != 1 or np.any(phase > max(PHASE_CODES.values())):
        raise ValueError(f"invalid phase index: {path}")
    return phase


def write_phase_index(path: str | Path, phase: np.ndarray) -> None:
    """原子写入压缩 phase sidecar。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.npz")
    np.savez_compressed(temporary, phase=np.asarray(phase, dtype=np.uint8))
    temporary.replace(target)


def append_phase_index(
    dataset_root: str | Path,
    chunks: list[np.ndarray],
    expected_frames: int,
) -> None:
    """把本次导出的 episode phase 追加到全局 LeRobot frame 索引。"""
    target = phase_index_path(dataset_root)
    existing = load_phase_index(target) if target.exists() else np.empty(0, dtype=np.uint8)
    phase = np.concatenate([existing, *chunks]) if chunks else existing
    if len(phase) != expected_frames:
        raise ValueError(
            f"phase index has {len(phase)} frames, but LeRobot metadata has {expected_frames}"
        )
    write_phase_index(target, phase)


__all__ = [
    "PHASE_CODES",
    "PHASE_INDEX_NAME",
    "PHASE_NAMES",
    "append_phase_index",
    "load_phase_index",
    "phase_index_path",
    "phase_values",
    "write_phase_index",
]
