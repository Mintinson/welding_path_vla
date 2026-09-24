"""按参考轨迹阶段重采样 LeRobot 训练帧。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from welding_path_vla.core.config import PhaseSamplingConfig
from welding_path_vla.dataset.phase_index import PHASE_CODES, load_phase_index


def phase_index_path(config: PhaseSamplingConfig, dataset_root: str | Path | None) -> Path:
    """解析 phase sidecar 路径；默认放在 LeRobot 数据集的 meta 目录。"""
    if config.index_path:
        return Path(config.index_path)
    if dataset_root is None:
        raise ValueError("phase_sampling.index_path is required when dataset_root is unset")
    return Path(dataset_root) / "meta/phase_index.npz"


class PhaseBalancedSampler:
    """保持 episode 边界并按 approach/track/retreat 比例重采样。

    采样顺序由 ``(seed, epoch)`` 唯一决定，因此所有 accelerate 进程会先得到相同
    的全局顺序，再由 accelerate 分片；``state_dict`` 与 LeRobot 默认 sampler 兼容。
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        phase_index_path: str | Path,
        episode_indices_to_use: list[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
        target_ratios: tuple[float, float, float] = (0.4, 0.5, 0.1),
        action_horizon: int = 30,
        action_stride: int = 1,
        track_threshold: float = 0.5,
    ) -> None:
        if not shuffle:
            raise ValueError("PhaseBalancedSampler requires shuffle=True")
        if action_horizon < 1 or action_stride < 1:
            raise ValueError("phase sampling action horizon and stride must be positive")
        if not 0 < track_threshold <= 1:
            raise ValueError("phase sampling track_threshold must be in (0, 1]")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError("episode boundary arrays must have the same shape")
        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        ends = to_indices - drop_n_last_frames
        used &= ends > starts
        if not used.any():
            raise ValueError("no valid frames remain for phase-balanced sampling")

        phase = load_phase_index(phase_index_path)
        if len(phase) < int(np.max(ends[used])):
            raise ValueError(
                f"phase index has {len(phase)} frames, but dataset requires "
                f"{int(np.max(ends[used]))}"
            )

        buckets: list[list[np.ndarray]] = [[], [], []]
        offsets = np.arange(action_horizon, dtype=np.int64) * action_stride
        for start, end in zip(starts[used], ends[used], strict=True):
            indices = np.arange(start, end, dtype=np.int64)
            window = np.minimum(indices[:, None] + offsets[None, :], end - 1)
            track_fraction = np.mean(phase[window] == PHASE_CODES["track"], axis=1)
            current = phase[indices]
            bucket = np.where(
                track_fraction >= track_threshold,
                PHASE_CODES["track"],
                np.where(
                    current == PHASE_CODES["retreat"],
                    PHASE_CODES["retreat"],
                    PHASE_CODES["approach"],
                ),
            )
            for code in range(3):
                selected = indices[bucket == code]
                if len(selected):
                    buckets[code].append(selected)

        self._buckets = [
            np.concatenate(values) if values else np.empty(0, dtype=np.int64) for values in buckets
        ]
        if any(not len(values) for values in self._buckets):
            names = ("approach", "track", "retreat")
            missing = [
                name for name, values in zip(names, self._buckets, strict=True) if not len(values)
            ]
            raise ValueError(
                "phase-balanced sampling requires non-empty approach, track and retreat buckets; "
                f"missing: {', '.join(missing)}"
            )
        ratios = np.asarray(target_ratios, dtype=np.float64)
        if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
            raise ValueError("phase sampling ratios must be three positive values summing to 1")
        self._counts = self._allocate_counts(len(np.concatenate(self._buckets)), ratios)
        self._num_frames = sum(self._counts)
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @staticmethod
    def _allocate_counts(total: int, ratios: np.ndarray) -> tuple[int, int, int]:
        raw = ratios * total
        counts = np.floor(raw).astype(int)
        for index in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
            counts[index] += 1
        return tuple(int(value) for value in counts)

    @property
    def indices(self) -> list[int]:
        """返回每个 phase bucket 的原始 frame 索引，供诊断使用。"""
        return np.concatenate(self._buckets).tolist()

    def set_epoch(self, epoch: int) -> None:
        """设置下一个 epoch 的确定性采样状态。"""
        self._epoch = epoch

    def state_dict(self) -> dict[str, int]:
        """保存可恢复的 epoch 和样本偏移。"""
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """恢复 sampler 的 epoch 和样本偏移。"""
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        generator = self._epoch_generator(epoch)
        sampled = [
            bucket[torch.randint(len(bucket), (count,), generator=generator).numpy()]
            for bucket, count in zip(self._buckets, self._counts, strict=True)
        ]
        order = torch.randperm(self._num_frames, generator=generator).numpy()
        absolute = np.concatenate(sampled)[order]
        for index in absolute[start:]:
            value = int(index)
            yield (
                self._absolute_to_relative.get(value, value)
                if self._absolute_to_relative
                else value
            )

    def __len__(self) -> int:
        return self._num_frames


@contextmanager
def phase_balanced_sampler(
    config: PhaseSamplingConfig,
    dataset_root: str | Path | None,
    action_horizon: int,
    action_stride: int,
):
    """在官方 LeRobot 训练循环中临时替换默认 sampler。"""
    if not config.enabled:
        yield
        return

    from lerobot.scripts import lerobot_train

    index_path = phase_index_path(config, dataset_root)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"phase index not found: {index_path}; run scripts/build_phase_index.py first"
        )
    original = lerobot_train.EpisodeAwareSampler

    def factory(*args: Any, **kwargs: Any) -> PhaseBalancedSampler:
        return PhaseBalancedSampler(
            *args,
            phase_index_path=index_path,
            target_ratios=(config.approach_ratio, config.track_ratio, config.retreat_ratio),
            action_horizon=action_horizon,
            action_stride=action_stride,
            track_threshold=config.track_threshold,
            **kwargs,
        )

    lerobot_train.EpisodeAwareSampler = factory
    try:
        yield
    finally:
        lerobot_train.EpisodeAwareSampler = original


__all__ = ["PhaseBalancedSampler", "phase_balanced_sampler", "phase_index_path"]
