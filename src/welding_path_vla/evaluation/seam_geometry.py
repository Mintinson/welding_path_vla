"""折线焊缝的投影、切向和姿态插值。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeamProjection:
    points: np.ndarray
    tangents: np.ndarray
    arc_lengths: np.ndarray
    segment_indices: np.ndarray
    segment_fractions: np.ndarray
    total_length: float


def project_to_seam(positions: np.ndarray, seam_points: np.ndarray, eps=1e-12) -> SeamProjection:
    """把 TCP 位置投影到有序折线焊缝。

    Args:
        positions: TCP 世界坐标，形状为 `(N, 3)`。
        seam_points: 按期望方向排列的焊缝点，形状为 `(M, 3)`。

    Returns:
        最近投影点、局部切向、弧长坐标和所属线段。
    """
    starts = seam_points[:-1]
    vectors = np.diff(seam_points, axis=0)  # (M-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)  # (M-1,)

    valid = lengths > eps
    tangents = np.zeros_like(vectors)
    tangents[valid] = vectors[valid] / lengths[valid, None]

    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))

    diff = positions[:, None, :] - starts[None, :, :]  # (N, M-1, 3)
    denom = lengths**2 + eps
    alpha = np.clip(np.sum(diff * vectors[None, :, :], axis=2) / denom, 0, 1)
    # 候选投影点 (N, M, 3)
    candidates = starts[None, :, :] + alpha[:, :, None] * vectors[None, :, :]
    # (N, M-1) 每个位置到每条线段的距离平方
    dist_sq = np.sum((positions[:, None, :] - candidates) ** 2, axis=2)
    # 最近线段索引
    indices = np.argmin(dist_sq, axis=1)  # (N,)

    arange = np.arange(len(positions))
    alpha_best = alpha[arange, indices]  # (N,)
    lengths_best = lengths[indices]  # (N,)

    projected = candidates[arange, indices]  # (N, 3)
    arc = cumulative[indices] + alpha_best * lengths_best  # (N)
    tangents_best = tangents[indices]  # (N, 3)

    return SeamProjection(
        points=projected,
        tangents=tangents_best,
        arc_lengths=arc,
        segment_indices=indices,
        segment_fractions=alpha_best,
        total_length=float(cumulative[-1]),
    )


def interpolate_quaternions(
    quaternions_wxyz: np.ndarray, indices: np.ndarray, fractions: np.ndarray
) -> np.ndarray:
    """以归一化线性插值获得投影点的期望姿态。"""
    start = quaternions_wxyz[indices]
    end = quaternions_wxyz[indices + 1].copy()
    # 由于四元数q和-q代表相同的旋转，因此在进行插值之前，
    # 需要确保我们选择的两个四元数位于四维超球面的同一半弧上
    end[np.sum(start * end, axis=1) < 0] *= -1
    # linear interpolation (LERP) and normalization
    values = (1 - fractions[:, None]) * start + fractions[:, None] * end
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def interpolate_speed(
    desired_speed_mps: float | np.ndarray,
    arc_lengths: np.ndarray,
    seam_points: np.ndarray,
) -> np.ndarray:
    """把标量或逐焊缝点速度插值到执行样本。

    Args:
        desired_speed_mps (float | np.ndarray): 每个焊缝点处定义的速度, 标量则代表匀速
        arc_lengths (np.ndarray): 执行机构实际采样位置
        seam_points (np.ndarray): 焊缝几何路径

    Returns:
        np.ndarray: 沿着弧长参数线性插值到采样点的速度值
    """
    if isinstance(desired_speed_mps, np.ndarray):
        segments = np.linalg.norm(np.diff(seam_points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
        return np.interp(arc_lengths, cumulative, desired_speed_mps)
    return np.full(len(arc_lengths), desired_speed_mps)
