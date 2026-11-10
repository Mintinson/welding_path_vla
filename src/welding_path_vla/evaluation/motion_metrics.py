"""轨迹精度、速度和光滑性指标。"""

from __future__ import annotations

import numpy as np

from welding_path_vla.core.geometry import rotation_error
from welding_path_vla.evaluation.schema import CompletionReport, TrackingReport
from welding_path_vla.evaluation.seam_geometry import (
    SeamProjection,
    interpolate_quaternions,
    interpolate_speed,
    project_to_seam,
)


def direction_ratio(travel: np.ndarray) -> float:
    """计算有效弧长增量中正向运动的距离占比。"""
    distance = float(np.sum(np.abs(travel)))
    return float(np.sum(np.clip(travel, 0, None)) / distance) if distance else 0.0


def closed_seam_direction_ratio(
    positions: np.ndarray,
    seam_points: np.ndarray,
    tracking: np.ndarray,
    max_step_m: float,
) -> float:
    """计算闭合焊缝的方向率，保留跨接缝时真实的运动方向。

    Args:
        positions: 与 tracking 对齐的 TCP 世界坐标。
        seam_points: 按任务方向排序的一圈焊缝参考点。
        tracking: 有效跟踪帧掩码。
        max_step_m: 相邻帧允许计入方向统计的最大位移，单位为米。
    """
    # 专家跟踪帧通常从第一个运动步开始采样，末点到首点也属于焊缝。
    ring = np.concatenate((seam_points, seam_points[:1]))
    projection = project_to_seam(positions, ring)
    period = projection.total_length
    delta = (np.diff(projection.arc_lengths) + period / 2) % period - period / 2
    # 不跨越离缝断段，也不让异常 TCP 跳步经取模后计入方向统计。
    tcp_step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    valid = tracking[:-1] & tracking[1:] & (np.maximum(np.abs(delta), tcp_step) <= max_step_m)
    return direction_ratio(delta[valid])


def completion_metrics(projection: SeamProjection) -> CompletionReport:
    """计算有界 PCR (Path Completion Ratio) 和正确方向运动占比。"""
    pcr = np.clip(
        (np.max(projection.arc_lengths, initial=0.0) - projection.arc_lengths[0])
        / projection.total_length,
        0,
        1,
    )
    return CompletionReport(
        pcr=float(pcr), direction_ratio=direction_ratio(np.diff(projection.arc_lengths))
    )


def derivative(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """对非等间隔时序执行数值微分。"""
    edge_order = 2 if len(timestamps) >= 3 else 1
    return np.gradient(values, timestamps, axis=0, edge_order=edge_order)


def jerk_rms(positions: np.ndarray, timestamps: np.ndarray) -> float | None:
    """计算平移 jerk 向量模长的均方根。"""
    if len(timestamps) < 4:
        return None
    jerk = derivative(derivative(derivative(positions, timestamps), timestamps), timestamps)
    return float(np.sqrt(np.mean(np.sum(jerk**2, axis=1))))


def tracking_metrics(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    seam_points: np.ndarray,
    seam_quaternions_wxyz: np.ndarray,
    desired_speed_mps: float | np.ndarray,
    projection: SeamProjection,
    expert_timestamps: np.ndarray | None = None,
    expert_positions: np.ndarray | None = None,
    compute_jerk: bool = True,
    jerk_reference_floor_m_s3: float = 0.001,
) -> TrackingReport:
    """计算 CTE (Cross-Track Error)、姿态、速度和 jerk 指标。"""
    # CTE
    displacement = positions - projection.points
    perpendicular = (
        displacement
        - np.sum(displacement * projection.tangents, axis=1)[:, None] * projection.tangents
    )
    cte = np.linalg.norm(perpendicular, axis=1)

    # 姿态误差
    desired_quaternions = interpolate_quaternions(
        seam_quaternions_wxyz,
        projection.segment_indices,
        projection.segment_fractions,
    )
    orientation = np.degrees(
        [
            np.linalg.norm(rotation_error(target, actual))
            for target, actual in zip(desired_quaternions, quaternions_wxyz, strict=True)
        ]
    )

    # Speed MAPE
    velocity = derivative(positions, timestamps)
    # 沿焊缝方向的有效速率
    parallel_speed = np.sum(velocity * projection.tangents, axis=1)
    desired_speed = interpolate_speed(desired_speed_mps, projection.arc_lengths, seam_points)
    speed_mape = np.mean(np.abs(parallel_speed - desired_speed) / (desired_speed + 1e-6))

    # 运动平滑度（Jerk 加加速度与专家对比）
    actual_jerk = jerk_rms(positions, timestamps) if compute_jerk else None
    expert_jerk = (
        jerk_rms(expert_positions, expert_timestamps)
        if compute_jerk and expert_positions is not None and expert_timestamps is not None
        else None
    )
    ratio = (
        actual_jerk / expert_jerk
        if actual_jerk is not None
        and expert_jerk is not None
        and expert_jerk >= jerk_reference_floor_m_s3
        else None
    )
    return TrackingReport(
        cte_rmse_m=float(np.sqrt(np.mean(cte**2))),
        cte_p95_m=float(np.percentile(cte, 95)),
        cte_max_m=float(np.max(cte)),
        orientation_p95_deg=float(np.percentile(orientation, 95)),
        orientation_max_deg=float(np.max(orientation)),
        speed_mape=float(speed_mape),
        jerk_rms_m_s3=actual_jerk,
        expert_jerk_rms_m_s3=expert_jerk,
        jerk_ratio=ratio,
    )
