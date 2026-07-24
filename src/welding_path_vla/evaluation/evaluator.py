"""组合任务合规、轨迹质量与安全条件。"""

from __future__ import annotations

from collections import Counter

import numpy as np

from welding_path_vla.core.config import EvaluationConfig
from welding_path_vla.evaluation.motion_metrics import completion_metrics, tracking_metrics
from welding_path_vla.evaluation.schema import (
    EpisodeEvaluation,
    EvaluationSummary,
    EvaluationTrace,
    SafetyReport,
)
from welding_path_vla.evaluation.seam_geometry import project_to_seam


def safety_report(trace: EvaluationTrace) -> SafetyReport:
    """汇总安全信号；缺少必需信号时按失败处理。"""
    missing = tuple(
        name for name in trace.required_safety_signals if name not in trace.safety_signals
    )
    violations = tuple(
        name
        for name in trace.required_safety_signals
        if name in trace.safety_signals and np.any(trace.safety_signals[name])
    )
    return SafetyReport(not missing and not violations, violations, missing)


def evaluate_trace(trace: EvaluationTrace, thresholds: EvaluationConfig) -> EpisodeEvaluation:
    """评估一条仿真或真机轨迹。"""
    mask = np.asarray(trace.track_mask, dtype=bool)
    timestamps = trace.timestamps[mask]
    positions = trace.tcp_positions[mask]
    quaternions = trace.tcp_quaternions_wxyz[mask]
    projection = project_to_seam(positions, trace.seam.points)
    completion = completion_metrics(projection)
    tracking = tracking_metrics(
        timestamps,
        positions,
        quaternions,
        trace.seam.points,
        trace.seam.quaternions_wxyz,
        trace.seam.desired_speed_mps,
        projection,
        trace.expert_timestamps,
        trace.expert_positions,
        trace.sample_rate_hz >= thresholds.jerk_min_sample_rate_hz,
        thresholds.jerk_reference_floor_m_s3,
    )
    safety = safety_report(trace)
    smoothness = tracking.jerk_ratio is not None and (
        tracking.jerk_ratio <= thresholds.jerk_ratio_max
    )
    if tracking.jerk_ratio is None and not thresholds.require_smoothness_for_success:
        smoothness = True
    conditions = {
        "instruction": trace.instruction.compliant,
        "completion": completion.pcr >= thresholds.pcr_min
        and completion.direction_ratio >= thresholds.direction_ratio_min,
        "position": tracking.cte_rmse_m <= thresholds.cte_rmse_m
        and tracking.cte_p95_m <= thresholds.cte_p95_m
        and tracking.cte_max_m <= thresholds.cte_max_m,
        "orientation": tracking.orientation_p95_deg <= thresholds.orientation_p95_deg,
        "speed": tracking.speed_mape <= thresholds.speed_mape_max,
        "smoothness": smoothness,
        "safety": safety.safe,
        "termination": trace.termination.normal,
    }
    required = (
        "instruction",
        "completion",
        "position",
        "orientation",
        "speed",
        "safety",
        "termination",
    )
    if thresholds.require_smoothness_for_success:
        required += ("smoothness",)
    return EpisodeEvaluation(
        all(conditions[name] for name in required),
        trace.instruction,
        completion,
        tracking,
        safety,
        trace.termination.normal,
        conditions,
        trace.sample_rate_hz,
    )


def aggregate_reports(reports: list[EpisodeEvaluation]) -> EvaluationSummary:
    """聚合主表 ESR/ICR 及连续指标均值。"""
    count = len(reports)
    jerk = [
        report.tracking.jerk_ratio for report in reports if report.tracking.jerk_ratio is not None
    ]
    failures = Counter(
        name for report in reports for name, passed in report.conditions.items() if not passed
    )
    return EvaluationSummary(
        count,
        sum(report.success for report in reports),
        float(np.mean([report.success for report in reports])),
        float(np.mean([report.instruction_compliant for report in reports])),
        float(np.mean([report.completion.pcr for report in reports])),
        float(np.mean([report.tracking.cte_rmse_m for report in reports])),
        float(np.mean([report.tracking.cte_p95_m for report in reports])),
        float(np.mean([report.tracking.orientation_p95_deg for report in reports])),
        float(np.mean([report.tracking.speed_mape for report in reports])),
        float(np.mean(jerk)) if jerk else None,
        dict(failures),
    )
