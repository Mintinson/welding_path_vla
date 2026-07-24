import json
from pathlib import Path

import numpy as np
import pytest

from welding_path_vla.config import EvaluationConfig
from welding_path_vla.evaluation.adapters import SAFETY_SIGNALS, trace_from_real_robot_log
from welding_path_vla.evaluation.evaluator import aggregate_reports, evaluate_trace
from welding_path_vla.evaluation.schema import (
    EvaluationTrace,
    InstructionAssessment,
    SeamReference,
    Termination,
)


def straight_trace() -> EvaluationTrace:
    timestamps = np.linspace(0, 10, 101)
    seam_points = np.column_stack((np.linspace(0, 1, 101), np.zeros((101, 2))))
    positions = seam_points.copy()
    positions[:, 1] = 0.001
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (101, 1))
    safety = {name: np.zeros(101, dtype=bool) for name in SAFETY_SIGNALS}
    return EvaluationTrace(
        timestamps,
        positions,
        quaternions,
        np.ones(101, dtype=bool),
        SeamReference(seam_points, quaternions, 0.1, "seam_a"),
        InstructionAssessment(True, True, True),
        safety,
        SAFETY_SIGNALS,
        Termination(True),
        10.0,
        timestamps,
        seam_points,
    )


def test_complete_accurate_trace_succeeds() -> None:
    report = evaluate_trace(straight_trace(), EvaluationConfig())
    assert report.success
    assert report.completion.pcr == pytest.approx(1)
    assert report.completion.direction_ratio == pytest.approx(1)
    assert report.tracking.cte_rmse_m == pytest.approx(0.001)
    assert report.tracking.cte_p95_m == pytest.approx(0.001)
    assert report.tracking.orientation_p95_deg == pytest.approx(0)
    assert report.tracking.speed_mape < 1e-10
    assert report.tracking.jerk_rms_m_s3 is None
    assert report.tracking.jerk_ratio is None
    assert report.conditions["smoothness"]


def test_missing_jerk_fails_when_smoothness_is_required() -> None:
    thresholds = EvaluationConfig(require_smoothness_for_success=True)
    report = evaluate_trace(straight_trace(), thresholds)
    assert not report.conditions["smoothness"]
    assert not report.success


def test_reverse_motion_fails_completion() -> None:
    trace = straight_trace()
    reverse = EvaluationTrace(
        trace.timestamps,
        trace.tcp_positions[::-1],
        trace.tcp_quaternions_wxyz,
        trace.track_mask,
        trace.seam,
        trace.instruction,
        trace.safety_signals,
        trace.required_safety_signals,
        trace.termination,
        trace.sample_rate_hz,
        trace.expert_timestamps,
        trace.expert_positions,
    )
    report = evaluate_trace(reverse, EvaluationConfig())
    assert report.completion.pcr == pytest.approx(0)
    assert report.completion.direction_ratio == pytest.approx(0)
    assert not report.conditions["completion"]
    assert not report.success


def test_missing_safety_signal_fails_closed() -> None:
    trace = straight_trace()
    signals = dict(trace.safety_signals)
    del signals["collision"]
    incomplete = EvaluationTrace(
        trace.timestamps,
        trace.tcp_positions,
        trace.tcp_quaternions_wxyz,
        trace.track_mask,
        trace.seam,
        trace.instruction,
        signals,
        trace.required_safety_signals,
        trace.termination,
        trace.sample_rate_hz,
        trace.expert_timestamps,
        trace.expert_positions,
    )
    report = evaluate_trace(incomplete, EvaluationConfig())
    assert report.safety.missing_signals == ("collision",)
    assert not report.conditions["safety"]


def test_real_robot_log_adapter_and_summary(tmp_path: Path) -> None:
    trace = straight_trace()
    control = {
        "timestamp": trace.timestamps,
        "tcp_position": trace.tcp_positions,
        "tcp_quaternion_wxyz": trace.tcp_quaternions_wxyz,
        "track_mask": trace.track_mask,
        "expert_timestamp": trace.expert_timestamps,
        "expert_position": trace.expert_positions,
        **trace.safety_signals,
    }
    np.savez_compressed(tmp_path / "control.npz", **control)
    task = {
        "seam": {
            "seam_id": "seam_a",
            "points_world": trace.seam.points.tolist(),
            "quaternions_wxyz": trace.seam.quaternions_wxyz.tolist(),
            "desired_speed_mps": 0.1,
        },
        "instruction_assessment": {
            "seam_correct": True,
            "direction_correct": True,
            "sequence_correct": True,
        },
        "required_safety_signals": list(SAFETY_SIGNALS),
        "termination": {"completed": True},
    }
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    report = evaluate_trace(trace_from_real_robot_log(tmp_path), EvaluationConfig())
    summary = aggregate_reports([report, report])
    assert report.success
    assert summary.esr == 1
    assert summary.icr == 1
    assert summary.episodes == 2
