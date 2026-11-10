"""闭环主指标的覆盖、方向和失败口径回归测试。"""

import numpy as np
import pytest

from welding_path_vla.core.config import EvaluationConfig
from welding_path_vla.evaluation.rollout_metrics import evaluate_rollout


def straight_path() -> tuple[np.ndarray, np.ndarray]:
    """返回 1 m 焊缝与轻微偏移的完整正向执行轨迹。"""
    x = np.linspace(0, 1, 201)
    seam = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    execution = seam.copy()
    execution[:, 1] = 0.001
    return execution, seam


def test_complete_path_reports_tracking_metrics_and_arrival() -> None:
    execution, seam = straight_path()
    report = evaluate_rollout(execution, seam, "completed", EvaluationConfig())
    assert report.success
    assert report.coverage_ratio == pytest.approx(1)
    assert report.direction_ratio == pytest.approx(1)
    assert report.cte_p95_m == pytest.approx(0.001)
    assert report.reached_start
    assert report.tracking_frames == len(execution)
    assert set(report.as_dict()) == {
        "success",
        "coverage_ratio",
        "direction_ratio",
        "reached_start",
        "tracking_frames",
        "cte_p95_m",
    }


def test_jump_to_endpoint_does_not_count_as_full_coverage() -> None:
    execution, seam = straight_path()
    report = evaluate_rollout(execution[[0, -1]], seam, "completed", EvaluationConfig())
    assert report.coverage_ratio < 0.1
    assert not report.success


def test_sparse_but_continuous_path_is_not_penalized_by_sampling() -> None:
    execution, seam = straight_path()
    sparse = execution[::3]
    report = evaluate_rollout(sparse, seam, "completed", EvaluationConfig())
    assert report.coverage_ratio > 0.95
    assert report.success


def test_reverse_and_no_entry_are_failures_with_complete_denominator() -> None:
    execution, seam = straight_path()
    reverse = evaluate_rollout(execution[::-1], seam, "timeout", EvaluationConfig())
    missed = evaluate_rollout(
        execution + np.array([0, 0.1, 0]), seam, "timeout", EvaluationConfig()
    )
    assert reverse.coverage_ratio == pytest.approx(1)
    assert reverse.direction_ratio == 0
    assert reverse.cte_p95_m == pytest.approx(0.001)
    assert reverse.reached_start
    assert not reverse.success
    assert missed.coverage_ratio == 0
    assert missed.cte_p95_m is None
    assert missed.tracking_frames == 0
    assert not missed.reached_start
    assert not missed.success


def test_partial_path_has_metrics_without_reaching_start() -> None:
    _, seam = straight_path()
    x = np.linspace(0.25, 0.75, 101)
    execution = np.column_stack((x, np.full_like(x, 0.002), np.zeros_like(x)))
    report = evaluate_rollout(execution, seam, "timeout", EvaluationConfig())
    assert 0.49 <= report.coverage_ratio <= 0.52
    assert report.direction_ratio == pytest.approx(1)
    assert report.cte_p95_m == pytest.approx(0.002)
    assert not report.success


def test_far_off_seam_frames_are_excluded_from_tracking_cte() -> None:
    execution, seam = straight_path()
    execution[50:150, 1] = 0.03
    report = evaluate_rollout(execution, seam, "completed", EvaluationConfig())
    assert report.cte_p95_m == pytest.approx(0.001)
    assert report.coverage_ratio < 0.9
    clean, _ = straight_path()
    collision = evaluate_rollout(clean, seam, "collision", EvaluationConfig())
    assert collision.coverage_ratio == pytest.approx(1)
    assert not collision.success


def test_relaxed_direction_threshold_allows_moderate_backtracking() -> None:
    _, seam = straight_path()
    x = np.concatenate((
        np.linspace(0, 0.5, 101),
        np.linspace(0.5, 0.3, 41)[1:],
        np.linspace(0.3, 1, 141)[1:],
    ))
    execution = np.column_stack((x, np.full_like(x, 0.001), np.zeros_like(x)))
    report = evaluate_rollout(execution, seam, "completed", EvaluationConfig())
    assert 0.85 <= report.direction_ratio < 0.90
    assert report.success


def circle_path(progress: np.ndarray) -> np.ndarray:
    """返回半径 50 mm 的圆形焊缝在指定进度处的点。"""
    angle = 2 * np.pi * progress
    return 0.05 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))


@pytest.mark.parametrize("sign", [1, -1])
def test_closed_seam_direction_accepts_forward_seam_crossing(sign: int) -> None:
    """沿任务方向走过圆环接缝不能被当成倒退一圈。"""
    seam = circle_path(sign * np.linspace(0, 1, 201))
    execution = circle_path(sign * np.linspace(0.002, 1.008, 201))
    report = evaluate_rollout(execution, seam, "completed", EvaluationConfig())

    assert report.coverage_ratio == pytest.approx(1)
    assert report.direction_ratio == pytest.approx(1)
    assert report.success


def test_closed_seam_direction_rejects_reverse_seam_crossing() -> None:
    """反向跨过圆环接缝不能被当成大幅正向推进。"""
    seam = circle_path(np.linspace(0, 1, 201))
    execution = circle_path(np.linspace(0.005, -0.005, 11))
    report = evaluate_rollout(execution, seam, "timeout", EvaluationConfig())

    assert report.direction_ratio == pytest.approx(0)


def test_closed_seam_direction_closes_reference_without_changing_coverage() -> None:
    """显式闭环拓扑应补齐专家参考缺少的首段，其他指标不变。"""
    seam = circle_path(np.linspace(0, 1, 201)[1:])
    execution = circle_path(np.linspace(0.002, 1.008, 201))
    config = EvaluationConfig()
    old = evaluate_rollout(execution, seam, "completed", config, seam_is_closed=False)
    corrected = evaluate_rollout(execution, seam, "completed", config, seam_is_closed=True)

    assert old.direction_ratio < 0.85
    assert corrected.direction_ratio == pytest.approx(1)
    assert corrected.success
    assert corrected.coverage_ratio == old.coverage_ratio
    assert corrected.reached_start == old.reached_start
    assert corrected.cte_p95_m == old.cte_p95_m


@pytest.mark.parametrize(
    ("progress", "tracking"),
    [([0.8, 0.1], None), ([0.99, 0.0, 0.01], [True, False, True])],
)
def test_closed_seam_direction_excludes_invalid_pairs(
    progress: list[float], tracking: list[bool] | None
) -> None:
    """异常跳步和离缝断段都不应计入正向弧长。"""
    seam = circle_path(np.linspace(0, 1, 201))
    mask = np.asarray(tracking) if tracking is not None else None
    report = evaluate_rollout(
        circle_path(np.asarray(progress)), seam, "timeout", EvaluationConfig(), mask
    )

    assert report.direction_ratio == 0
