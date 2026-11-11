"""闭环焊缝路径执行的主结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from welding_path_vla.core.config import EvaluationConfig
from welding_path_vla.evaluation.motion_metrics import closed_seam_direction_ratio, direction_ratio
from welding_path_vla.evaluation.seam_geometry import project_to_seam


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    """成功、覆盖、到达和 tracking 阶段误差指标。"""

    success: bool
    coverage_ratio: float
    direction_ratio: float
    reached_start: bool
    tracking_frames: int
    cte_p95_m: float | None

    def as_dict(self) -> dict[str, bool | float | int | None]:
        """返回可直接写入 episode JSON 的主结果。"""
        return asdict(self)


def evaluate_rollout(
    positions: np.ndarray,
    seam_points: np.ndarray,
    termination_reason: str,
    config: EvaluationConfig,
    track_mask: np.ndarray | None = None,
    *,
    seam_is_closed: bool | None = None,
) -> RolloutMetrics:
    """评价有向焊缝覆盖，并只在 tracking 阶段统计 CTE。

    Args:
        positions: 每个策略步后的 TCP 世界坐标，形状为 ``(N, 3)``。
        seam_points: 按任务要求方向排列的焊缝参考点。
        termination_reason: rollout 的终止原因；碰撞与硬安全停止不能成功。
        config: 全模型共用的评价阈值。
        track_mask: 与 ``positions`` 对齐的 tracking 阶段掩码；未提供时由
            ``tracking_band_m`` 根据焊缝距离推导。
        seam_is_closed: 焊缝是否闭合，仅影响方向计算。未提供时，
            只将首尾坐标在浮点误差内重合的参考折线视为闭环。

    Returns:
        三项连续轨迹指标和一次成功判定。非空轨迹始终有数值结果。
    """
    if len(positions) == 0:
        return RolloutMetrics(False, 0.0, 0.0, False, 0, None)

    # 将 TCP 位置投影到焊缝折线上
    # projection 对象包含：
    #   arc_lengths: 每个点沿焊缝的弧长坐标
    #   total_length: 焊缝总长度
    #   points: 每个点在焊缝上的最近投影点
    projection = project_to_seam(np.asarray(positions), seam_points)
    arc = projection.arc_lengths  # 每个点投影到焊缝后的弧长
    length = projection.total_length  # 焊缝总长度

    # 每个原始点到其投影点的欧氏距离（即横向跟踪误差 CTE）
    distance = np.linalg.norm(positions - projection.points, axis=1)  # 点到焊缝的垂直距离

    if track_mask is None:
        tracking = distance <= config.tracking_band_m
    else:
        tracking = np.asarray(track_mask, dtype=bool)
        if tracking.shape != distance.shape:
            raise ValueError("track_mask must have the same length as positions")

    # 判断每个帧是否“足够靠近”焊缝（用于覆盖率计算）
    near = distance <= config.coverage_tolerance_m  # 表示哪些帧足够靠近焊缝（用于覆盖率计算）

    # 寻找轨迹中第一次进入焊缝起点附近（前 5% 弧长）且靠近焊缝的帧
    start = np.flatnonzero(near & (arc <= 0.05 * length))
    first = int(start[0]) if len(start) else None
    # 寻找从起点之后，进入焊缝终点附近（后 5% 弧长）且靠近焊缝的帧
    end = (
        np.flatnonzero(near[first:] & (arc[first:] >= 0.95 * length))
        if first is not None
        else np.empty(0, dtype=int)
    )

    # ---------- 计算有效覆盖率 ----------
    # 将焊缝均匀分成 coverage_bins 个区间（bin），标记哪些区间被贴近覆盖
    covered = np.zeros(config.coverage_bins, dtype=bool)
    for frame in range(len(positions)):
        if not near[frame]:
            continue
        progress = arc[frame]  # 当前帧在焊缝上的弧长位置
        # 折线投影在线段端点有微小浮点偏差，避免把边界样本落入前一格。

        # 计算当前帧落在哪个 bin。
        index = min(int(progress / length * config.coverage_bins + 1e-4), config.coverage_bins - 1)
        covered[index] = True

        # 如果前一帧也靠近焊缝，并且两帧之间的弧长步长和 TCP 位移都足够小，
        # 则认为这是一段连续有效的跟踪运动，将跨越的所有 bin 都标记为覆盖。
        # 这可以防止因采样率低而漏掉中间的 bin。
        if frame > 0 and near[frame - 1]:
            arc_step = abs(progress - arc[frame - 1])
            tcp_step = float(np.linalg.norm(positions[frame] - positions[frame - 1]))
            if (
                arc_step <= config.coverage_max_step_m + 1e-9
                and tcp_step <= config.coverage_max_step_m + 1e-9
            ):
                previous_bin = min(
                    int(arc[frame - 1] / length * config.coverage_bins + 1e-4),
                    config.coverage_bins - 1,
                )
                low, high = sorted((previous_bin, index))
                covered[low : high + 1] = True

    coverage = float(np.mean(covered))  # 已覆盖 bin 的比例

    # 覆盖率和起终点沿用原始折线，仅方向率使用闭环弧长。
    if seam_is_closed is None:
        seam_is_closed = bool(np.allclose(seam_points[0], seam_points[-1], rtol=0, atol=1e-9))
    if seam_is_closed:
        direction = closed_seam_direction_ratio(
            positions, seam_points, tracking, config.coverage_max_step_m
        )
    else:
        direction = direction_ratio(np.diff(arc)[tracking[:-1] & tracking[1:]])

    # ---------- 计算 tracking 阶段 CTE 的 95 分位数 ----------
    # approach 和 retreat 只用于判断是否到达与覆盖，不混入焊缝跟踪误差。
    tracking_distance = distance[tracking]
    cte_p95 = (
        float(np.percentile(tracking_distance, 95))
        if tracking_distance.size
        else None
    )

    # ---------- 硬失败判定 ----------
    # 碰撞、动作违规、安全违规等直接视为失败
    hard_failure = termination_reason in {
        "collision",
        "action_violation",
    } or termination_reason.startswith("safety_violation")

    # ---------- 成功判定 ----------
    # 需要同时满足：找到起点、找到终点、覆盖率达标、方向率达标、无硬失败
    success = bool(
        first is not None
        and len(end)
        and coverage >= config.coverage_ratio_min
        and direction >= config.direction_ratio_min
        and not hard_failure
    )

    return RolloutMetrics(
        success,
        coverage,
        direction,
        first is not None,
        int(tracking_distance.size),
        cte_p95,
    )
