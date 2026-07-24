"""论文级轨迹评估的数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SeamReference:
    """按期望执行方向排列的离散焊缝参考。"""

    points: np.ndarray
    quaternions_wxyz: np.ndarray
    desired_speed_mps: float | np.ndarray
    seam_id: str


@dataclass(frozen=True, slots=True)
class InstructionAssessment:
    """任务理解模块或人工标注给出的指令合规结果。"""

    seam_correct: bool
    direction_correct: bool
    sequence_correct: bool = True

    @property
    def compliant(self) -> bool:
        return self.seam_correct and self.direction_correct and self.sequence_correct


@dataclass(frozen=True, slots=True)
class Termination:
    """一次执行的终止原因。"""

    completed: bool
    timed_out: bool = False
    operator_stopped: bool = False

    @property
    def normal(self) -> bool:
        return self.completed and not self.timed_out and not self.operator_stopped


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    """与仿真器和真机驱动解耦的评估输入。"""

    timestamps: np.ndarray
    tcp_positions: np.ndarray
    tcp_quaternions_wxyz: np.ndarray
    track_mask: np.ndarray
    seam: SeamReference
    instruction: InstructionAssessment
    safety_signals: dict[str, np.ndarray]
    required_safety_signals: tuple[str, ...]
    termination: Termination
    sample_rate_hz: float
    expert_timestamps: np.ndarray | None = None
    expert_positions: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CompletionReport:
    pcr: float
    direction_ratio: float


@dataclass(frozen=True, slots=True)
class TrackingReport:
    cte_rmse_m: float
    cte_p95_m: float
    cte_max_m: float
    orientation_p95_deg: float
    orientation_max_deg: float
    speed_mape: float
    jerk_rms_m_s3: float | None
    expert_jerk_rms_m_s3: float | None
    jerk_ratio: float | None


@dataclass(frozen=True, slots=True)
class SafetyReport:
    safe: bool
    violations: tuple[str, ...]
    missing_signals: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    success: bool
    instruction: InstructionAssessment
    completion: CompletionReport
    tracking: TrackingReport
    safety: SafetyReport
    termination_normal: bool
    conditions: dict[str, bool]
    sample_rate_hz: float

    @property
    def instruction_compliant(self) -> bool:
        return self.instruction.compliant

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episodes: int
    successes: int
    esr: float
    icr: float
    pcr_mean: float
    cte_rmse_mean_m: float
    cte_p95_mean_m: float
    orientation_p95_mean_deg: float
    speed_mape_mean: float
    jerk_ratio_mean: float | None
    condition_failures: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
