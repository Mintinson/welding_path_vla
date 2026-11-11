from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from welding_path_vla.core.config import TaskConfig


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp: float
    images: dict[str, np.ndarray]
    state: np.ndarray
    instruction: str
    direction: str | None = None
    welding_speed_mps: float | None = None
    work_angle_deg: float | None = None
    travel_angle_deg: float | None = None
    tool_roll_deg: float | None = None

    @classmethod
    def from_state(
        cls,
        timestamp: float,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: TaskConfig,
        instruction: str | None = None,
    ) -> Observation:
        """用统一状态和任务参数构造仿真或实机的策略输入。"""
        return cls(
            timestamp,
            images,
            state,
            instruction or task.instruction,
            task.direction,
            task.speed_mps,
            task.work_angle_deg,
            task.travel_angle_deg,
            task.tool_roll_deg,
        )


class Policy(Protocol):
    """Map one synchronized observation to an action chunk in the dataset convention."""

    def reset(self) -> None: ...

    def select_action(self, observation: Observation) -> np.ndarray: ...


__all__ = ["Observation", "Policy"]
