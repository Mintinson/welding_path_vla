from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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


class Policy(Protocol):
    """Map one synchronized observation to an action chunk in the dataset convention."""

    def reset(self) -> None: ...

    def select_action(self, observation: Observation) -> np.ndarray: ...


__all__ = ["Observation", "Policy"]
