from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RealRobotEvaluation:
    episode_path: Path
    task_id: str
    operator_approved: bool


class RealRobotEvaluator(Protocol):
    """Boundary for supervised, safety-gated evaluation on the physical cell."""

    def evaluate(self, task_id: str) -> RealRobotEvaluation: ...


__all__ = ["RealRobotEvaluation", "RealRobotEvaluator"]
