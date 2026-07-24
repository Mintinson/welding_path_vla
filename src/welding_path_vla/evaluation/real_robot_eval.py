from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from welding_path_vla.core.config import EvaluationConfig
from welding_path_vla.evaluation.adapters import trace_from_real_robot_log
from welding_path_vla.evaluation.evaluator import evaluate_trace
from welding_path_vla.evaluation.schema import EpisodeEvaluation


@dataclass(frozen=True, slots=True)
class RealRobotEvaluation:
    episode_path: Path
    task_id: str
    report: EpisodeEvaluation


class RealRobotEvaluator(Protocol):
    """Boundary for supervised, safety-gated evaluation on the physical cell."""

    def evaluate(self, task_id: str) -> RealRobotEvaluation: ...


def evaluate_real_robot_episode(
    episode_path: str | Path, thresholds: EvaluationConfig
) -> RealRobotEvaluation:
    """离线评估一条经过安全监督的真机日志。"""
    path = Path(episode_path)
    trace = trace_from_real_robot_log(path)
    return RealRobotEvaluation(path, trace.seam.seam_id, evaluate_trace(trace, thresholds))


__all__ = ["RealRobotEvaluation", "RealRobotEvaluator", "evaluate_real_robot_episode"]
