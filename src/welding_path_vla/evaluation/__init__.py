"""Offline and real-robot evaluation interfaces."""

from welding_path_vla.evaluation.collision_metrics import CollisionReport
from welding_path_vla.evaluation.evaluator import aggregate_reports, evaluate_trace
from welding_path_vla.evaluation.schema import EpisodeEvaluation, EvaluationTrace
from welding_path_vla.evaluation.trajectory_metrics import EpisodeReport, validate_episode

__all__ = [
    "CollisionReport",
    "EpisodeEvaluation",
    "EpisodeReport",
    "EvaluationTrace",
    "aggregate_reports",
    "evaluate_trace",
    "validate_episode",
]
