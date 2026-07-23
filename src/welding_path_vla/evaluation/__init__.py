"""Offline and real-robot evaluation interfaces."""

from welding_path_vla.evaluation.collision_metrics import CollisionReport
from welding_path_vla.evaluation.trajectory_metrics import EpisodeReport, validate_episode

__all__ = ["CollisionReport", "EpisodeReport", "validate_episode"]
