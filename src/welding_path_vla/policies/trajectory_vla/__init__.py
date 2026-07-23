from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrajectoryVlaPolicyConfig:
    waypoint_horizon: int = 20
    action_frame: str = "seam"


__all__ = ["TrajectoryVlaPolicyConfig"]
