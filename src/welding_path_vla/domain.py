from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


class Phase(StrEnum):
    APPROACH = "approach"
    TRACK = "track"
    RETREAT = "retreat"


class EpisodeStatus(StrEnum):
    VALID_SUCCESS = "valid_success"
    VALID_RECOVERY = "valid_recovery"
    INVALID_PLANNING = "invalid_planning"
    INVALID_SIMULATION = "invalid_simulation"
    COLLISION_FAILURE = "collision_failure"


@dataclass(frozen=True, slots=True)
class Pose:
    position: FloatArray
    quaternion_wxyz: FloatArray


@dataclass(frozen=True, slots=True)
class Seam:
    seam_id: str
    start_local: FloatArray
    end_local: FloatArray
    normal_local: FloatArray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end_local - self.start_local))

    @property
    def tangent_local(self) -> FloatArray:
        return (self.end_local - self.start_local) / self.length


@dataclass(frozen=True, slots=True)
class WeldingTask:
    instruction: str
    seam_id: str
    direction: str
    speed_mps: float
    work_angle_rad: float
    travel_angle_rad: float


@dataclass(frozen=True, slots=True)
class RobotState:
    joint_position: FloatArray
    joint_velocity: FloatArray
    tcp: Pose


@dataclass(frozen=True, slots=True)
class CommandAction:
    delta_pose_seam: FloatArray
    delta_pose_base: FloatArray
    delta_pose_world: FloatArray
    joint_position: FloatArray
