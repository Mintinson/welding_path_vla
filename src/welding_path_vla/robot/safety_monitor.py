from __future__ import annotations

import numpy as np

from welding_path_vla.core.config import SafetyConfig
from welding_path_vla.core.domain import RobotState


class SafetyViolation(RuntimeError):
    pass


class SafetyMonitor:
    def __init__(self, config: SafetyConfig, joint_ranges_rad: np.ndarray) -> None:
        self.config = config
        self.joint_ranges = np.asarray(joint_ranges_rad, dtype=np.float64)

    def validate_state(self, state: RobotState) -> None:
        if not np.all(np.isfinite(state.joint_position)) or not np.all(
            np.isfinite(state.joint_velocity)
        ):
            raise SafetyViolation("robot state contains non-finite values")
        if np.any(np.abs(state.joint_velocity) > self.config.joint_velocity_limit_rad_s):
            raise SafetyViolation("joint velocity limit exceeded")

    def validate_joint_command(self, command_rad: np.ndarray) -> None:
        command = np.asarray(command_rad, dtype=np.float64)
        lower = self.joint_ranges[:, 0] + self.config.joint_position_margin_rad
        upper = self.joint_ranges[:, 1] - self.config.joint_position_margin_rad
        if command.shape != lower.shape or not np.all(np.isfinite(command)):
            raise SafetyViolation("invalid joint command")
        if np.any(command < lower) or np.any(command > upper):
            raise SafetyViolation("joint command exceeds configured safe range")


__all__ = ["SafetyMonitor", "SafetyViolation"]
