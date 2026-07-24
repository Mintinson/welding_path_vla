from __future__ import annotations

import numpy as np

from welding_path_vla.core.domain import RobotState
from welding_path_vla.robot.elfin5pro_driver import Elfin5ProDriver
from welding_path_vla.robot.safety_monitor import SafetyMonitor


class RealtimeController:
    """Small safety gate around one physical-robot control tick."""

    def __init__(self, driver: Elfin5ProDriver, safety: SafetyMonitor) -> None:
        self.driver = driver
        self.safety = safety

    def command_joint_position(self, command_rad: np.ndarray) -> RobotState:
        state = self.driver.read_state()
        self.safety.validate_state(state)
        self.safety.validate_joint_command(command_rad)
        self.driver.command_joint_position(np.asarray(command_rad, dtype=np.float64))
        return state

    def stop(self) -> None:
        self.driver.stop()


__all__ = ["RealtimeController"]
