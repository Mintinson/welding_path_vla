from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from welding_path_vla.domain import RobotState


@runtime_checkable
class Elfin5ProDriver(Protocol):
    """SI-unit adapter boundary; implementations expose TCP poses in the project world frame."""

    def connect(self) -> None: ...

    def read_state(self) -> RobotState: ...

    def command_joint_position(self, joint_position_rad: np.ndarray) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class CameraSource(Protocol):
    """Timestamped RGB source shared by global and wrist cameras."""

    def capture(self) -> tuple[float, np.ndarray]: ...

    def close(self) -> None: ...


__all__ = ["CameraSource", "Elfin5ProDriver"]
