"""供 LeRobot 自动发现的焊接采集与控制插件。"""

from .adapter import (
    ElfinPassive,
    ElfinPassiveConfig,
    ElfinWelding,
    ElfinWeldingConfig,
)
from .black_camera import BlackCamera, BlackCameraConfig
from .rollout import install_rollout_support

install_rollout_support()

__all__ = [
    "BlackCamera",
    "BlackCameraConfig",
    "ElfinPassive",
    "ElfinPassiveConfig",
    "ElfinWelding",
    "ElfinWeldingConfig",
]
