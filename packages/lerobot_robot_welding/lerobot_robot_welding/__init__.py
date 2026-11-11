"""供 LeRobot 自动发现的焊接采集插件。"""

from .adapter import (
    ElfinPassive,
    ElfinPassiveConfig,
    ElfinWelding,
    ElfinWeldingConfig,
)
from .black_camera import BlackCamera, BlackCameraConfig

__all__ = [
    "BlackCamera",
    "BlackCameraConfig",
    "ElfinPassive",
    "ElfinPassiveConfig",
    "ElfinWelding",
    "ElfinWeldingConfig",
]
