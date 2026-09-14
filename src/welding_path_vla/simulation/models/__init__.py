"""robosuite 使用的机器人、场景和工件模型。"""

from welding_path_vla.simulation.models.arena import WeldingArena
from welding_path_vla.simulation.models.robot import Elfin5ProRobotModel
from welding_path_vla.simulation.models.workpiece import (
    SEAM_PENDING_RGBA,
    SEAM_WELDED_RGBA,
    WORKPIECE_MATERIAL,
    WORKPIECE_TEXTURE,
    WorkpieceObject,
)

__all__ = [
    "SEAM_PENDING_RGBA",
    "SEAM_WELDED_RGBA",
    "WORKPIECE_MATERIAL",
    "WORKPIECE_TEXTURE",
    "Elfin5ProRobotModel",
    "WeldingArena",
    "WorkpieceObject",
]
