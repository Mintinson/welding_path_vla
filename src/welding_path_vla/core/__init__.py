"""跨数据、仿真、机器人和策略模块共享的核心类型。"""

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import CommandAction, Pose, RobotState

__all__ = ["AppConfig", "CommandAction", "Pose", "RobotState"]
