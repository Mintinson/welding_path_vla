"""Elfin5-Pro 的 robosuite RobotModel。"""

from __future__ import annotations

from importlib.resources import files

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import look_at_quaternion
from welding_path_vla.simulation.robosuite_compat import RobotModel, array_to_string


class Elfin5ProRobotModel(RobotModel):
    """封装机械臂、焊枪、TCP、腕部相机和位置执行器。

    项目使用自有的阻尼最小二乘 IK 与 120 Hz 位置控制，因此这里只负责
    MJCF 模型组织，不额外接入 robosuite 的通用控制器。
    """

    def __init__(self, config: AppConfig, idn: int = 0) -> None:
        """加载模型并写入底座及腕部相机配置。

        Args:
            config: 项目统一配置。
            idn: robosuite 模型实例编号。
        """
        self.config = config
        path = files("welding_path_vla").joinpath("assets", config.robot.model_asset)
        super().__init__(str(path), idn=idn)
        self.set_base_xpos(np.asarray(config.scene.robot_base_position_m))
        self.set_base_ori(np.radians([0, 0, config.scene.robot_base_yaw_deg]))
        camera = self.worldbody.find(f".//camera[@name='{config.camera.wrist_name}']")
        assert camera is not None
        position = np.asarray(config.camera.wrist_position_link6_m)
        camera.set("pos", array_to_string(position))
        camera.set(
            "quat",
            array_to_string(
                look_at_quaternion(
                    position,
                    np.asarray(config.camera.wrist_target_link6_m),
                    np.asarray(config.camera.wrist_up_link6),
                )
            ),
        )
        camera.set("fovy", str(config.camera.wrist_fovy_deg))

    @property
    def naming_prefix(self) -> str:
        """保留已发布数据所使用的关节、TCP 和相机名称。"""
        return ""

    @property
    def default_base(self) -> str:
        """返回空底座；真实安装板已包含在机器人模型中。"""
        return ""

    @property
    def default_controller_config(self) -> dict[str, str]:
        """返回空配置；环境使用项目自有 TCP 控制链。"""
        return {}

    @property
    def init_qpos(self) -> np.ndarray:
        """返回 YAML 指定的六轴初始角，单位为弧度。"""
        return np.radians(self.config.robot.initial_joint_deg)

    @property
    def base_xpos_offset(self) -> dict[str, np.ndarray]:
        """返回机器人在焊接桌面上的标定安装位置。"""
        return {"table": np.asarray(self.config.scene.robot_base_position_m)}

    @property
    def top_offset(self) -> np.ndarray:
        """返回模型根部到最大竖直工作范围的近似偏移。"""
        return np.array([0.0, 0.0, 1.0])

    @property
    def horizontal_radius(self) -> float:
        """返回不含工具线缆的最大水平包络半径。"""
        return 0.85

    @property
    def _horizontal_radius(self) -> float:
        """实现 RobotModel 要求的水平包络属性。"""
        return self.horizontal_radius

    @property
    def _important_sites(self) -> dict[str, str]:
        """声明环境控制使用的末端 TCP site。"""
        return {"eef": "tcp"}
