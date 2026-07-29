"""固定于焊接桌面的 robosuite Arena。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from importlib.resources import files

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import look_at_quaternion
from welding_path_vla.simulation.robosuite_compat import Arena, array_to_string


class WeldingArena(Arena):
    """包含地面、焊接桌和桌面固定全局相机的场景模型。"""

    def __init__(self, config: AppConfig) -> None:
        """加载场景，并应用桌面尺寸及相机外参。

        Args:
            config: 项目统一配置。
        """
        path = files("welding_path_vla").joinpath("assets", "elfin5", "welding_arena.xml")
        super().__init__(str(path))
        table = self.worldbody.find("./body[@name='table_frame']")
        assert table is not None
        table.set("pos", array_to_string(config.scene.table_center_m))
        for name in ("table", "table_collision"):
            geom = table.find(f"./geom[@name='{name}']")
            assert geom is not None
            geom.set("size", array_to_string(config.scene.table_half_size_m))
        camera = table.find(f"./camera[@name='{config.camera.global_name}']")
        assert camera is not None
        position = np.asarray(config.scene.global_camera_position_table_m)
        camera.set("pos", array_to_string(position))
        camera.set(
            "quat",
            array_to_string(
                look_at_quaternion(
                    position,
                    np.asarray(config.scene.global_camera_target_table_m),
                    np.asarray(config.scene.global_camera_up_table),
                )
            ),
        )
        camera.set("fovy", str(config.camera.global_fovy_deg))

    def apply_headlight(self, task_root: ET.Element) -> None:
        """把 Arena 的主视角补光参数应用到 robosuite Task。

        robosuite 合并 Arena 时只复制 worldbody 和资产，不会复制 ``visual``，
        因此需要在 Task 组合完成后显式保留 Arena 的 headlight。

        Args:
            task_root: 已组合完成的 robosuite Task XML 根节点。
        """
        source = self.root.find("./visual/headlight")
        visual = task_root.find("./visual")
        assert source is not None and visual is not None
        target = ET.SubElement(visual, "headlight")
        target.attrib.update(source.attrib)
