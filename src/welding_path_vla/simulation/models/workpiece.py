"""可替换焊接工件的 robosuite ObjectModel。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from itertools import pairwise

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation.robosuite_compat import MujocoObject, array_to_string
from welding_path_vla.simulation.tasks import (
    CircularSeamPath,
    SeamPath,
    SinusoidalSeamPath,
    StraightSeamPath,
)

STEEL_RGBA = "0.34 0.27 0.22 1"
PLATE_RGBA = "0.46 0.45 0.42 1"


def append_geom_pair(body: ET.Element, attributes: dict[str, str]) -> None:
    """为同一几何添加隐藏碰撞体和可见外观体。

    Args:
        body: 接收 geom 的 MJCF body。
        attributes: 除碰撞掩码和渲染分组外的 MJCF geom 属性。
    """
    collision = ET.SubElement(body, "geom", attrib=attributes | {"group": "0"})
    collision.set("contype", "2")
    collision.set("conaffinity", "1")
    collision.set("rgba", "0 0 0 0")
    visual = deepcopy(collision)
    visual.set("name", f"{attributes['name']}_visual")
    visual.set("group", "1")
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    visual.set("mass", "1e-8")
    visual.set("rgba", attributes.get("rgba", PLATE_RGBA))
    body.append(visual)


class WorkpieceObject(MujocoObject):
    """由 YAML 选择几何，并提供对应的有向焊缝。

    当前类只包含已经用于实验的工件，避免为尚未出现的模型建立额外层级。
    后续新增工件时，只需增加一个几何构造函数和一个 ``seam`` 分支。
    """

    def __init__(self, config: AppConfig) -> None:
        """构建固定在桌面上的工件模型。

        Args:
            config: 项目统一配置。
        """
        super().__init__(obj_type="all", duplicate_collision_geoms=False)
        self.config = config
        self._name = "workpiece"
        builders = {
            "l_joint": self.build_l_joint,
            "pipe_on_plate": self.build_pipe_on_plate,
            "curve_plate": self.build_curve_plate,
        }
        self._obj = builders[config.workpiece.kind]()
        self._get_object_properties()

    @property
    def naming_prefix(self) -> str:
        """保留历史数据中的 ``workpiece`` 和板件 geom 名称。"""
        return ""

    def exclude_from_prefixing(self, inp: ET.Element | str) -> bool:
        """空前缀已保证名称稳定，无需排除任何元素。"""
        del inp
        return False

    def build_root(self) -> ET.Element:
        """创建位于配置世界坐标的工件根 body。"""
        return ET.Element(
            "body",
            {
                "name": "workpiece",
                "pos": array_to_string(self.config.scene.workpiece_position_m),
            },
        )

    def build_l_joint(self) -> ET.Element:
        """创建与原仿真尺寸一致的 L 形双板工件。"""
        body = self.build_root()
        workpiece = self.config.workpiece
        half_width = workpiece.l_joint_width_m / 2
        half_length = workpiece.l_joint_length_m / 2
        half_thickness = workpiece.l_joint_thickness_m / 2
        append_geom_pair(
            body,
            {
                "name": "plate_horizontal",
                "type": "box",
                "pos": array_to_string([half_width, 0, 0]),
                "size": array_to_string([half_width, half_length, half_thickness]),
                "rgba": "0.55 0.59 0.63 1",
            },
        )
        append_geom_pair(
            body,
            {
                "name": "plate_vertical",
                "type": "box",
                "pos": array_to_string([0, 0, half_width]),
                "size": array_to_string([half_thickness, half_length, half_width]),
                "rgba": "0.60 0.64 0.68 1",
            },
        )
        self.append_seam_sites(body)
        return body

    def build_pipe_on_plate(self) -> ET.Element:
        """用环向箱体近似照片中的空心圆管与方形底板。"""
        body = self.build_root()
        workpiece = self.config.workpiece
        plate = np.asarray(workpiece.pipe_plate_size_m) / 2
        append_geom_pair(
            body,
            {
                "name": "pipe_base_plate",
                "type": "box",
                "size": array_to_string(plate),
                "rgba": PLATE_RGBA,
            },
        )
        radius = workpiece.pipe_outer_radius_m - workpiece.pipe_wall_thickness_m / 2
        tangent_half = radius * np.tan(np.pi / workpiece.pipe_segments)
        z = plate[2] + workpiece.pipe_height_m / 2
        for index in range(workpiece.pipe_segments):
            angle = 2 * np.pi * index / workpiece.pipe_segments
            append_geom_pair(
                body,
                {
                    "name": f"pipe_wall_{index:02d}",
                    "type": "box",
                    "pos": array_to_string([radius * np.cos(angle), radius * np.sin(angle), z]),
                    "size": array_to_string(
                        [
                            workpiece.pipe_wall_thickness_m / 2,
                            tangent_half,
                            workpiece.pipe_height_m / 2,
                        ]
                    ),
                    "quat": array_to_string([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
                    "rgba": STEEL_RGBA,
                },
            )
        self.append_seam_sites(body)
        return body

    def build_curve_plate(self) -> ET.Element:
        """创建带有可见正弦或余弦焊缝的水平平板。"""
        body = self.build_root()
        plate = np.asarray(self.config.workpiece.curve_plate_size_m) / 2
        append_geom_pair(
            body,
            {
                "name": "curve_plate",
                "type": "box",
                "size": array_to_string(plate),
                "rgba": "0.48 0.51 0.54 1",
            },
        )
        seam = self.seam(np.zeros(3), np.eye(3))
        count = self.config.workpiece.curve_visual_segments
        points = []
        for index in range(count + 1):
            frame = seam.sample(index / count)
            points.append(frame.position - self.config.task.tcp_clearance_m * frame.normal)
        for index, (start, end) in enumerate(pairwise(points)):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"curve_seam_visual_{index:02d}",
                    "type": "capsule",
                    "fromto": array_to_string([*start, *end]),
                    "size": "0.0015",
                    "rgba": "0.12 0.06 0.025 1",
                    "group": "1",
                    "contype": "0",
                    "conaffinity": "0",
                    "mass": "1e-8",
                },
            )
        self.append_seam_sites(body)
        return body

    def append_seam_sites(self, body: ET.Element) -> None:
        """添加不可见起终点 site，兼容检查工具并辅助调试。"""
        seam = self.seam(np.zeros(3), np.eye(3))
        for name, frame in (("seam_start", seam.start), ("seam_end", seam.end)):
            ET.SubElement(
                body,
                "site",
                {
                    "name": name,
                    "pos": array_to_string(frame.position),
                    "size": "0.003",
                    "group": "5",
                    "rgba": "0 0 0 0",
                },
            )
        ET.SubElement(
            body,
            "site",
            {
                "name": "default_site",
                "pos": "0 0 0",
                "size": "0.002",
                "group": "5",
                "rgba": "0 0 0 0",
            },
        )

    def seam(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> SeamPath:
        """按工件类型返回世界坐标下的任务焊缝。

        Args:
            position: 工件原点世界坐标。
            rotation: 工件局部坐标到世界坐标的旋转矩阵。

        Returns:
            与当前工件匹配的焊缝路径对象。
        """
        task = self.config.task
        workpiece = self.config.workpiece
        work_angle = np.radians(task.work_angle_deg)
        if workpiece.kind == "l_joint":
            half_thickness = workpiece.l_joint_thickness_m / 2
            normal_local = np.array([np.sin(work_angle), 0, np.cos(work_angle)])
            seam_center = np.array([half_thickness, 0, half_thickness])
            seam_center += task.tcp_clearance_m * normal_local
            half_length = task.seam_length_m / 2
            start_local = seam_center + np.array([0, -half_length, 0])
            end_local = seam_center + np.array([0, half_length, 0])
            if task.direction == "reverse":
                start_local, end_local = end_local, start_local
            return StraightSeamPath(
                task.seam_id,
                position + rotation @ start_local,
                position + rotation @ end_local,
                rotation @ normal_local,
            )

        if workpiece.kind == "curve_plate":
            height = workpiece.curve_plate_size_m[2] / 2 + task.tcp_clearance_m
            return SinusoidalSeamPath(
                task.seam_id,
                position,
                rotation,
                task.seam_length_m,
                height,
                task.curve_amplitude_m,
                task.curve_frequency,
                task.curve_kind,
                task.direction == "reverse",
            )

        plate_half_height = workpiece.pipe_plate_size_m[2] / 2
        height = (
            plate_half_height
            if task.seam_id == "pipe_bottom"
            else plate_half_height + workpiece.pipe_height_m
        )
        sweep = np.radians(abs(task.arc_sweep_deg))
        if task.direction == "reverse":
            sweep = -sweep
        return CircularSeamPath(
            task.seam_id,
            position,
            rotation,
            workpiece.pipe_outer_radius_m,
            height,
            np.radians(task.arc_start_deg),
            sweep,
            work_angle,
            task.tcp_clearance_m,
        )

    @property
    def asset_id(self) -> str:
        """返回写入 episode 元数据的稳定工件标识。"""
        workpiece = self.config.workpiece
        if workpiece.kind == "l_joint":
            dimensions = (
                workpiece.l_joint_length_m,
                workpiece.l_joint_width_m,
                workpiece.l_joint_thickness_m,
            )
            return "l_joint_" + "x".join(str(round(value * 1000)) for value in dimensions)
        if workpiece.kind == "curve_plate":
            dimensions = "x".join(
                str(round(value * 1000)) for value in workpiece.curve_plate_size_m
            )
            return f"curve_plate_{dimensions}"
        diameter = round(2 * workpiece.pipe_outer_radius_m * 1000)
        height = round(workpiece.pipe_height_m * 1000)
        plate = "x".join(str(round(value * 1000)) for value in workpiece.pipe_plate_size_m)
        return f"pipe_d{diameter}_h{height}_plate_{plate}"

    @property
    def bottom_offset(self) -> np.ndarray:
        """返回根原点到工件最低点的偏移。"""
        thickness = {
            "l_joint": self.config.workpiece.l_joint_thickness_m,
            "pipe_on_plate": self.config.workpiece.pipe_plate_size_m[2],
            "curve_plate": self.config.workpiece.curve_plate_size_m[2],
        }[self.config.workpiece.kind]
        return np.array([0.0, 0.0, -thickness / 2])

    @property
    def top_offset(self) -> np.ndarray:
        """返回根原点到工件最高点的偏移。"""
        if self.config.workpiece.kind == "l_joint":
            height = self.config.workpiece.l_joint_width_m
        elif self.config.workpiece.kind == "pipe_on_plate":
            height = (
                self.config.workpiece.pipe_plate_size_m[2] / 2 + self.config.workpiece.pipe_height_m
            )
        else:
            height = self.config.workpiece.curve_plate_size_m[2] / 2
        return np.array([0.0, 0.0, height])

    @property
    def horizontal_radius(self) -> float:
        """返回工件水平包围圆半径。"""
        if self.config.workpiece.kind == "l_joint":
            size = np.array(
                [
                    self.config.workpiece.l_joint_width_m,
                    self.config.workpiece.l_joint_length_m,
                ]
            )
        elif self.config.workpiece.kind == "pipe_on_plate":
            size = np.asarray(self.config.workpiece.pipe_plate_size_m[:2])
        else:
            size = np.asarray(self.config.workpiece.curve_plate_size_m[:2])
        return float(np.linalg.norm(size) / 2)

    @property
    def contact_geom_rgba(self) -> np.ndarray:
        """返回 robosuite 调试碰撞体时使用的半透明颜色。"""
        return np.array([0.8, 0.1, 0.1, 0.3])

    def get_bounding_box_half_size(self) -> np.ndarray:
        """返回用于摆放检查的轴对齐包围盒半尺寸。"""
        return np.array([self.horizontal_radius, self.horizontal_radius, self.top_offset[2]])
