"""专家轨迹生成: 根据几何参数构造焊接过程的参考帧序列。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Phase, Pose
from welding_path_vla.core.geometry import matrix_to_quaternion


@dataclass(slots=True)
class ReferenceFrame:
    """焊接路径中的一个参考帧。

    Attributes:
        pose: 目标位姿 (位置 + 四元数)。
        phase: 当前所属阶段 (APPROACH / TRACK / RETREAT)。
        seam_progress: 在焊缝上的归一化进度 [0, 1], 非跟踪阶段取边界值。
    """

    pose: Pose
    phase: Phase
    seam_progress: float


class ExpertTrajectory:
    """从初始位姿到焊缝路径的完整专家轨迹。

    包含抬升 -> 平移 -> 下降 -> 下探 -> 跟踪 -> 后退六个阶段的参考帧序列,
    每个阶段根据速度配置和距离自动计算插值帧数。
    """

    def __init__(
        self,
        config: AppConfig,
        initial: Pose,
        seam_start: np.ndarray,
        seam_end: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        """初始化轨迹参数并构建完整帧序列。

        Args:
            config: 全局应用配置, 含速度、角度、间距等参数。
            initial: 机器人的初始 TCP 位姿。
            seam_start: 焊缝起点世界坐标。
            seam_end: 焊缝终点世界坐标。
            normal: 焊接法向 (工件坐标系下的法向量)。
        """
        self.config = config
        self.initial = initial
        self.seam_start = seam_start
        self.seam_end = seam_end
        self.tangent = (seam_end - seam_start) / np.linalg.norm(seam_end - seam_start)
        self.normal = normal / np.linalg.norm(normal)
        travel = np.radians(config.task.travel_angle_deg)
        self.approach_direction = np.cos(travel) * self.normal - np.sin(travel) * self.tangent
        self.welding_quaternion = self.welding_orientation()
        self.pre = seam_start + config.task.approach_distance_m * self.approach_direction
        self.post = seam_end + config.task.retreat_distance_m * self.approach_direction
        self.staging_height = max(
            float(initial.position[2]),
            float(seam_start[2] + config.task.staging_clearance_m),
        )
        self.lift = initial.position.copy()
        self.lift[2] = self.staging_height
        self.above_pre = self.pre.copy()
        self.above_pre[2] = self.staging_height
        self.frames = self.build_frames()

    def welding_orientation(self) -> np.ndarray:
        """计算焊接姿态四元数。

        根据进给方向、法向和工具倾角构建焊缝坐标系, 再叠加工具滚动角,
        得到保证焊丝对准熔池的 TCP 姿态四元数。

        Returns:
            焊接姿态四元数 (w, x, y, z)。
        """
        z_axis = self.approach_direction / np.linalg.norm(self.approach_direction)
        x_axis = self.tangent - np.dot(self.tangent, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)
        seam_frame = np.column_stack([x_axis, y_axis, z_axis])
        roll = Rotation.from_euler("z", self.config.task.tool_roll_deg, degrees=True).as_matrix()
        return matrix_to_quaternion(seam_frame @ roll)

    def segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        phase: Phase,
        start_quaternion: np.ndarray,
        end_quaternion: np.ndarray,
    ) -> list[ReferenceFrame]:
        """生成一段直线运动轨迹的参考帧序列。

        根据距离和焊接速度计算所需帧数, 在起点和终点之间线性插值位置,
        SLERP 插值姿态。

        Args:
            start: 段起点位置。
            end: 段终点位置。
            phase: 该段所属阶段。
            start_quaternion: 起点姿态四元数 (w, x, y, z)。
            end_quaternion: 终点姿态四元数 (w, x, y, z)。

        Returns:
            参考帧列表, 按时间顺序排列。
        """
        distance = float(np.linalg.norm(end - start))
        count = max(
            1, int(np.ceil(distance * self.config.timing.policy_hz / self.config.task.speed_mps))
        )
        frames: list[ReferenceFrame] = []
        for index in range(1, count + 1):
            alpha = index / count
            position = start + alpha * (end - start)
            progress = (
                float(
                    np.clip(
                        np.dot(position - self.seam_start, self.seam_end - self.seam_start)
                        / np.dot(self.seam_end - self.seam_start, self.seam_end - self.seam_start),
                        0,
                        1,
                    )
                )
                if phase is Phase.TRACK
                else (0.0 if phase is Phase.APPROACH else 1.0)
            )
            frames.append(
                ReferenceFrame(
                    Pose(
                        position,
                        interpolate_quaternion(start_quaternion, end_quaternion, alpha),
                    ),
                    phase,
                    progress,
                )
            )
        return frames

    def build_frames(self) -> list[ReferenceFrame]:
        """构建完整的六阶段专家轨迹帧序列。

        阶段顺序:
        1. 抬升 (lift): 从初始 TCP 位姿竖直向上至安全高度。
        2. 平移 (transfer): 水平移动到焊缝起点上方。
        3. 下降 (lower): 竖直下降至接近点。
        4. 下探 (descend): 从接近点移至焊缝起点。
        5. 跟踪 (track): 沿焊缝运动, 保持焊接姿态。
        6. 后退 (retreat): 焊缝终点后继续运动一段距离。

        Returns:
            按时间顺序拼接的所有参考帧列表。
        """
        initial_quaternion = self.initial.quaternion_wxyz
        welding_quaternion = self.welding_quaternion
        lift = self.segment(
            self.initial.position,
            self.lift,
            Phase.APPROACH,
            initial_quaternion,
            initial_quaternion,
        )
        transfer = self.segment(
            self.lift,
            self.above_pre,
            Phase.APPROACH,
            initial_quaternion,
            welding_quaternion,
        )
        lower = self.segment(
            self.above_pre,
            self.pre,
            Phase.APPROACH,
            welding_quaternion,
            welding_quaternion,
        )
        descend = self.segment(
            self.pre,
            self.seam_start,
            Phase.APPROACH,
            welding_quaternion,
            welding_quaternion,
        )
        track = self.segment(
            self.seam_start,
            self.seam_end,
            Phase.TRACK,
            welding_quaternion,
            welding_quaternion,
        )
        retreat = self.segment(
            self.seam_end,
            self.post,
            Phase.RETREAT,
            welding_quaternion,
            welding_quaternion,
        )
        return lift + transfer + lower + descend + track + retreat


def interpolate_quaternion(
    start_wxyz: np.ndarray, end_wxyz: np.ndarray, alpha: float
) -> np.ndarray:
    """对两个四元数进行球面线性插值 (SLERP)。

    scipy 的 Slerp 使用 (x, y, z, w) 格式, 因此需要先 roll 转换,
    插值完成后再 roll 回 (w, x, y, z) 格式。

    Args:
        start_wxyz: 起始四元数 (w, x, y, z)。
        end_wxyz: 终止四元数 (w, x, y, z)。
        alpha: 插值因子 [0, 1], 0 为起点, 1 为终点。

    Returns:
        插值后的四元数 (w, x, y, z)。
    """
    start = np.roll(start_wxyz, -1)
    end = np.roll(end_wxyz, -1)
    value = Slerp([0, 1], Rotation.from_quat([start, end]))([alpha]).as_quat()[0]
    return np.roll(value, 1)
