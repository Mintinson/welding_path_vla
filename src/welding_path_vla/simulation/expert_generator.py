"""根据有向焊缝几何生成接近、跟踪和退出参考轨迹。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Phase, Pose
from welding_path_vla.core.geometry import matrix_to_quaternion
from welding_path_vla.simulation.tasks import SeamFrame, SeamPath


@dataclass(slots=True)
class ReferenceFrame:
    """焊接路径中的一个参考帧。

    Attributes:
        pose: 世界坐标目标位姿。
        phase: 接近、跟踪或退出阶段。
        seam_progress: 有向焊缝归一化进度；非跟踪阶段取相邻边界值。
    """

    pose: Pose
    phase: Phase
    seam_progress: float


class ExpertTrajectory:
    """从当前 TCP 到任意 ``SeamPath`` 的完整专家轨迹。"""

    def __init__(self, config: AppConfig, initial: Pose, seam: SeamPath) -> None:
        """构建抬升、转移、下降、跟踪和退出轨迹。

        Args:
            config: 全局应用配置。
            initial: 当前 TCP 世界位姿。
            seam: 当前工件提供的有向直线或圆弧焊缝。
        """
        self.config = config
        self.initial = initial
        self.seam = seam
        self.start_frame = seam.start
        self.end_frame = seam.end
        self.seam_start = self.start_frame.position
        self.seam_end = self.end_frame.position
        self.tangent = self.start_frame.tangent
        self.normal = self.start_frame.normal
        self.fixed_welding_quaternion = self.geometric_welding_orientation(self.start_frame)
        self.welding_quaternion = self.fixed_welding_quaternion
        start_approach = self.approach_direction(self.start_frame)
        end_approach = self.approach_direction(self.end_frame)
        self.pre = self.seam_start + config.task.approach_distance_m * start_approach
        self.post = self.seam_end + config.task.retreat_distance_m * end_approach
        self.staging_height = max(
            float(initial.position[2]),
            float(self.seam_start[2] + config.task.staging_clearance_m),
        )
        self.lift = initial.position.copy()
        self.lift[2] = self.staging_height
        self.above_pre = self.pre.copy()
        self.above_pre[2] = self.staging_height
        self.frames = self.build_frames()

    def approach_direction(self, frame: SeamFrame) -> np.ndarray:
        """计算某焊缝标架处带行走角的焊枪接近方向。"""
        travel = np.radians(self.config.task.travel_angle_deg)
        direction = np.cos(travel) * frame.normal - np.sin(travel) * frame.tangent
        return direction / np.linalg.norm(direction)

    def geometric_welding_orientation(self, frame: SeamFrame) -> np.ndarray:
        """根据当前焊缝局部标架计算完整跟随姿态。

        Args:
            frame: 当前焊缝局部标架。

        Returns:
            wxyz 顺序的 TCP 姿态四元数。
        """
        z_axis = self.approach_direction(frame)
        x_axis = frame.tangent - np.dot(frame.tangent, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)
        seam_rotation = np.column_stack([x_axis, y_axis, z_axis])
        roll = Rotation.from_euler(
            "z",
            self.config.task.tool_roll_deg,
            degrees=True,
        ).as_matrix()
        return matrix_to_quaternion(seam_rotation @ roll)

    def welding_orientation(self, frame: SeamFrame) -> np.ndarray:
        """按配置比例跟随焊缝局部姿态。

        ``orientation_follow_ratio=0`` 时整段保持起点姿态，取 ``1`` 时
        完全跟随局部切向和法向，中间值用于限制圆弧任务的姿态变化。

        Args:
            frame: 当前焊缝局部标架。

        Returns:
            wxyz 顺序的 TCP 姿态四元数。
        """
        dynamic_quaternion = self.geometric_welding_orientation(frame)
        return interpolate_quaternion(
            self.fixed_welding_quaternion,
            dynamic_quaternion,
            self.config.task.orientation_follow_ratio,
        )

    def segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        phase: Phase,
        start_quaternion: np.ndarray,
        end_quaternion: np.ndarray,
        seam_progress: float,
        speed_mps: float,
    ) -> list[ReferenceFrame]:
        """生成一段直线过渡，并使用 SLERP 连续插值姿态。

        Args:
            start: 世界坐标起点。
            end: 世界坐标终点。
            phase: 轨迹阶段。
            start_quaternion: 起点 wxyz 姿态。
            end_quaternion: 终点 wxyz 姿态。
            seam_progress: 该过渡段记录的焊缝边界进度。
            speed_mps: 当前阶段的 TCP 平移速度。

        Returns:
            不重复起点、包含终点的参考帧序列。
        """
        distance = float(np.linalg.norm(end - start))
        count = max(
            1,
            int(np.ceil(distance * self.config.timing.policy_hz / speed_mps)),
        )
        return [
            ReferenceFrame(
                Pose(
                    start + index / count * (end - start),
                    interpolate_quaternion(
                        start_quaternion,
                        end_quaternion,
                        index / count,
                    ),
                ),
                phase,
                seam_progress,
            )
            for index in range(1, count + 1)
        ]

    def track_frames(self) -> list[ReferenceFrame]:
        """按焊接速度离散焊缝，并按配置连续插值圆弧姿态。"""
        count = max(
            1,
            int(
                np.ceil(
                    self.seam.length_m * self.config.timing.policy_hz / self.config.task.speed_mps
                )
            ),
        )
        frames: list[ReferenceFrame] = []
        for index in range(1, count + 1):
            progress = index / count
            seam_frame = self.seam.sample(progress)
            frames.append(
                ReferenceFrame(
                    Pose(
                        seam_frame.position,
                        self.welding_orientation(seam_frame),
                    ),
                    Phase.TRACK,
                    progress,
                )
            )
        return frames

    def build_frames(self) -> list[ReferenceFrame]:
        """按安全接近、焊缝跟踪和法向退出的顺序拼接完整轨迹。"""
        initial_quaternion = self.initial.quaternion_wxyz
        start_quaternion = self.welding_quaternion
        end_quaternion = self.welding_orientation(self.end_frame)
        lift = self.segment(
            self.initial.position,
            self.lift,
            Phase.APPROACH,
            initial_quaternion,
            initial_quaternion,
            0.0,
            self.config.task.approach_speed_mps,
        )
        transfer = self.segment(
            self.lift,
            self.above_pre,
            Phase.APPROACH,
            initial_quaternion,
            start_quaternion,
            0.0,
            self.config.task.approach_speed_mps,
        )
        lower = self.segment(
            self.above_pre,
            self.pre,
            Phase.APPROACH,
            start_quaternion,
            start_quaternion,
            0.0,
            self.config.task.approach_speed_mps,
        )
        descend = self.segment(
            self.pre,
            self.seam_start,
            Phase.APPROACH,
            start_quaternion,
            start_quaternion,
            0.0,
            self.config.task.approach_speed_mps,
        )
        retreat = self.segment(
            self.seam_end,
            self.post,
            Phase.RETREAT,
            end_quaternion,
            end_quaternion,
            1.0,
            self.config.task.retreat_speed_mps,
        )
        return lift + transfer + lower + descend + self.track_frames() + retreat


def interpolate_quaternion(
    start_wxyz: np.ndarray,
    end_wxyz: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """使用 scipy 对两个 wxyz 四元数做球面线性插值。

    Args:
        start_wxyz: 起始 wxyz 四元数。
        end_wxyz: 终止 wxyz 四元数。
        alpha: 插值比例 ``[0, 1]``。

    Returns:
        插值后的 wxyz 四元数。
    """
    start = np.roll(start_wxyz, -1)
    end = np.roll(end_wxyz, -1)
    value = Slerp([0, 1], Rotation.from_quat([start, end]))([alpha]).as_quat()[0]
    return np.roll(value, 1)
