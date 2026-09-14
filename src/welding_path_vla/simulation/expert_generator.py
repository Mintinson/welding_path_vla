"""
根据有向焊缝几何生成接近、跟踪和退出参考轨迹。

该模块实现了从当前 TCP 位姿到指定有向焊缝（直线或圆弧）的完整专家轨迹规划，
包括抬升、水平转移、下降、焊缝跟踪和法向退出等阶段。所有轨迹帧以固定控制频率
离散，并输出世界坐标系下的目标位姿（位置 + 四元数姿态）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Phase, Pose
from welding_path_vla.core.geometry import matrix_to_quaternion, quaternion_to_matrix
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
    """从当前 TCP 到任意 ``SeamPath`` 的完整专家轨迹。

    该类根据配置参数和焊缝几何信息，生成一条包含接近、跟踪和退出阶段的平滑轨迹。
    轨迹生成考虑了焊枪的工艺角度（工作角、行走角、滚转角）以及安全抬升高度等约束，
    并保证姿态的连续性（避免圆周焊接时的姿态跳变）。
    """

    def __init__(
        self,
        config: AppConfig,
        initial: Pose,
        seam: SeamPath,
        build_frames: bool = True,
    ) -> None:
        """构建抬升、转移、下降、跟踪和退出轨迹。

        Args:
            config: 全局应用配置，包含任务参数（速度、角度、距离等）和时序参数。
            initial: 当前 TCP 世界位姿（通常为机械臂初始末端位姿）。
            seam: 当前工件提供的有向直线或圆弧焊缝，包含起点/终点局部标架和长度。
            build_frames: 是否立即离散完整轨迹；在 staging IK 只需要关键位姿时可设为
                False，以节省计算开销。
        """
        self.config = config
        self.initial = initial
        self.seam = seam

        # 焊缝起点/终点局部标架及其位置
        self.start_frame = seam.start
        self.end_frame = seam.end
        self.seam_start = self.start_frame.position
        self.seam_end = self.end_frame.position

        # 起点处的切向和法向（用于后续姿态计算）
        self.tangent = self.start_frame.tangent
        self.normal = self.start_frame.normal

        # 根据起点局部标架计算固定的焊接姿态（四元数）
        self.fixed_welding_quaternion = self.geometric_welding_orientation(self.start_frame)
        # 焊接姿态的初始值（在跟踪过程中可能会根据姿态跟随比例进行调整）
        self.welding_quaternion = self.fixed_welding_quaternion

        # 计算接近起点和退出终点：沿焊枪轴线方向偏移设定距离
        start_approach = self.approach_direction(self.start_frame)
        end_approach = self.approach_direction(self.end_frame)
        self.pre = self.seam_start + config.task.approach_distance_m * start_approach
        self.post = self.seam_end + config.task.retreat_distance_m * end_approach

        # 抬升高度：取当前 TCP 高度和焊缝起点上方安全间隙的较大值
        self.staging_height = max(
            float(initial.position[2]),
            float(self.seam_start[2] + config.task.staging_clearance_m),
        )

        # 抬升目标点（保持水平位置，只改变高度）
        self.lift = initial.position.copy()
        self.lift[2] = self.staging_height

        # 焊缝起点正上方（与 pre 点等高）
        self.above_pre = self.pre.copy()
        self.above_pre[2] = self.staging_height
        self.frames = self.build_frames() if build_frames else []

    def approach_direction(self, frame: SeamFrame) -> np.ndarray:
        """按统一的 work/travel 定义计算焊枪轴线方向。

        该函数根据配置的工作角和行走角，基于焊缝局部标架计算焊枪 TCP 的 Z 轴方向
        （即焊枪指向焊缝的方向）。工作角（work_angle）绕焊缝切向旋转参考法向，
        行走角（travel_angle）使焊枪在工作角轴线与切向之间前后倾斜。

        Args:
            frame: 焊缝局部标架（包含切向、法向和工作参考方向）。

        Returns:
            单位向量，表示焊枪 TCP 的 Z 轴在世界坐标系中的方向。
        """
        work = np.radians(self.config.task.work_angle_deg)
        travel = np.radians(self.config.task.travel_angle_deg)

        # 先绕切向旋转工作参考方向得到工作角后的轴线
        work_axis = Rotation.from_rotvec(work * frame.tangent).apply(frame.work_reference)

        # 再在 work_axis 与切向形成的平面内倾斜 travel 角
        # 公式：direction = cos(travel)*work_axis - sin(travel)*tangent
        direction = np.cos(travel) * work_axis - np.sin(travel) * frame.tangent
        return direction / np.linalg.norm(direction)

    def geometric_welding_orientation(self, frame: SeamFrame) -> np.ndarray:
        """根据当前焊缝局部标架计算完整跟随姿态。

        该方法构建一个右手坐标系，使得：
          - Z 轴与焊枪轴线方向一致（由 approach_direction 计算）；
          - X 轴尽可能与焊缝切向对齐（正交化后）；
          - Y 轴由右手定则确定。
        然后在此基础上施加绕 Z 轴的工具滚转角（tool_roll_deg）。

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

    def follow_orientation(
        self,
        previous_output: np.ndarray,
        previous_geometric: np.ndarray,
        current_geometric: np.ndarray,
    ) -> np.ndarray:
        """按比例累积相邻焊缝帧之间的小旋转。

        逐点从起始姿态做 SLERP 会在圆周超过 180° 时切换最短旋转方向，
        产生不连续姿态。相邻帧增量始终很小，累积后可稳定覆盖完整圆周。

        Args:
            previous_output: 上一帧实际输出的 wxyz 姿态。
            previous_geometric: 上一帧完整几何跟随姿态。
            current_geometric: 当前帧完整几何跟随姿态。

        Returns:
            连续且按配置比例跟随的 wxyz 姿态。
        """
        previous_matrix = quaternion_to_matrix(previous_geometric)
        current_matrix = quaternion_to_matrix(current_geometric)
        rotation_vector = Rotation.from_matrix(previous_matrix.T @ current_matrix).as_rotvec()
        scaled_delta = Rotation.from_rotvec(
            self.config.task.orientation_follow_ratio * rotation_vector
        ).as_matrix()
        return matrix_to_quaternion(quaternion_to_matrix(previous_output) @ scaled_delta)

    def transport_welding_orientation(
        self,
        previous_orientation: np.ndarray,
        previous_frame: SeamFrame,
        current_frame: SeamFrame,
    ) -> np.ndarray:
        """用最小旋转把上一焊枪轴平行运输到当前焊枪轴。

        圆管整周若每帧都强制 TCP X 轴对齐切向，会额外让腕部滚转 360°；这不是
        焊接工艺要求，也会把连续 IK 推到关节限位。平行运输只跟随 work/travel
        决定的实际焊枪轴，保留上一帧滚转，因此 ``orientation_follow_ratio=1``
        仍表示完整轴线跟随，同时避免无意义的腕部绕枪轴旋转。
        """
        previous_axis = self.approach_direction(previous_frame)
        current_axis = self.approach_direction(current_frame)
        cross = np.cross(previous_axis, current_axis)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-12:
            return previous_orientation.copy()
        angle = np.arctan2(cross_norm, np.dot(previous_axis, current_axis))
        delta = Rotation.from_rotvec(angle * cross / cross_norm).as_matrix()
        return matrix_to_quaternion(delta @ quaternion_to_matrix(previous_orientation))

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
        """按焊接速度离散焊缝，并连续累积圆弧姿态增量。

        对于圆弧焊缝，直接使用全局几何姿态可能会在超过半圆时产生姿态跳变，
        因此采用相邻帧增量累积的方式保证姿态连续。对于轴对称为主的焊缝
        （工作角和行走角接近 0），使用平行运输法避免额外的滚转。

        Returns:
            焊缝跟踪阶段的参考帧列表。
        """
        # 根据焊缝长度、焊接速度和策略频率确定离散帧数
        count = max(
            1,
            int(
                np.ceil(
                    self.seam.length_m * self.config.timing.policy_hz / self.config.task.speed_mps
                )
            ),
        )
        frames: list[ReferenceFrame] = []
        previous_frame = self.start_frame
        previous_geometric = self.fixed_welding_quaternion
        output_orientation = self.fixed_welding_quaternion
        for index in range(1, count + 1):
            progress = index / count
            seam_frame = self.seam.sample(progress)  # 采样焊缝上的点
            # 判断是否接近轴对称（工作角和行走角都很小），此时使用平行运输
            axisymmetric = (
                abs(self.config.task.work_angle_deg) < 5
                and abs(self.config.task.travel_angle_deg) < 5
            )
            # 计算当前帧的“几何目标姿态”（未应用跟随比例）
            current_geometric = (
                # 平行运输：只跟随焊枪轴，保留滚转
                self.transport_welding_orientation(
                    previous_geometric,
                    previous_frame,
                    seam_frame,
                )
                if axisymmetric
                # 完整几何跟随：根据局部标架计算姿态
                else self.geometric_welding_orientation(seam_frame)
            )
            # 按 orientation_follow_ratio 比例累积增量，得到实际输出姿态
            output_orientation = self.follow_orientation(
                output_orientation,
                previous_geometric,
                current_geometric,
            )
            frames.append(
                ReferenceFrame(
                    Pose(
                        seam_frame.position,
                        output_orientation,
                    ),
                    Phase.TRACK,
                    progress,
                )
            )
            # 更新上一帧状态
            previous_frame = seam_frame
            previous_geometric = current_geometric
        return frames

    def build_frames(self) -> list[ReferenceFrame]:
        """按安全接近、焊缝跟踪和法向退出的顺序拼接完整轨迹。

        轨迹阶段：
          1. 抬升：从当前 TCP 垂直上升到安全高度；
          2. 水平转移：从抬升点水平移动到焊缝起点的正上方；
          3. 下降：从正上方下降到接近起点（pre 点）；
          4. 最终接近：从 pre 点沿焊枪轴缓慢移动到焊缝起点；
          5. 起焊点短暂停顿（用于消除残余误差）；
          6. 焊缝跟踪；
          7. 法向退出：从焊缝终点沿焊枪轴缓慢后退。
        """
        initial_quaternion = self.initial.quaternion_wxyz
        start_quaternion = self.welding_quaternion
        track = self.track_frames()
        end_quaternion = track[-1].pose.quaternion_wxyz
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
        # 起焊点短暂停稳，让位置伺服消除下降段残余误差；这些帧仍属于接近阶段，
        # 不会把控制器的瞬态误差错误计入焊缝跟踪质量。
        settle = [
            ReferenceFrame(
                Pose(self.seam_start.copy(), start_quaternion.copy()), Phase.APPROACH, 0.0
            )
            for _ in range(round(0.25 * self.config.timing.policy_hz))
        ]
        retreat = self.segment(
            self.seam_end,
            self.post,
            Phase.RETREAT,
            end_quaternion,
            end_quaternion,
            1.0,
            self.config.task.retreat_speed_mps,
        )
        return lift + transfer + lower + descend + settle + track + retreat


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
