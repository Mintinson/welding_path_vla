"""Episode 录制器: 在仿真或真机运行中将状态、动作和图像写到磁盘。"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import CommandAction, Pose, RobotState
from welding_path_vla.core.geometry import frame_delta, rotation_error, yaw_degrees_to_matrix
from welding_path_vla.dataset.raw_schema import RAW_DATASET_FORMAT
from welding_path_vla.dataset.video import VideoRecorder


class EpisodeRecorder:
    """在暂存目录中逐步记录 episode 数据, 完成后原子地移至最终位置。

    数据包括时间戳、关节状态、TCP 位姿、多视角视频, 以及每一步的动作
    指令和跟踪误差。finish() 方法统一持久化并写入 metadata。
    """

    def __init__(self, root: Path, episode_index: int, config: AppConfig) -> None:
        """创建录制器, 在 .incomplete 下建立暂存目录。

        Args:
            root: 数据集根目录。
            episode_index: 当前 episode 序号。
            config: 全局配置, 用于视频编码和分辨率。
        """
        self.root = root
        self.episode_index = episode_index
        self.config = config
        self.final_path = root / "episodes" / f"episode_{episode_index:06d}"
        self.temporary_path = root / ".incomplete" / f"episode_{episode_index:06d}"
        self.arrays: dict[str, list[Any]] = defaultdict(list)
        self.temporary_path.mkdir(parents=True, exist_ok=False)
        camera_names = (config.camera.global_name, config.camera.wrist_name)
        self.video = VideoRecorder.start(
            self.temporary_path,
            camera_names,
            config.timing.policy_hz,
        )

    def append_state(
        self, timestamp: float, state: RobotState, images: dict[str, np.ndarray]
    ) -> None:
        """记录一个时间步的状态和相机图像。

        Args:
            timestamp: 相对于 episode 开始的时间戳 (秒)。
            state: 当前机器人状态 (关节、TCP)。
            images: 相机名称到 RGB 图像的映射。
        """
        self.arrays["timestamp"].append(timestamp)
        self.arrays["joint_position"].append(state.joint_position)
        self.arrays["joint_velocity"].append(state.joint_velocity)
        self.arrays["tcp_position"].append(state.tcp.position)
        self.arrays["tcp_quaternion_wxyz"].append(state.tcp.quaternion_wxyz)
        self.video.append(images)

    def append_action(
        self,
        action: CommandAction,
        reference: Pose,
        safe_command: Pose,
        phase: str,
        seam_progress: float,
        cross_track_error: float,
        orientation_error_deg: float,
        ik_residual: float,
        collision: bool,
        collision_pairs: str,
        recovery_window: bool,
    ) -> None:
        """记录一步动作指令及执行结果指标。

        Args:
            action: 多坐标系下的动作增量命令。
            reference: 参考位姿 (专家帧目标)。
            safe_command: IK 求解后经过限幅的安全关节命令对应位姿。
            phase: 当前阶段 (approach/track/retreat)。
            seam_progress: 在焊缝上的归一化进度。
            cross_track_error: 横向跟踪误差 (米)。
            orientation_error_deg: 姿态跟踪误差 (度)。
            ik_residual: IK 残差。
            collision: 是否发生碰撞。
            collision_pairs: 碰撞几何体名称对, "|" 分隔。
            recovery_window: 是否在恢复扰动窗口内。
        """
        self.arrays["command_delta_pose_seam"].append(action.delta_pose_seam)
        self.arrays["command_delta_pose_base"].append(action.delta_pose_base)
        self.arrays["command_delta_pose_world"].append(action.delta_pose_world)
        self.arrays["joint_position_command"].append(action.joint_position)
        self.arrays["reference_position"].append(reference.position)
        self.arrays["reference_quaternion_wxyz"].append(reference.quaternion_wxyz)
        self.arrays["safe_command_position"].append(safe_command.position)
        self.arrays["safe_command_quaternion_wxyz"].append(safe_command.quaternion_wxyz)
        self.arrays["phase"].append(phase)
        self.arrays["seam_progress"].append(seam_progress)
        self.arrays["cross_track_error"].append(cross_track_error)
        self.arrays["orientation_error_deg"].append(orientation_error_deg)
        self.arrays["ik_residual"].append(ik_residual)
        self.arrays["collision"].append(collision)
        self.arrays["collision_pairs"].append(collision_pairs)
        self.arrays["recovery_window"].append(recovery_window)

    def finish(self, metadata: dict[str, Any]) -> Path:
        """完成录制: 关闭写入器, 验证数据, 持久化到磁盘。

        检查 state 数量 = action 数量 + 1, 计算执行增量,
        写入 trajectory.npz 和 metadata.json, 最后原子重命名
        暂存目录到最终位置。

        Args:
            metadata: 需要写入 JSON 的元数据字典。

        Returns:
            最终 episode 目录路径。

        Raises:
            ValueError: state 与 action 数量不匹配。
        """
        self.video.finish()
        arrays = {name: np.asarray(values) for name, values in self.arrays.items()}
        state_count = len(arrays["timestamp"])
        action_count = len(arrays["phase"])
        if state_count != action_count + 1:
            raise ValueError(
                f"episode requires N actions and N+1 states, got {action_count}/{state_count}"
            )
        arrays["episode_done"] = np.arange(action_count) == action_count - 1
        executed_rotation = np.asarray(
            [
                rotation_error(end, start)
                for start, end in zip(
                    arrays["tcp_quaternion_wxyz"][:-1],
                    arrays["tcp_quaternion_wxyz"][1:],
                    strict=True,
                )
            ]
        )
        executed = np.concatenate(
            [arrays["tcp_position"][1:] - arrays["tcp_position"][:-1], executed_rotation],
            axis=1,
        )
        arrays["executed_delta_pose_world"] = executed
        world_from_base = yaw_degrees_to_matrix(self.config.scene.robot_base_yaw_deg)
        arrays["executed_delta_pose_base"] = np.asarray(
            [frame_delta(delta, world_from_base) for delta in executed]
        )
        np.savez_compressed(
            self.temporary_path / "trajectory.npz",
            **arrays,  # pyright: ignore[reportArgumentType]
        )
        document = {
            **metadata,
            "dataset_format": RAW_DATASET_FORMAT,
            "episode_index": self.episode_index,
            "state_count": state_count,
            "action_count": action_count,
            "quaternion_order": "wxyz",
            "position_unit": "m",
            "angle_unit": "rad",
            "resolved_config": self.config.as_dict(),
        }
        (self.temporary_path / "metadata.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.temporary_path, self.final_path)
        return self.final_path

    def abort(self) -> None:
        """中止录制: 释放写入器并清理暂存目录。"""
        self.video.close()
        shutil.rmtree(self.temporary_path, ignore_errors=True)
