"""把 raw episode 和真机控制日志转换为统一评估输入。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from welding_path_vla.config import AppConfig
from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.evaluation.schema import (
    EvaluationTrace,
    InstructionAssessment,
    SeamReference,
    Termination,
)

SAFETY_SIGNALS = (
    "collision",
    "joint_limit",
    "joint_velocity",
    "joint_acceleration",
    "action_increment",
)


def sample_rate(timestamps: np.ndarray) -> float:
    """从日志时间戳估计采样率。"""
    return float(1 / np.median(np.diff(timestamps)))


def instruction_from_metadata(
    metadata: dict[str, object], assume_reference_task: bool
) -> InstructionAssessment:
    """读取结构化任务判断；专家基准可显式声明全部合规。"""
    if assume_reference_task:
        return InstructionAssessment(True, True, True)
    values = metadata.get("instruction_assessment", {})
    assessment = values if isinstance(values, dict) else {}
    return InstructionAssessment(
        bool(assessment.get("seam_correct", False)),
        bool(assessment.get("direction_correct", False)),
        bool(assessment.get("sequence_correct", False)),
    )


def trace_from_raw_episode(
    path: str | Path,
    config: AppConfig,
    assume_reference_task: bool = False,
) -> EvaluationTrace:
    """把项目 raw episode 转换为论文级评估轨迹。"""
    episode = EpisodeReader(path)
    trajectory = episode.trajectory
    phase = np.asarray(trajectory["phase"])
    track = phase == "track"
    timestamps = np.asarray(trajectory["timestamp"][1:], dtype=np.float64)
    reference_positions = np.asarray(trajectory["reference_position"])[track]
    reference_quaternions = np.asarray(trajectory["reference_quaternion_wxyz"])[track]
    joint_velocity = np.asarray(trajectory["joint_velocity"][1:])
    joint_acceleration = np.gradient(joint_velocity, timestamps, axis=0)
    step_time = np.diff(np.asarray(trajectory["timestamp"]))
    tcp_step_speed = (
        np.linalg.norm(trajectory["executed_delta_pose_world"][:, :3], axis=1) / step_time
    )
    safety = {
        "collision": np.asarray(trajectory["collision"], dtype=bool),
        "joint_limit": np.zeros(episode.action_count, dtype=bool),
        "joint_velocity": np.any(
            np.abs(joint_velocity) > config.safety.joint_velocity_limit_rad_s, axis=1
        ),
        "joint_acceleration": np.any(
            np.abs(joint_acceleration) > config.evaluation.joint_acceleration_limit_rad_s2,
            axis=1,
        ),
        "action_increment": tcp_step_speed > config.safety.tcp_speed_limit_m_s,
    }
    metadata = episode.metadata
    task = metadata.get("task_parameters", {})
    task_parameters = task if isinstance(task, dict) else {}
    return EvaluationTrace(
        timestamps,
        np.asarray(trajectory["tcp_position"][1:]),
        np.asarray(trajectory["tcp_quaternion_wxyz"][1:]),
        track,
        SeamReference(
            reference_positions,
            reference_quaternions,
            float(task_parameters.get("speed_mps", config.task.speed_mps)),
            str(metadata.get("seam_id", "unknown")),
        ),
        instruction_from_metadata(metadata, assume_reference_task),
        safety,
        SAFETY_SIGNALS,
        Termination(
            bool(np.asarray(trajectory["episode_done"])[-1]),
            bool(metadata.get("timed_out", False)),
            bool(metadata.get("operator_stopped", False)),
        ),
        sample_rate(timestamps),
        timestamps[track],
        reference_positions,
    )


def trace_from_real_robot_log(path: str | Path) -> EvaluationTrace:
    """读取标准真机评估目录中的 `control.npz + task.json`。"""
    root = Path(path)
    task = json.loads((root / "task.json").read_text(encoding="utf-8"))
    control = np.load(root / "control.npz", allow_pickle=False)
    timestamps = np.asarray(control["timestamp"], dtype=np.float64)
    track = (
        np.asarray(control["track_mask"], dtype=bool)
        if "track_mask" in control
        else np.asarray(control["phase"]) == "track"
    )
    seam = task["seam"]
    required = tuple(task.get("required_safety_signals", SAFETY_SIGNALS))
    safety = {name: np.asarray(control[name], dtype=bool) for name in required if name in control}
    assessment = task["instruction_assessment"]
    termination = task["termination"]
    expert_timestamps = (
        np.asarray(control["expert_timestamp"], dtype=np.float64)
        if "expert_timestamp" in control
        else None
    )
    expert_positions = (
        np.asarray(control["expert_position"], dtype=np.float64)
        if "expert_position" in control
        else None
    )
    return EvaluationTrace(
        timestamps,
        np.asarray(control["tcp_position"], dtype=np.float64),
        np.asarray(control["tcp_quaternion_wxyz"], dtype=np.float64),
        track,
        SeamReference(
            np.asarray(seam["points_world"], dtype=np.float64),
            np.asarray(seam["quaternions_wxyz"], dtype=np.float64),
            np.asarray(seam["desired_speed_mps"], dtype=np.float64)
            if isinstance(seam["desired_speed_mps"], list)
            else float(seam["desired_speed_mps"]),
            str(seam["seam_id"]),
        ),
        InstructionAssessment(
            bool(assessment["seam_correct"]),
            bool(assessment["direction_correct"]),
            bool(assessment.get("sequence_correct", True)),
        ),
        safety,
        required,
        Termination(
            bool(termination["completed"]),
            bool(termination.get("timed_out", False)),
            bool(termination.get("operator_stopped", False)),
        ),
        sample_rate(timestamps),
        expert_timestamps,
        expert_positions,
    )


__all__ = [
    "SAFETY_SIGNALS",
    "instruction_from_metadata",
    "trace_from_raw_episode",
    "trace_from_real_robot_log",
]
