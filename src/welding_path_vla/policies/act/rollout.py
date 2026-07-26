"""ACT 在焊接 MuJoCo 场景中的闭环部署与轨迹评估。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Phase
from welding_path_vla.core.geometry import apply_tcp_action_to_world
from welding_path_vla.evaluation import evaluate_trace
from welding_path_vla.evaluation.adapters import SAFETY_SIGNALS
from welding_path_vla.evaluation.schema import (
    EvaluationTrace,
    InstructionAssessment,
    SeamReference,
    Termination,
)
from welding_path_vla.policies.base import Observation
from welding_path_vla.robot import SafetyMonitor, SafetyViolation
from welding_path_vla.simulation import ExpertTrajectory, WeldingSimulation
from welding_path_vla.simulation.collector import (
    sample_collision_free_task,
    sample_initial_tcp_offset,
)


@dataclass(frozen=True, slots=True)
class SimulationRolloutReport:
    """一次 ACT 仿真 rollout 的产物和终止状态。"""

    episode: int
    seed: int
    steps: int
    completed: bool
    termination_reason: str
    collision: bool
    trace_path: str
    videos: tuple[str, ...]
    evaluation: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RolloutVideoRecorder:
    """使用 LeRobot 流式编码器记录浏览器兼容的 H.264 视频。"""

    root: Path
    names: tuple[str, ...]
    encoder: Any

    @classmethod
    def start(cls, root: Path, config: AppConfig) -> RolloutVideoRecorder:
        """创建双相机 H.264 编码任务。"""
        from lerobot.configs import RGBEncoderConfig
        from lerobot.datasets.video_utils import StreamingVideoEncoder

        names = (config.camera.global_name, config.camera.wrist_name)
        video_config = RGBEncoderConfig(
            vcodec="h264",
            pix_fmt="yuv420p",
            g=config.timing.policy_hz * 2,
            crf=23,
            preset="veryfast",
        )
        encoder = StreamingVideoEncoder(config.timing.policy_hz, rgb_encoder=video_config)
        encoder.start_episode(list(names), root)
        return cls(root, names, encoder)

    def append(self, images: dict[str, np.ndarray]) -> None:
        """写入同一策略时刻的所有相机帧。"""
        for name in self.names:
            self.encoder.feed_frame(name, images[name])

    def finish(self) -> tuple[str, ...]:
        """完成编码并把临时视频移动到 episode 根目录。"""
        encoded = self.encoder.finish_episode()
        videos: list[str] = []
        for name in self.names:
            source, _ = encoded[name]
            destination = self.root / f"{name}.mp4"
            shutil.move(source, destination)
            shutil.rmtree(source.parent, ignore_errors=True)
            videos.append(str(destination))
        return tuple(videos)

    def close(self) -> None:
        """关闭编码器；异常中止时清理临时视频。"""
        self.encoder.close()


def rollout_episode(
    config: AppConfig,
    runtime: Any,
    episode: int,
    output: Path,
) -> SimulationRolloutReport:
    """运行一个带安全门、日志和论文指标的 ACT episode。"""
    seed = config.deployment.seed + episode
    rng = np.random.default_rng(seed)
    simulation = WeldingSimulation(config)
    output.mkdir(parents=True, exist_ok=False)
    recorder = (
        RolloutVideoRecorder.start(output, config) if config.deployment.record_video else None
    )
    arrays: dict[str, list[Any]] = {
        "timestamp": [],
        "tcp_position": [],
        "tcp_quaternion_wxyz": [],
        "joint_position": [],
        "joint_velocity": [],
        "action": [],
        "collision": [],
        "joint_limit": [],
        "joint_velocity_limit": [],
        "joint_acceleration": [],
        "action_increment": [],
        "track_mask": [],
    }
    termination_reason = "timeout"
    try:
        seam_start, seam_end, normal, _, _ = sample_collision_free_task(
            simulation,
            config,
            rng,
        )
        simulation.randomize_joint_position(
            rng,
            config.randomization.joint_degs,
            config.randomization.max_sampling_attempts,
        )
        sample_initial_tcp_offset(simulation, config, rng)
        expert = ExpertTrajectory(config, simulation.tcp_pose(), seam_start, seam_end, normal)
        safety = SafetyMonitor(config.safety, simulation.joint_ranges)
        runtime.reset()
        line = seam_end - seam_start
        line_norm_sq = float(np.dot(line, line))
        max_translation = config.safety.tcp_speed_limit_m_s / config.timing.policy_hz
        previous_velocity: np.ndarray | None = None
        for step in range(config.deployment.max_steps):
            state = simulation.state()
            images = simulation.render()
            if recorder:
                recorder.append(images)
            observation = Observation(
                step / config.timing.policy_hz,
                images,
                np.concatenate(
                    (state.joint_position, state.tcp.position, state.tcp.quaternion_wxyz)
                ).astype(np.float32),
                "",
            )
            action = runtime.select_action(observation)
            collision = False
            joint_limit = False
            action_increment = False
            try:
                command_pose = apply_tcp_action_to_world(state.tcp, action, max_translation)
                joint_command, residual = simulation.solve_ik(command_pose)
                if residual > config.robot.ik_tolerance * 10:
                    raise SafetyViolation(f"IK residual too large: {residual:.6f}")
                safety.validate_state(state)
                safety.validate_joint_command(joint_command)
                simulation.execute_pose(command_pose)
                collision = bool(simulation.last_collision_pairs)
                if collision:
                    termination_reason = "collision"
            except ValueError:
                action_increment = True
                termination_reason = "action_violation"
            except SafetyViolation as error:
                joint_limit = "joint command" in str(error)
                termination_reason = f"safety_violation: {error}"

            next_state = simulation.state()
            alpha = float(
                np.clip(np.dot(next_state.tcp.position - seam_start, line) / line_norm_sq, 0, 1)
            )
            closest = seam_start + alpha * line
            track = np.linalg.norm(next_state.tcp.position - closest) <= 0.01
            velocity_limit = bool(
                np.any(np.abs(next_state.joint_velocity) > config.safety.joint_velocity_limit_rad_s)
            )
            acceleration_limit = False
            if previous_velocity is not None:
                acceleration = (
                    next_state.joint_velocity - previous_velocity
                ) * config.timing.policy_hz
                acceleration_limit = bool(
                    np.any(np.abs(acceleration) > config.evaluation.joint_acceleration_limit_rad_s2)
                )
            previous_velocity = next_state.joint_velocity.copy()
            arrays["timestamp"].append((step + 1) / config.timing.policy_hz)
            arrays["tcp_position"].append(next_state.tcp.position)
            arrays["tcp_quaternion_wxyz"].append(next_state.tcp.quaternion_wxyz)
            arrays["joint_position"].append(next_state.joint_position)
            arrays["joint_velocity"].append(next_state.joint_velocity)
            arrays["action"].append(action)
            arrays["collision"].append(collision)
            arrays["joint_limit"].append(joint_limit)
            arrays["joint_velocity_limit"].append(velocity_limit)
            arrays["joint_acceleration"].append(acceleration_limit)
            arrays["action_increment"].append(action_increment)
            arrays["track_mask"].append(track)
            if collision or action_increment or termination_reason.startswith("safety_violation"):
                break
            if alpha >= config.evaluation.pcr_min and track:
                termination_reason = "completed"
                break

        trajectory = {name: np.asarray(values) for name, values in arrays.items()}
        trace_path = output / "rollout.npz"
        np.savez_compressed(
            trace_path,
            **trajectory,  # pyright: ignore[reportArgumentType]
        )
        track_count = int(np.sum(trajectory["track_mask"]))
        evaluation = None
        if track_count >= 2:
            safety_signals = {
                "collision": trajectory["collision"],
                "joint_limit": trajectory["joint_limit"],
                "joint_velocity": trajectory["joint_velocity_limit"],
                "joint_acceleration": trajectory["joint_acceleration"],
                "action_increment": trajectory["action_increment"],
            }
            track_frames = [frame for frame in expert.frames if frame.phase is Phase.TRACK]
            trace = EvaluationTrace(
                trajectory["timestamp"],
                trajectory["tcp_position"],
                trajectory["tcp_quaternion_wxyz"],
                trajectory["track_mask"],
                SeamReference(
                    np.stack([frame.pose.position for frame in track_frames]),
                    np.stack([frame.pose.quaternion_wxyz for frame in track_frames]),
                    config.task.speed_mps,
                    "straight_fillet_200mm",
                ),
                InstructionAssessment(True, True, True),
                safety_signals,
                SAFETY_SIGNALS,
                Termination(termination_reason == "completed", termination_reason == "timeout"),
                float(config.timing.policy_hz),
            )
            evaluation = evaluate_trace(trace, config.evaluation).as_dict()
        videos = recorder.finish() if recorder else ()
        report = SimulationRolloutReport(
            episode,
            seed,
            len(trajectory["timestamp"]),
            termination_reason == "completed",
            termination_reason,
            bool(np.any(trajectory["collision"])),
            str(trace_path),
            videos,
            evaluation,
        )
        (output / "summary.json").write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
    finally:
        if recorder:
            recorder.close()
        simulation.close()


def deploy_simulation(config: AppConfig, checkpoint: str) -> list[SimulationRolloutReport]:
    """加载一次 ACT，然后连续运行并保存多个仿真 episode。"""
    from welding_path_vla.policies.act.runtime import ACTRuntime

    runtime = ACTRuntime.from_pretrained(checkpoint, config.policy.device)
    root = Path(config.deployment.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = [
        rollout_episode(config, runtime, episode, root / f"episode_{episode:06d}")
        for episode in range(config.deployment.episodes)
    ]
    (root / "summary.json").write_text(
        json.dumps([report.as_dict() for report in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return reports


__all__ = [
    "RolloutVideoRecorder",
    "SimulationRolloutReport",
    "deploy_simulation",
    "rollout_episode",
]
