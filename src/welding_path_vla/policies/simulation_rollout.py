"""不同策略共享的 robosuite 闭环 rollout 与轨迹评估。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import Phase
from welding_path_vla.core.geometry import apply_tcp_action_to_world
from welding_path_vla.dataset.video import VideoRecorder
from welding_path_vla.evaluation import evaluate_trace
from welding_path_vla.evaluation.adapters import SAFETY_SIGNALS
from welding_path_vla.evaluation.schema import (
    EvaluationTrace,
    InstructionAssessment,
    SeamReference,
    Termination,
)
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.rollout_diagnostics import (
    build_rollout_diagnostics,
    new_rollout_arrays,
    rollout_completed,
)
from welding_path_vla.robot import SafetyMonitor, SafetyViolation
from welding_path_vla.simulation import ExpertTrajectory, WeldingEnv
from welding_path_vla.simulation.task_sampling import (
    sample_collision_free_task,
    sample_initial_tcp_offset,
)


@dataclass(frozen=True, slots=True)
class SimulationRolloutReport:
    """一次策略仿真 rollout 的产物和终止状态。"""

    episode: int
    seed: int
    steps: int
    completed: bool
    termination_reason: str
    collision: bool
    trace_path: str
    videos: tuple[str, ...]
    diagnostics: dict[str, Any]
    evaluation: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rollout_episode(
    config: AppConfig,
    runtime: Any,
    episode: int,
    output: Path,
) -> SimulationRolloutReport:
    """运行一个带安全门、日志和论文指标的策略 episode。"""
    seed = config.deployment.seed + episode
    rng = np.random.default_rng(seed)
    simulation = WeldingEnv(config, seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(config.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    camera_names = (config.camera.global_name, config.camera.wrist_name)
    recorder = (
        VideoRecorder.start(output, camera_names, config.timing.policy_hz)
        if config.deployment.record_video
        else None
    )
    arrays = new_rollout_arrays()
    termination_reason = "timeout"
    try:
        seam, _, _ = sample_collision_free_task(
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
        expert = ExpertTrajectory(config, simulation.tcp_pose(), seam)
        safety = SafetyMonitor(config.safety, simulation.joint_ranges)
        runtime.reset()
        max_translation = config.safety.tcp_speed_limit_m_s / config.timing.policy_hz
        previous_velocity: np.ndarray | None = None
        previous_progress = 0.0
        suite_observation = simulation.observe()
        for step in range(config.deployment.max_steps):
            state = simulation.state()
            images = simulation.images_from_observation(suite_observation)
            if recorder:
                recorder.append(images)
            observation = Observation(
                step / config.timing.policy_hz,
                images,
                np.concatenate(
                    (state.joint_position, state.tcp.position, state.tcp.quaternion_wxyz)
                ).astype(np.float32),
                config.task.instruction,
            )
            action = runtime.select_action(observation)
            command_position = np.full(3, np.nan)
            command_quaternion = np.full(4, np.nan)
            joint_command = np.full_like(state.joint_position, np.nan)
            residual = float("nan")
            collision = False
            collision_pairs: tuple[tuple[str, str], ...] = ()
            joint_limit = False
            action_increment = False
            step_error = ""
            try:
                bounded_action = action.copy()
                translation_norm = float(np.linalg.norm(bounded_action[:3]))
                if translation_norm > max_translation:
                    bounded_action[:3] *= 0.999999 * max_translation / translation_norm
                    action_increment = True
                    step_error = (
                        f"clipped TCP increment {translation_norm:.6f} "
                        f"to {max_translation:.6f}"
                    )
                command_pose = apply_tcp_action_to_world(
                    state.tcp,
                    bounded_action,
                    max_translation,
                )
                command_position = command_pose.position
                command_quaternion = command_pose.quaternion_wxyz
                joint_command, residual = simulation.solve_ik(command_pose)
                if residual > config.robot.ik_tolerance * 10:
                    raise SafetyViolation(f"IK residual too large: {residual:.6f}")
                safety.validate_state(state)
                safety.validate_joint_command(joint_command)
                pose_action = np.concatenate([command_pose.position, command_pose.quaternion_wxyz])
                suite_observation = simulation.step(pose_action)[0]
                collision_pairs = simulation.last_collision_pairs
                collision = bool(collision_pairs)
                if collision:
                    termination_reason = "collision"
                    step_error = "collision: " + "|".join(
                        f"{first}:{second}" for first, second in collision_pairs
                    )
            except ValueError as error:
                action_increment = True
                termination_reason = "action_violation"
                step_error = str(error)
            except SafetyViolation as error:
                joint_limit = "joint command" in str(error)
                termination_reason = f"safety_violation: {error}"
                step_error = str(error)

            next_state = simulation.state()
            projection = seam.project(next_state.tcp.position, previous_progress)
            alpha = projection.progress
            seam_distance = projection.distance_m
            previous_progress = alpha
            track = seam_distance <= config.evaluation.tracking_band_m
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
            arrays["observation_tcp_position"].append(state.tcp.position)
            arrays["observation_tcp_quaternion_wxyz"].append(state.tcp.quaternion_wxyz)
            arrays["observation_joint_position"].append(state.joint_position)
            arrays["tcp_position"].append(next_state.tcp.position)
            arrays["tcp_quaternion_wxyz"].append(next_state.tcp.quaternion_wxyz)
            arrays["joint_position"].append(next_state.joint_position)
            arrays["joint_velocity"].append(next_state.joint_velocity)
            arrays["action"].append(action)
            arrays["command_tcp_position"].append(command_position)
            arrays["command_tcp_quaternion_wxyz"].append(command_quaternion)
            arrays["joint_command"].append(joint_command)
            arrays["ik_residual_m"].append(residual)
            arrays["seam_progress"].append(alpha)
            arrays["seam_distance_m"].append(seam_distance)
            arrays["collision"].append(collision)
            arrays["collision_pairs"].append(
                "|".join(f"{first}:{second}" for first, second in collision_pairs)
            )
            arrays["joint_limit"].append(joint_limit)
            arrays["joint_velocity_limit"].append(velocity_limit)
            arrays["joint_acceleration"].append(acceleration_limit)
            arrays["action_increment"].append(action_increment)
            arrays["step_error"].append(step_error)
            arrays["track_mask"].append(track)
            if (
                collision
                or termination_reason == "action_violation"
                or termination_reason.startswith("safety_violation")
            ):
                break
            if rollout_completed(alpha, seam_distance, config.deployment):
                termination_reason = "completed"
                break

        if recorder:
            recorder.append(simulation.images_from_observation(suite_observation))
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
                    seam.seam_id,
                ),
                InstructionAssessment(True, True, True),
                safety_signals,
                SAFETY_SIGNALS,
                Termination(termination_reason == "completed", termination_reason == "timeout"),
                float(config.timing.policy_hz),
            )
            evaluation = evaluate_trace(trace, config.evaluation).as_dict()
        diagnostics = build_rollout_diagnostics(
            trajectory,
            config,
            seam.start.position,
            seam.end.position,
            seam.start.normal,
            seam.end.normal,
            termination_reason,
            recorder is not None,
        )
        videos = recorder.finish() if recorder else ()
        report = SimulationRolloutReport(
            episode=episode,
            seed=seed,
            steps=len(trajectory["timestamp"]),
            completed=termination_reason == "completed",
            termination_reason=termination_reason,
            collision=bool(np.any(trajectory["collision"])),
            trace_path=str(trace_path),
            videos=videos,
            diagnostics=diagnostics,
            evaluation=evaluation,
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


def deploy_episodes(config: AppConfig, runtime: Any) -> list[SimulationRolloutReport]:
    """用已加载的策略连续运行并保存多个仿真 episode。"""
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
    "SimulationRolloutReport",
    "deploy_episodes",
    "rollout_episode",
]
