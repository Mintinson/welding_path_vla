"""不同策略共享的 robosuite 闭环 rollout 与轨迹评估。

本模块实现了在 robosuite 焊接环境中运行策略的完整闭环流程：
  1. 采样/回放任务（焊缝几何、初始关节位置、TCP 偏移）；
  2. 逐步执行策略动作，进行安全校验、IK 求解和仿真步进；
  3. 记录每步的 TCP/关节/动作/焊缝进度等轨迹数据；
  4. 保存轨迹为 .npz，并计算覆盖率、到达率、方向率和 tracking CTE；
  5. 汇总多个 episode 的指标，输出 JSON 报告。

设计目标：
  - 不同策略（VLA、专家、基线）共享同一套 rollout 与评估流程；
  - 支持预验证的 scene bank，保证评估可复现；
  - 任何异常（碰撞、安全违规、动作违规）都被记录并计入统计。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.config_files import compose_config, merge_config
from welding_path_vla.core.deployment_rules import (
    COLLISION_FRAME_HOLD_S,
    deployment_rules_metadata,
)
from welding_path_vla.core.domain import Phase
from welding_path_vla.core.geometry import absolute_ee_action_to_pose
from welding_path_vla.dataset.video import VideoRecorder
from welding_path_vla.evaluation import evaluate_trace
from welding_path_vla.evaluation.adapters import SAFETY_SIGNALS
from welding_path_vla.evaluation.rollout_metrics import evaluate_rollout
from welding_path_vla.evaluation.schema import (
    EvaluationTrace,
    InstructionAssessment,
    SeamReference,
    Termination,
)
from welding_path_vla.policies.base import Observation, Policy
from welding_path_vla.policies.rollout_diagnostics import (
    RolloutCompletion,
    build_rollout_diagnostics,
    new_rollout_arrays,
)
from welding_path_vla.robot import SafetyMonitor, SafetyViolation
from welding_path_vla.simulation import ExpertTrajectory, WeldingEnv
from welding_path_vla.simulation.evaluation_scenes import (
    config_for_scene,
    load_scene_bank,
    replay_scene,
    scene_task_ids,
)
from welding_path_vla.simulation.task_sampling import (
    TaskSamplingError,
    sample_collision_free_task,
    sample_grouped_task_config,
    sample_initial_tcp_offset,
)

TASK_CONFIG_DIR = Path("configs/tasks")


@dataclass(frozen=True, slots=True)
class SimulationRolloutReport:
    """一次策略仿真 rollout 的产物和终止状态。

    Attributes:
        task: 稳定任务标识，用于跨任务汇总。
        episode: 当前任务内的 episode 编号。
        seed: 本次环境与任务采样使用的随机种子。
        steps: 实际执行的策略步数。
        completed: 是否满足自然完成条件。
        termination_reason: 完成、超时、碰撞或安全终止原因。
        collision: 本次执行是否出现有效碰撞。
        trace_path: 逐步 rollout 数组的保存路径。
        videos: 已保存的相机视频路径。
        diagnostics: 用于排查控制和跟踪问题的统计摘要。
        evaluation: 覆盖率、到达、方向率和 tracking 阶段 CTE 主结果。
    """

    task: str
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


def terminal_video_frame_count(
    termination_reason: str,
    fps: int,
) -> int:
    """返回终态需要写入的视频帧数。

    普通完成、超时和安全门终止只写一帧，用于保留最后一次动作后的真实状态；碰撞额外
    按固定时长重复碰撞帧，方便逐帧定位接触发生时的机械臂姿态。

    Args:
        termination_reason: rollout 的终止原因。
        fps: 视频帧率。
    Returns:
        需要追加到动作帧之后的终态帧数量。
    """
    if termination_reason != "collision":
        return 1
    return max(1, round(COLLISION_FRAME_HOLD_S * fps))


def rollout_episode(
    config: AppConfig,
    runtime: Policy,
    episode: int,
    output: Path,
    scene: dict[str, Any] | None = None,
) -> SimulationRolloutReport:
    """运行一个带安全门、日志和论文指标的策略 episode。

    执行流程：
      1. 初始化仿真环境、视频录制器、轨迹数组；
      2. 采样/回放任务，构建专家轨迹；
      3. 进入主循环：观察 → 策略动作 → 位移限幅 → IK → 安全校验 → 仿真步进；
      4. 记录每步轨迹数据，检查终止条件（碰撞/违规/完成/超时）；
      5. 收尾：保存轨迹、计算主结果、构建报告。

    Args:
        config: 全局应用配置。
        runtime: 策略运行时对象，需实现 `reset()` 和 `select_action(observation)`。
        episode: 当前 episode 编号（影响输出目录和默认种子）。
        output: 该 episode 的输出目录，必须不存在（避免覆盖）。
        scene: 可选预验证场景字典；提供时使用固定种子和预采样焊缝。

    Returns:
        包含完整轨迹、诊断和主结果的 SimulationRolloutReport。
    """
    # 固定 scene 已保存全部状态；seed 仅用于报告和随机模式的 episode 采样。
    seed = config.deployment.seed + episode

    # 无 scene 时按配置采样任务（方向、速度、角度等），用于随机化评估。
    if scene is None:
        config = sample_grouped_task_config(
            config,
            episode,
            config.deployment.seed,
            config.deployment.task_group_size,
            config.deployment.balance_directions,
        )

    # 初始化随机数生成器、仿真环境和输出目录。
    rng = np.random.default_rng(seed)
    simulation = WeldingEnv(config, seed)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(config.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 可选视频录制：按配置的相机名称和控制频率启动。
    camera_names = (config.camera.global_name, config.camera.wrist_name)
    recorder = (
        VideoRecorder.start(output, camera_names, config.timing.policy_hz)
        if config.deployment.record_video
        else None
    )

    # 初始化轨迹数组：所有需要的字段都是 list，便于逐步 append，最后转 np.ndarray。
    arrays = new_rollout_arrays()
    termination_reason = "timeout"
    try:
        # ---------- 任务准备 ----------
        if scene is not None:
            sampled = replay_scene(simulation, scene)  # 回放预验证场景
            seam, expert = sampled.seam, sampled.expert
        else:
            # 随机采样：先采样无碰撞焊缝，再随机化关节位置和初始 TCP 偏移。
            seam, _, _ = sample_collision_free_task(simulation, config, rng)
            simulation.randomize_joint_position(
                rng, config.randomization.joint_degs, config.randomization.max_sampling_attempts
            )
            sample_initial_tcp_offset(simulation, config, rng)
            expert = ExpertTrajectory(config, simulation.tcp_pose(), seam)

        completion = RolloutCompletion()  # 固定完成规则判定器

        # 固定场景不携带部署控制参数，安全限制统一来自用户配置。
        safety = SafetyMonitor(config.safety, simulation.joint_ranges)

        # 重置策略内部状态（如 VLA 的历史缓存）
        runtime.reset()

        # 单步最大位移：优先用配置值，否则由速度限制/控制频率推导。
        max_translation = config.deployment.max_target_increment_m or (
            config.safety.tcp_speed_limit_m_s / config.timing.policy_hz
        )
        max_ik_residual = config.deployment.max_ik_residual or config.robot.ik_tolerance * 10
        # 上一帧关节速度（用于加速度超限检测）和上一帧焊缝进度（用于投影单调性）。
        previous_velocity: np.ndarray | None = None
        previous_progress = 0.0
        suite_observation = simulation.observe()

        # ---------- 主控制循环 ----------
        for step in range(config.deployment.max_steps):
            # 每步开始时获取当前状态和图像，构造策略输入。
            state = simulation.state()
            images = simulation.images_from_observation(suite_observation)
            if recorder:
                recorder.append(images)

            # 构造统一的 Observation：包含时间、图像、状态向量、任务指令和工艺参数。
            observation = Observation(
                step / config.timing.policy_hz,
                images,
                np.concatenate(
                    (state.joint_position, state.tcp.position, state.tcp.quaternion_wxyz)
                ).astype(np.float32),
                config.task.instruction,
                config.task.direction,
                config.task.speed_mps,
                config.task.work_angle_deg,
                config.task.travel_angle_deg,
                config.task.tool_roll_deg,
            )
            # 策略推理：返回绝对末端执行器动作（位置 + 四元数）
            action = runtime.select_action(observation)

            # 初始化本步诊断变量；若执行失败，这些字段保持默认值（NaN/False/空）。
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
                # 将策略输出转换为目标位姿，并检查位移是否超过单步上限。
                command_pose = absolute_ee_action_to_pose(action)
                translation = command_pose.position - state.tcp.position
                translation_norm = float(np.linalg.norm(translation))

                # 位移限幅：若超出上限，按比例缩小到略小于上限（0.999999 倍避免边界）。
                if translation_norm > max_translation:
                    command_pose = type(command_pose)(
                        state.tcp.position
                        + translation * (0.999999 * max_translation / translation_norm),
                        command_pose.quaternion_wxyz,
                    )
                    action_increment = True
                    step_error = (
                        f"clipped absolute target increment {translation_norm:.6f} "
                        f"to {max_translation:.6f}"
                    )
                    # 后续绝对目标仍以旧 TCP 为锚点；限幅后立即丢弃策略缓存的动作，
                    # 以免策略内部仍按未限幅的目标计算，导致误差累积。
                    discard_pending = getattr(runtime, "discard_pending_actions", None)
                    if callable(discard_pending):
                        discard_pending()

                # 记录最终命令位姿（限幅后的）。
                command_position = command_pose.position
                command_quaternion = command_pose.quaternion_wxyz

                # IK 求解：从目标位姿求关节指令，残差过大视为安全违规。
                joint_command, residual = simulation.solve_ik(command_pose)
                if residual > max_ik_residual:
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
                # 动作违规：如四元数非法、动作维度错误等。
                action_increment = True
                termination_reason = "action_violation"
                step_error = str(error)
            except SafetyViolation as error:
                # 安全违规：可能是关节限位或 IK 残差过大。
                joint_limit = "joint command" in str(error)
                termination_reason = f"safety_violation: {error}"
                step_error = str(error)

            # ---------- 记录本步轨迹数据 ----------
            next_state = simulation.state()

            # 将新 TCP 投影到焊缝上，获取进度 alpha 和到焊缝的距离。
            # previous_progress 保证投影沿焊缝单调推进，避免回跳。
            projection = seam.project(next_state.tcp.position, previous_progress)
            alpha = projection.progress
            seam_distance = projection.distance_m
            previous_progress = alpha

            # 跟踪判定：距离在 tracking_band_m 内视为有效跟踪。
            track = seam_distance <= config.evaluation.tracking_band_m
            # 速度/加速度超限检测（用于详细评估的安全信号）。
            velocity_limit = bool(
                np.any(np.abs(next_state.joint_velocity) > config.safety.joint_velocity_limit_rad_s)
            )
            acceleration_limit = False
            if previous_velocity is not None:
                # 用有限差分计算加速度：Δv / Δt，其中 Δt = 1/频率。
                acceleration = (
                    next_state.joint_velocity - previous_velocity
                ) * config.timing.policy_hz
                acceleration_limit = bool(
                    np.any(np.abs(acceleration) > config.evaluation.joint_acceleration_limit_rad_s2)
                )
            previous_velocity = next_state.joint_velocity.copy()

            # 将所有字段追加到轨迹数组。注意区分 observation_*（执行前）和
            # 无前缀（执行后）的两套位姿，便于对比策略输入和实际结果。
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

            # ---------- 终止判定 ----------
            # 硬失败（碰撞/动作违规/安全违规）立即终止。
            if (
                collision
                or termination_reason == "action_violation"
                or termination_reason.startswith("safety_violation")
            ):
                break

            # 两阶段完成判定：先到达焊缝末端，再等待 TCP 明显离开焊缝。
            # 不约束退出方向，抬枪、侧移或越过终点均可完成。
            if completion.update(alpha, seam_distance):
                termination_reason = "completed"
                break

        terminal_frames = 0
        if recorder:
            terminal_images = simulation.images_from_observation(suite_observation)
            terminal_frames = terminal_video_frame_count(
                termination_reason,
                config.timing.policy_hz,
            )
            for _ in range(terminal_frames):
                recorder.append(terminal_images)
        trajectory = {name: np.asarray(values) for name, values in arrays.items()}
        trace_path = output / "rollout.npz"
        np.savez_compressed(
            trace_path,
            **trajectory,  # pyright: ignore[reportArgumentType]
        )
        track_frames = [frame for frame in expert.frames if frame.phase is Phase.TRACK]
        seam_points = np.stack([frame.pose.position for frame in track_frames])
        # 轨迹包含策略执行前的 TCP，避免第一步跨过起端窗口时误判为从未进入焊缝。
        metric_positions = np.concatenate(
            (trajectory["observation_tcp_position"][:1], trajectory["tcp_position"])
        )
        evaluation = evaluate_rollout(
            metric_positions,
            seam_points,
            termination_reason,
            config.evaluation,
            np.concatenate((np.array([False]), trajectory["track_mask"])),
            seam_is_closed=seam.is_closed,
        ).as_dict()
        if config.evaluation.report_detailed_metrics and np.sum(trajectory["track_mask"]) >= 2:
            safety_signals = {
                "collision": trajectory["collision"],
                "joint_limit": trajectory["joint_limit"],
                "joint_velocity": trajectory["joint_velocity_limit"],
                "joint_acceleration": trajectory["joint_acceleration"],
                "action_increment": trajectory["action_increment"],
            }
            trace = EvaluationTrace(
                trajectory["timestamp"],
                trajectory["tcp_position"],
                trajectory["tcp_quaternion_wxyz"],
                trajectory["track_mask"],
                SeamReference(
                    seam_points,
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
            detailed = evaluate_trace(
                trace, config.evaluation, seam_is_closed=seam.is_closed
            ).as_dict()
        else:
            detailed = None
        diagnostics = (
            build_rollout_diagnostics(
                trajectory,
                config,
                seam.start.position,
                seam.end.position,
                seam.start.normal,
                seam.end.normal,
                expert.post,
                termination_reason,
                terminal_frames,
            )
            if config.evaluation.report_detailed_metrics
            else {
                "termination_reason": termination_reason,
                "step_errors": sorted({str(value) for value in trajectory["step_error"] if value}),
            }
        )
        diagnostics["deployment_rules"] = deployment_rules_metadata()
        if detailed is not None:
            diagnostics["detailed_evaluation"] = detailed
        videos = recorder.finish() if recorder else ()
        report = SimulationRolloutReport(
            task=config.task.task_id,
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


def deployment_task_configs(config: AppConfig) -> tuple[AppConfig, ...]:
    """按 ``include_tasks`` 展开任务配置；空列表表示全部任务。"""
    task_ids = config.deployment.include_tasks or sorted(
        path.stem for path in TASK_CONFIG_DIR.glob("*.yaml")
    )
    task_paths = [TASK_CONFIG_DIR / f"{task_id}.yaml" for task_id in task_ids]
    missing = [path for path in task_paths if not path.is_file()]
    if missing:
        raise ValueError(f"unknown task IDs: {[path.stem for path in missing]}")
    if not task_paths:
        raise FileNotFoundError(f"no task YAML files under: {TASK_CONFIG_DIR}")
    configurations: list[AppConfig] = []
    for path in task_paths:
        values = merge_config(config.as_dict(), compose_config(path))
        values["deployment"]["include_tasks"] = [path.stem]
        task_config = AppConfig.from_dict(values)
        task_config.source_config_path = config.source_config_path
        configurations.append(task_config)
    return tuple(configurations)


def checkpoint_experiment_name(checkpoint: str | None) -> str:
    """从 checkpoint 路径提取训练实验目录名。"""
    path = Path(checkpoint or "checkpoint")
    if path.name == "pretrained_model" and path.parent.parent.name == "checkpoints":
        return path.parent.parent.parent.name
    return path.name


def deployment_output_dir(config: AppConfig) -> Path:
    """返回 ``output_root/{checkpoint_experiment}/YYYY-MM-DD-HH-MM``。"""
    return (
        Path(config.deployment.output_root)
        / checkpoint_experiment_name(config.policy.checkpoint)
        / datetime.now().strftime("%Y-%m-%d-%H-%M")
    )


def rollout_metric_summary(
    config: AppConfig,
    reports: list[SimulationRolloutReport],
) -> dict[str, Any]:
    """汇总部署来源、整体指标和 episode 分类索引。

    关键点：分母是 len(reports)（全部计划 episode），而非仅有评估的 episode。
    这样 setup 失败、无评估的 episode 也会拉低成功率，避免"失败被剔除"导致指标虚高。

    Args:
        config: 本次部署使用的模型与入口配置。
        reports: 全部 episode 的报告列表。

    Returns:
        指标在前、episode 分类索引在后的可比较字典。
    """
    metrics = [report.evaluation or {} for report in reports]
    cte = [item["cte_p95_m"] for item in metrics if item.get("cte_p95_m") is not None]
    count = len(reports)
    evaluated = [report.evaluation for report in reports if report.evaluation is not None]
    complete = [report for report in reports if report.completed]
    success = [
        report
        for report in reports
        if report.evaluation and bool(report.evaluation.get("success", False))
    ]
    timeout = [report for report in reports if report.termination_reason == "timeout"]
    collision = [report for report in reports if report.collision]

    def episode_refs(items: list[SimulationRolloutReport]) -> list[dict[str, str | int]]:
        """返回跨任务也不会混淆的 episode 索引。"""
        return [{"task": report.task, "episode": report.episode} for report in items]

    return {
        "checkpoint": config.policy.checkpoint,
        "config_path": config.source_config_path,
        "metrics": {
            "episodes": count,
            "complete_rate": len(complete) / count if count else 0.0,
            "complete_episodes": len(complete),
            "success_rate": len(success) / count if count else 0.0,
            "success_episodes": len(success),
            "arrival_rate": sum(bool(item.get("reached_start", False)) for item in metrics)
            / count
            if count
            else 0.0,
            "arrival_episodes": sum(bool(item.get("reached_start", False)) for item in metrics),
            "timeout_rate": len(timeout) / count if count else 0.0,
            "timeout_episodes": len(timeout),
            "collision_rate": len(collision) / count if count else 0.0,
            "collision_episodes": len(collision),
            "coverage_ratio_mean": sum(
                float(item.get("coverage_ratio", 0.0)) for item in metrics
            )
            / count
            if count
            else 0.0,
            "direction_ratio_mean": sum(
                float(item.get("direction_ratio", 0.0)) for item in metrics
            )
            / count
            if count
            else 0.0,
            "cte_p95_mean_m": float(np.mean(cte)) if cte else None,
            "cte_observed_episodes": len(cte),
            "tracking_frames_mean": float(
                np.mean([int(item.get("tracking_frames", 0)) for item in evaluated])
            )
            if evaluated
            else 0.0,
        },
        "episode_groups": {
            "complete": episode_refs(complete),
            "success": episode_refs(success),
            "timeout": episode_refs(timeout),
            "collision": episode_refs(collision),
        },
    }


def deploy_task_episodes(
    config: AppConfig,
    runtime: Policy,
    scenes: list[dict[str, Any]] | None = None,
    run_root: Path | None = None,
) -> list[SimulationRolloutReport]:
    """在一个任务的自动目录中连续运行多个 episode。

    两种模式：
      - scene 模式：按 scenes 列表依次执行，每个 scene 对应一个 episode。
      - 随机模式：运行 config.deployment.episodes 个 episode。

    采样失败（TaskSamplingError）时：
      - scene 模式下直接抛错（因为预验证场景不应失败）；
      - 随机模式下生成 setup_failure 报告，保证分母正确。

    Args:
        config: 任务配置。
        runtime: 策略运行时。
        scenes: 可选场景列表。
        run_root: 本次部署唯一的时间戳根目录；直接调用时自动创建。

    Returns:
        全部 episode 的报告列表。
    """
    if scenes is not None and not scenes:
        raise ValueError(f"scene bank has no scenes for task: {config.task.task_id}")
    task_config = config_for_scene(config, scenes[0]) if scenes else config
    root = (run_root or deployment_output_dir(task_config)) / task_config.task.task_id
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    episode_items = (
        [(int(scene["episode"]), scene) for scene in scenes]
        if scenes is not None
        else [(episode, None) for episode in range(config.deployment.episodes)]
    )
    for episode, scene in episode_items:
        output = root / f"episode_{episode:06d}"
        episode_config = config_for_scene(config, scene) if scene else config
        try:
            report = (
                rollout_episode(episode_config, runtime, episode, output, scene)
                if scene is not None
                else rollout_episode(episode_config, runtime, episode, output)
            )
        except TaskSamplingError as error:
            if scene is not None:
                raise RuntimeError(
                    "prevalidated scene could not be replayed: "
                    f"{scene['task']['task_id']} {episode}"
                ) from error
            report = SimulationRolloutReport(
                task=config.task.task_id,
                episode=episode,
                seed=config.deployment.seed + episode,
                steps=0,
                completed=False,
                termination_reason="setup_failure",
                collision=False,
                trace_path="",
                videos=(),
                diagnostics={"setup_error": str(error)},
                evaluation=None,
            )
            (output / "summary.json").write_text(
                json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        reports.append(report)
    return reports


def deploy_episodes(config: AppConfig, runtime: Policy) -> list[SimulationRolloutReport]:
    """按固定场景或随机模式执行全部任务，或执行指定任务 ID。"""
    root = deployment_output_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    if config.deployment.mode == "scene":
        if not config.deployment.scene_bank_path:
            raise ValueError("scene mode requires deployment.scene_bank_path")
        # 场景库模式：加载银行，确定要执行的任务列表。
        bank = load_scene_bank(config.deployment.scene_bank_path)
        tasks = scene_task_ids(config, bank)
        # 按任务筛选 scenes，逐个任务执行。
        reports = [
            report
            for task in tasks
            for report in deploy_task_episodes(
                config,
                runtime,
                [scene for scene in bank["scenes"] if scene["task"]["task_id"] == task],
                root,
            )
        ]
    elif config.deployment.mode == "random":
        reports = [
            report
            for task_config in deployment_task_configs(config)
            for report in deploy_task_episodes(task_config, runtime, run_root=root)
        ]
    else:
        raise ValueError(f"unsupported deployment mode: {config.deployment.mode}")
    (root / "metrics_summary.json").write_text(
        json.dumps(rollout_metric_summary(config, reports), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return reports


__all__ = [
    "SimulationRolloutReport",
    "deploy_episodes",
    "deploy_task_episodes",
    "deployment_output_dir",
    "deployment_task_configs",
    "rollout_episode",
    "rollout_metric_summary",
    "terminal_video_frame_count",
]
