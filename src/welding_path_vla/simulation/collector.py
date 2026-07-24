"""数据采集模块: 在仿真环境中采样无碰撞任务并记录演示轨迹。

职责:
  - 随机化工件位姿, 求解无碰撞预置姿态
  - 生成六阶段专家轨迹 (抬升→平移→下降→下探→跟踪→后退)
  - 执行轨迹并记录状态/动作/图像, 支持恢复扰动注入
  - 对每个 episode 计算质量指标, 过滤无效数据
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

from welding_path_vla.config import AppConfig
from welding_path_vla.dataset.raw_schema import RAW_DATASET_FORMAT
from welding_path_vla.dataset.recorder import EpisodeRecorder
from welding_path_vla.domain import CommandAction, Pose
from welding_path_vla.evaluation.trajectory_metrics import report_from_arrays
from welding_path_vla.geometry import frame_delta, pose_delta, rotation_error
from welding_path_vla.simulation import ExpertTrajectory, WeldingSimulation


class StagingPoseError(RuntimeError):
    """IK 求解失败或预置位姿发生碰撞时抛出的异常。"""


def stage_for_task(
    simulation: WeldingSimulation, config: AppConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """求解无碰撞预置位姿, 返回焊缝几何信息与 IK 残差。

    流程: 读取焊缝起止点 → 根据工件转角和工作角计算焊接法向 →
    构建草稿轨迹获得 approaching 点 → IK 求解该点 → 碰撞检查。

    Args:
        simulation: 焊接仿真环境实例。
        config: 全局应用配置。

    Returns:
        (seam_start, seam_end, normal, residual) 四元组,
        分别为焊缝起点、终点、焊接法向和 IK 残差。

    Raises:
        StagingPoseError: IK 解算残差 > 0.005 或预置位姿发生碰撞。
    """
    # 读取焊缝起止点在当前工件位姿下的世界坐标
    seam_start = simulation.site_position("seam_start")
    seam_end = simulation.site_position("seam_end")

    # 将工件坐标系下的焊接法向变换到世界坐标系
    workpiece_rotation = simulation.body_rotation("workpiece")
    work_angle = np.radians(config.task.work_angle_deg)
    normal = workpiece_rotation @ np.array([np.sin(work_angle), 0, np.cos(work_angle)])

    # 生成草稿轨迹, 提取 approaching 点作为 IK 目标
    draft = ExpertTrajectory(config, simulation.tcp_pose(), seam_start, seam_end, normal)
    solution, residual = simulation.solve_ik(Pose(draft.above_pre, draft.welding_quaternion))

    # IK 残差过大说明目标位姿不可达
    if residual > 0.005:
        raise StagingPoseError(f"cannot solve collision-free staging pose: residual={residual:.6f}")

    # 将 IK 解写入仿真并前向计算, 检查碰撞
    simulation.data.qpos[simulation.qpos_ids] = solution
    simulation.data.qvel[:] = 0
    simulation.data.ctrl[simulation.motor_ids] = solution
    mujoco.mj_forward(simulation.model, simulation.data)
    if simulation.collision:
        raise StagingPoseError(f"staging pose collides: {simulation.collision_pairs}")

    return seam_start, seam_end, normal, residual


def sample_collision_free_task(
    simulation: WeldingSimulation,
    config: AppConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """采样工件位姿, 直到得到可行的无碰撞预置位姿。

    外层循环随机化工件位置/偏航角, 内层 stage_for_task 求解预置 IK。
    失败时重置仿真并重新采样, 直至达到 max_sampling_attempts 上限。

    Args:
        simulation: 焊接仿真环境实例。
        config: 全局应用配置。
        rng: 随机数生成器。

    Returns:
        (seam_start, seam_end, normal, residual, attempt) 五元组,
        包含焊缝起点、终点、法向、IK 残差和成功时的尝试次数。

    Raises:
        RuntimeError: 超过最大采样次数仍无可行位姿。
    """
    last_error: StagingPoseError | None = None
    for attempt in range(1, config.randomization.max_sampling_attempts + 1):
        # 首次重试时不需要 reset（刚初始化）, 之后每次都要重置
        if attempt > 1:
            simulation.reset()
        simulation.randomize_workpiece(rng)
        try:
            seam_start, seam_end, normal, residual = stage_for_task(simulation, config)
        except StagingPoseError as error:
            last_error = error  # 保存最后一次错误信息用于最终异常
            continue
        return seam_start, seam_end, normal, residual, attempt
    raise RuntimeError(
        "cannot sample a reachable, collision-free task after "
        f"{config.randomization.max_sampling_attempts} attempts: {last_error}"
    ) from last_error


def sample_initial_tcp_offset(
    simulation: WeldingSimulation, config: AppConfig, rng: np.random.Generator
) -> tuple[np.ndarray, int, bool]:
    """采样一个可行的初始 TCP 偏移量。

    在给定范围内随机采样三维平移, 尝试 perturb_tcp 施加到仿真,
    如果 IK 无解或发生碰撞则重试。所有尝试均失败时返回零偏移,
    但保持已随机化的关节构型不变 (不影响后续采集)。

    Args:
        simulation: 焊接仿真环境实例。
        config: 全局应用配置。
        rng: 随机数生成器。

    Returns:
        (offset, attempts, applied) 三元组:
        - offset: 最终应用的偏移量, 失败时为 [0,0,0]。
        - attempts: 采样尝试次数。
        - applied: 是否成功应用了非零偏移。
    """
    for attempt in range(1, config.randomization.max_sampling_attempts + 1):
        # 在 [-initial_tcp_m, initial_tcp_m] 范围内均匀采样三维偏移
        offset = rng.uniform(
            -config.randomization.initial_tcp_m,
            config.randomization.initial_tcp_m,
            size=3,
        )
        if simulation.perturb_tcp(offset):
            return offset, attempt, True
    # 所有采样均失败, 返回零偏移并标记未应用
    return np.zeros(3), config.randomization.max_sampling_attempts, False


def collect_episode(config: AppConfig, episode_index: int, seed: int) -> Path:
    """采集单条演示轨迹并保存到磁盘。

    完整流程:
      1. 采样无碰撞工件位姿 (工件随机化 → 预置 IK → 碰撞检查)
      2. 随机化初始关节构型 (在安全范围内偏离标称位姿)
      3. 采样可行 TCP 偏移 (模拟实际部署时的手眼标定误差)
      4. 构建六阶段专家轨迹
      5. 沿专家轨迹逐帧执行, 每帧记录动作命令和跟踪误差
      6. 可选在轨迹中点注入恢复扰动, 模拟偏离-回归场景
      7. 计算轨迹质量指标, 写入 metadata.json

    Args:
        config: 全局应用配置。
        episode_index: 当前 episode 在数据集中的序号。
        seed: 该 episode 使用的随机种子。

    Returns:
        保存该 episode 数据的目录路径。

    Raises:
        RuntimeError: 采样无碰撞任务失败。
    """
    rng = np.random.default_rng(seed)
    simulation = WeldingSimulation(config)

    # --- 阶段 1: 采样无碰撞任务 (工件位姿随机化 + 预置 IK) ---
    (
        seam_start,
        seam_end,
        normal,
        staging_residual,
        scene_sampling_attempts,
    ) = sample_collision_free_task(simulation, config, rng)

    # --- 阶段 2: 在初始构型附近随机化关节位置, 增加轨迹多样性 ---
    initial_joint_offset_deg, joint_sampling_attempts = simulation.randomize_joint_position(
        rng,
        config.randomization.joint_degs,
        config.randomization.max_sampling_attempts,
    )

    # --- 阶段 3: 采样可行的 TCP 初始偏移, 模拟标定误差 ---
    initial_offset, tcp_sampling_attempts, initial_tcp_offset_applied = sample_initial_tcp_offset(
        simulation, config, rng
    )

    # --- 阶段 4: 初始化录制器和专家轨迹 ---
    root = Path(config.collection.dataset_root)
    recorder = EpisodeRecorder(root, episode_index, config)
    recovery = bool(rng.random() < config.randomization.recovery_probability)

    try:
        initial = simulation.tcp_pose()
        expert = ExpertTrajectory(config, initial, seam_start, seam_end, normal)
        state = simulation.state()

        # 记录初始状态 (t=0), 包含关节/TCP 信息和多视角图像
        recorder.append_state(0.0, state, simulation.render())

        # 构建焊缝坐标系: 三个正交方向 (切向/法向/副法向)
        seam_tangent = (seam_end - seam_start) / np.linalg.norm(seam_end - seam_start)
        seam_normal = expert.normal
        seam_binormal = np.cross(seam_normal, seam_tangent)
        seam_rotation = np.column_stack([seam_tangent, seam_binormal, seam_normal])

        # --- 阶段 5: 逐帧跟踪专家轨迹 ---
        recovery_step = len(expert.frames) // 2  # 恢复扰动施加在轨迹中点

        for index, frame in enumerate(expert.frames):
            # ---- 可选的恢复扰动: 在轨迹中点施加位置/姿态偏移 ----
            if recovery and index == recovery_step:
                # 沿法向 + 切向 + 副法向三个方向叠加随机偏移
                magnitude = config.randomization.recovery_position_m
                offset = (
                    seam_normal * rng.uniform(0.5 * magnitude, magnitude)
                    + seam_tangent * rng.uniform(-0.5 * magnitude, 0.5 * magnitude)
                    + seam_binormal * rng.uniform(-0.25 * magnitude, 0.25 * magnitude)
                )
                # 叠加随机姿态旋转
                rotation = np.radians(
                    rng.uniform(
                        -config.randomization.recovery_rotation_deg,
                        config.randomization.recovery_rotation_deg,
                        size=3,
                    )
                )
                simulation.perturb_tcp(offset, rotation)
                state = simulation.state()

            # ---- 计算三种坐标系下的动作增量命令 ----
            # 世界坐标系: 当前位姿 → 目标位姿的 delta pose
            command_world = pose_delta(
                state.tcp.position,
                state.tcp.quaternion_wxyz,
                frame.pose.position,
                frame.pose.quaternion_wxyz,
            )
            # 机器人基坐标系: 将世界 delta 旋转到基座标架
            command_base = frame_delta(command_world, simulation.base_rotation())
            # 焊缝坐标系: 将世界 delta 旋转到焊缝标架 (用于策略输入)
            command_seam = np.concatenate(
                [seam_rotation.T @ command_world[:3], seam_rotation.T @ command_world[3:]]
            )

            # ---- IK 执行目标位姿, 得到关节命令和残差 ----
            joint_command, ik_residual = simulation.execute_pose(frame.pose)

            # 计算安全关节命令对应的 TCP 位姿 (用于记录和校验)
            safe_command = simulation.pose_for_joint_position(joint_command)

            next_state = simulation.state()

            # ---- 计算跟踪误差 ----
            # 横向误差: 实际 TCP 位置到焊缝线段的最短距离
            line = seam_end - seam_start
            along = np.clip(
                np.dot(next_state.tcp.position - seam_start, line) / np.dot(line, line), 0, 1
            )
            closest = seam_start + along * line
            cross_track = float(np.linalg.norm(next_state.tcp.position - closest))

            # 姿态误差: 实际姿态与参考帧姿态之间的轴角范数 (度)
            orientation_error = float(
                np.degrees(
                    np.linalg.norm(
                        rotation_error(frame.pose.quaternion_wxyz, next_state.tcp.quaternion_wxyz)
                    )
                )
            )

            # ---- 记录碰撞信息 ----
            collision_pairs = simulation.last_collision_pairs

            # ---- 记录一步动作数据 ----
            recorder.append_action(
                CommandAction(command_seam, command_base, command_world, joint_command),
                frame.pose,
                safe_command,
                frame.phase.value,
                frame.seam_progress,
                cross_track,
                orientation_error,
                ik_residual,
                bool(collision_pairs),
                "|".join(f"{first}:{second}" for first, second in collision_pairs),
                recovery and recovery_step <= index < recovery_step + 10,
            )

            # ---- 记录执行后的状态 (t > 0) ----
            timestamp = (index + 1) / config.timing.policy_hz
            recorder.append_state(timestamp, next_state, simulation.render())
            state = next_state

        # --- 阶段 6: 计算质量指标, 构建元数据, 完成录制 ---
        provisional = {name: np.asarray(values) for name, values in recorder.arrays.items()}
        report = report_from_arrays(provisional, config.quality, recovery)

        metadata = {
            "seed": seed,
            "robot_model": config.robot.model_id,
            "asset_id": "l_joint_300x100x5",
            "seam_id": "straight_fillet_200mm",
            "instruction": config.task.instruction,
            "direction": "forward",
            "episode_start": "collision_checked_staging_pose",
            # 采样过程元数据
            "staging_ik_residual": staging_residual,
            "scene_sampling_attempts": scene_sampling_attempts,
            "initial_joint_offset_deg": initial_joint_offset_deg.tolist(),
            "initial_joint_sampling_attempts": joint_sampling_attempts,
            "initial_tcp_offset_m": initial_offset.tolist(),
            "initial_tcp_sampling_attempts": tcp_sampling_attempts,
            "initial_tcp_offset_applied": initial_tcp_offset_applied,
            # 任务参数 (供训练时恢复参考)
            "task_parameters": {
                "speed_mps": config.task.speed_mps,
                "work_angle_deg": config.task.work_angle_deg,
                "travel_angle_deg": config.task.travel_angle_deg,
                "tool_roll_deg": config.task.tool_roll_deg,
            },
            # 各坐标系定义说明
            "coordinate_frames": {
                "tcp_and_reference": "world",
                "safe_command": "world",
                "command_delta_pose_base": "robot_base",
                "command_delta_pose_world": "world",
                "command_delta_pose_seam": "seam",
            },
            "recovery": recovery,
            # 工件最终位姿 (可用于域随机化分析)
            "workpiece_position": simulation.model.body_pos[simulation.workpiece_id].tolist(),
            "workpiece_quaternion_wxyz": simulation.model.body_quat[
                simulation.workpiece_id
            ].tolist(),
            # 轨迹质量指标
            "quality": report.as_dict(),
        }
        return recorder.finish(metadata)
    except Exception:
        recorder.abort()
        raise
    finally:
        simulation.close()


def collect_datasetx(config: AppConfig, episodes: int | None = None) -> list[Path]:
    """采集指定数量的有效轨迹, 构建完整数据集。

    不断采集 episode 直到有效数量达标, 对每个 episode 记录质量元数据,
    最终生成 dataset.json 摘要文件。无效 episode 会保留在磁盘上供人工检查,
    但不会计入有效计数。

    Args:
        config: 全局应用配置。
        episodes: 目标有效 episode 数, 未指定则使用配置默认值。

    Returns:
        所有采集到的 episode 目录路径列表 (含无效 episode)。

    Raises:
        RuntimeError: 达到 max_attempt_multiplier 倍尝试次数后
                      仍无法收集到足够的有效 episode。
    """
    target = episodes or config.collection.episodes
    root = Path(config.collection.dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    # 从已有 episode 推断下一个起始序号, 支持增量采集
    existing = sorted((root / "episodes").glob("episode_*"))
    next_index = max((int(path.name.rsplit("_", 1)[1]) for path in existing), default=-1) + 1

    paths: list[Path] = []
    valid = 0
    attempts = 0

    # 循环采集, 直到有效数达标或超过最大尝试次数
    while valid < target and attempts < target * config.collection.max_attempt_multiplier:
        episode_index = next_index + attempts
        path = collect_episode(config, episode_index, config.collection.seed + episode_index)
        paths.append(path)
        # 从元数据读取质量判定, 累加有效计数
        quality = json.loads((path / "metadata.json").read_text(encoding="utf-8"))["quality"]
        valid += int(quality["valid"])
        attempts += 1

    # 汇总全部 episode (含之前已有的) 的质量分布
    all_episodes = sorted((root / "episodes").glob("episode_*"))
    quality = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))["quality"]
        for path in all_episodes
    ]
    summary = {
        "dataset": root.name,
        "last_request_valid_episodes": target,
        "last_request_collected_valid_episodes": valid,
        "valid_episodes": sum(item["valid"] for item in quality),
        "attempted_episodes": len(all_episodes),
        "status": dict(Counter(item["status"] for item in quality)),
        "seed": config.collection.seed,
        "format": RAW_DATASET_FORMAT,
    }
    (root / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if valid < target:
        raise RuntimeError(
            f"collected only {valid}/{target} valid episodes after {attempts} attempts; "
            "failed episodes were retained"
        )
    return paths


def collect_dataset(config: AppConfig, episodes: int | None = None) -> list[Path]:
    """采集指定数量的有效轨迹, 构建完整数据集。

    不断采集 episode 直到有效数量达标, 对每个 episode 记录质量元数据,
    最终生成 dataset.json 摘要文件。无效 episode 会保留在磁盘上供人工检查,
    但不会计入有效计数。

    Args:
        config: 全局应用配置。
        episodes: 目标有效 episode 数, 未指定则使用配置默认值。

    Returns:
        所有采集到的 episode 目录路径列表 (含无效 episode)。

    Raises:
        RuntimeError: 达到 max_attempt_multiplier 倍尝试次数后
                      仍无法收集到足够的有效 episode。
    """

    target = episodes or config.collection.episodes
    root = Path(config.collection.dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    # 从已有 episode 推断下一个起始序号, 支持增量采集
    existing = sorted((root / "episodes").glob("episode_*"))
    next_index = max((int(path.name.rsplit("_", 1)[1]) for path in existing), default=-1) + 1

    paths: list[Path] = []
    valid = 0
    attempts = 0
    max_attempts = target * config.collection.max_attempt_multiplier

    def right_align(text: str) -> str:
        """根据当前终端宽度将文本右对齐。"""
        width = shutil.get_terminal_size(fallback=(120, 20)).columns
        return text.rjust(max(width - 1, len(text)))

    with (
        tqdm(
            total=target,
            desc=f"Collecting Valid Episodes (target: {target})",
            unit="episode",
            position=0,
            dynamic_ncols=True,
            bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"),
        ) as pbar,
        tqdm(
            total=0,
            position=1,
            bar_format="{desc}",
            leave=False,
            dynamic_ncols=False,
        ) as stats_bar,
    ):
        while valid < target and attempts < max_attempts:
            episode_index = next_index + attempts

            path = collect_episode(
                config,
                episode_index,
                config.collection.seed + episode_index,
            )
            paths.append(path)

            quality = json.loads((path / "metadata.json").read_text(encoding="utf-8"))["quality"]

            is_valid = int(quality["valid"])
            status_emoji = "✅" if is_valid else "❌"

            tqdm.write(f"Episode {episode_index:04d}: {status_emoji} {quality['status']}")

            if is_valid:
                valid += 1
                pbar.update(1)

            attempts += 1

            stats = (
                f"attempts={attempts}/{max_attempts}, "
                f"success_rate={valid / attempts:.1%}, "
                f"valid={valid}"
            )

            stats_bar.set_description_str(right_align(stats), refresh=True)
            pbar.refresh()
            # pbar.refresh()
    # 汇总全部 episode (含之前已有的) 的质量分布
    all_episodes = sorted((root / "episodes").glob("episode_*"))
    quality = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))["quality"]
        for path in all_episodes
    ]
    summary = {
        "dataset": root.name,
        "last_request_valid_episodes": target,
        "last_request_collected_valid_episodes": valid,
        "valid_episodes": sum(item["valid"] for item in quality),
        "attempted_episodes": len(all_episodes),
        "status": dict(Counter(item["status"] for item in quality)),
        "seed": config.collection.seed,
        "format": RAW_DATASET_FORMAT,
    }
    (root / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if valid < target:
        raise RuntimeError(
            f"collected only {valid}/{target} valid episodes after {attempts} attempts; "
            "failed episodes were retained"
        )
    return paths
