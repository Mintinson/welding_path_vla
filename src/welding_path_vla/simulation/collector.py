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

import numpy as np
from tqdm import tqdm

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.domain import CommandAction
from welding_path_vla.core.geometry import frame_delta, pose_delta, rotation_error
from welding_path_vla.dataset.raw_schema import RAW_DATASET_FORMAT
from welding_path_vla.dataset.recorder import EpisodeRecorder
from welding_path_vla.evaluation.trajectory_metrics import report_from_arrays
from welding_path_vla.simulation import ExpertTrajectory, WeldingEnv
from welding_path_vla.simulation.task_sampling import (
    sample_collision_free_task,
    sample_initial_tcp_offset,
)


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
    simulation = WeldingEnv(config, seed, ignore_done=True)

    # --- 阶段 1: 采样无碰撞任务 (工件位姿随机化 + 预置 IK) ---
    seam, staging_residual, scene_sampling_attempts = sample_collision_free_task(
        simulation,
        config,
        rng,
    )

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
        expert = ExpertTrajectory(config, initial, seam)
        state = simulation.state()
        observation = simulation.observe()

        # 记录初始状态 (t=0), 包含关节/TCP 信息和多视角图像
        recorder.append_state(
            0.0,
            state,
            simulation.images_from_observation(observation),
        )

        # --- 阶段 5: 逐帧跟踪专家轨迹 ---
        recovery_step = len(expert.frames) // 2  # 恢复扰动施加在轨迹中点

        for index, frame in enumerate(expert.frames):
            # 圆弧任务中切向与法向会随进度变化，所有局部量都使用当前焊缝标架。
            seam_frame = seam.sample(frame.seam_progress)
            seam_tangent = seam_frame.tangent
            seam_normal = seam_frame.normal
            seam_binormal = np.cross(seam_normal, seam_tangent)
            seam_rotation = seam_frame.rotation

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

            # ---- 通过 robosuite step 执行目标位姿，并读取控制诊断 ----
            pose_action = np.concatenate([frame.pose.position, frame.pose.quaternion_wxyz])
            step_result = simulation.step(pose_action)
            observation = step_result[0]
            step_info = step_result[3]
            joint_command = step_info["joint_command"]
            ik_residual = step_info["ik_residual_m"]

            # 计算安全关节命令对应的 TCP 位姿 (用于记录和校验)
            safe_command = simulation.pose_for_joint_position(joint_command)

            next_state = simulation.state()

            # ---- 计算跟踪误差 ----
            # 横向误差: 实际 TCP 到有限直线或圆弧中心线的最短距离。
            projection = seam.project(next_state.tcp.position, frame.seam_progress)
            cross_track = projection.distance_m

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
            recorder.append_state(
                timestamp,
                next_state,
                simulation.images_from_observation(observation),
            )
            state = next_state

        # --- 阶段 6: 计算质量指标, 构建元数据, 完成录制 ---
        provisional = {name: np.asarray(values) for name, values in recorder.arrays.items()}
        report = report_from_arrays(provisional, config.quality, recovery)

        metadata = {
            "seed": seed,
            "robot_model": config.robot.model_id,
            "asset_id": simulation.workpiece.asset_id,
            "seam_id": seam.seam_id,
            "instruction": config.task.instruction,
            "direction": config.task.direction,
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
                "approach_speed_mps": config.task.approach_speed_mps,
                "speed_mps": config.task.speed_mps,
                "retreat_speed_mps": config.task.retreat_speed_mps,
                "orientation_follow_ratio": config.task.orientation_follow_ratio,
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
            "workpiece_position": simulation.mj_model.body_pos[simulation.workpiece_id].tolist(),
            "workpiece_quaternion_wxyz": simulation.mj_model.body_quat[
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
