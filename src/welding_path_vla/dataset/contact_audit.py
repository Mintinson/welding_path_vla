"""以未过滤的 MuJoCo 几何接触审计执行状态及训练动作标签。"""

from __future__ import annotations

import json
from dataclasses import fields
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.dataset.actions import build_absolute_actions
from welding_path_vla.dataset.raw_schema import EpisodeReader


def physical_config(metadata: dict) -> str:
    """提取历史物理配置，忽略与离线几何检查无关的训练、部署字段。"""
    defaults = AppConfig()
    sections = ("robot", "scene", "camera", "timing", "workpiece", "task", "safety")
    values = {}
    for section in sections:
        saved = metadata["resolved_config"][section]
        allowed = {field.name for field in fields(type(getattr(defaults, section)))}
        unknown = saved.keys() - allowed
        if unknown:
            raise ValueError(f"unsupported historical {section} fields: {sorted(unknown)}")
        values[section] = saved
    return json.dumps(values, sort_keys=True)


@lru_cache(maxsize=16)
def audit_environment(config_json: str) -> Any:
    """缓存完整机器人/工件碰撞模型，不渲染、不积分、不沿用旧接触过滤。"""
    from welding_path_vla.simulation.welding_env import WeldingEnv

    config = AppConfig.from_dict(json.loads(config_json))
    environment = WeldingEnv(config, camera_observations=False, ignore_done=True)
    environment.reset()
    root = mujoco.mj_name2id(
        environment.mj_model, mujoco.mjtObj.mjOBJ_BODY, environment.robot_model.root_body
    )
    bodies = {root}
    for body in range(root + 1, environment.mj_model.nbody):
        if environment.mj_model.body_parentid[body] in bodies:
            bodies.add(body)
    environment.audit_robot_geoms = {
        geom
        for geom in range(environment.mj_model.ngeom)
        if environment.mj_model.geom_bodyid[geom] in bodies
    }
    return environment


def robot_workpiece_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    workpiece_geoms: set[int],
    robot_geoms: set[int],
) -> list[tuple[str, str, float]]:
    """返回机器人与工件间 dist <= 0 的全部接触，深度单位为米。

    机器人集合包含底座及全部子刚体，排除工件与桌面接触。
    不使用旧环境的力阈值、穿入深度阈值或允许接触白名单。
    """
    contacts = []
    for contact in data.contact:
        first, second = int(contact.geom1), int(contact.geom2)
        if contact.dist > 0 or (first in workpiece_geoms) == (second in workpiece_geoms):
            continue
        robot, workpiece = (second, first) if first in workpiece_geoms else (first, second)
        if robot not in robot_geoms:
            continue
        contacts.append((model.geom(robot).name, model.geom(workpiece).name, -float(contact.dist)))
    return contacts


def inspect_configurations(environment: Any, joints: np.ndarray, targets=None) -> dict:
    """逐个关节配置执行运动学与碰撞检测，汇总命中帧、几何对和 FK 误差。"""
    model = environment.mj_model
    data = mujoco.MjData(model)
    indices, pairs, maximum_depth, fk_error = [], set(), 0.0, 0.0
    first_contact = None
    for index, joint_position in enumerate(joints):
        data.qpos[environment.qpos_ids] = joint_position
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        contacts = robot_workpiece_contacts(
            model, data, environment.workpiece_geom_ids, environment.audit_robot_geoms
        )
        if contacts:
            indices.append(index)
            pairs.update((robot, workpiece) for robot, workpiece, depth in contacts)
            maximum_depth = max(maximum_depth, *(depth for robot, workpiece, depth in contacts))
            if first_contact is None:
                first_contact = {"frame": index, "contacts": contacts}
        if targets is not None:
            fk_error = max(
                fk_error, float(np.linalg.norm(data.site_xpos[environment.tcp_id] - targets[index]))
            )
    return {
        "checked_frames": len(joints),
        "contact_frames": len(indices),
        "indices": indices,
        "pairs": sorted(pairs),
        "max_depth_m": maximum_depth,
        "first_contact": first_contact,
        "max_fk_position_error_m": fk_error,
    }


def audit_episode(job: tuple[dict, np.ndarray, np.ndarray]) -> dict:
    """核实导出帧与 raw 完全对应后，分别检查执行状态与 safe_command 标签。"""
    entry, stored_states, stored_actions = job
    episode = EpisodeReader(entry["raw_path"])
    try:
        metadata, trajectory = episode.metadata, episode.trajectory
        source = metadata["resolved_config"]["policy"]["action_source"]
        if source != "safe_command":
            raise ValueError(f"joint command audit requires safe_command labels, got {source}")
        count = episode.action_count
        states = np.concatenate(
            [
                trajectory[name][:count]
                for name in ("joint_position", "tcp_position", "tcp_quaternion_wxyz")
            ],
            axis=1,
        )
        actions = build_absolute_actions(episode, source)
        if not np.array_equal(states.astype(np.float32), stored_states):
            raise ValueError(f"raw/export state mismatch: {entry['raw_path']}")
        if not np.array_equal(actions, stored_actions):
            raise ValueError(f"raw/export action mismatch: {entry['raw_path']}")
        if metadata["instruction"] != entry["instruction"] or count != entry["frames"]:
            raise ValueError(f"raw/export episode metadata mismatch: {entry['raw_path']}")
        environment = audit_environment(physical_config(metadata))
        environment.mj_model.body_pos[environment.workpiece_id] = metadata["workpiece_position"]
        environment.mj_model.body_quat[environment.workpiece_id] = metadata[
            "workpiece_quaternion_wxyz"
        ]
        executed = inspect_configurations(
            environment, trajectory["joint_position"], trajectory["tcp_position"]
        )
        labels = inspect_configurations(
            environment, trajectory["joint_position_command"], trajectory["safe_command_position"]
        )
        # joint_position_command 必须确实实现标签位置；否则不能用它代替标签进行检查。
        if labels["max_fk_position_error_m"] > 1e-6:
            raise ValueError(f"joint command does not realize labels: {entry['raw_path']}")
        phases = trajectory["phase"].astype(str)
        for result in (executed, labels):
            result["phase_contacts"] = {
                phase: sum(phases[min(index, count - 1)] == phase for index in result["indices"])
                for phase in sorted(set(phases))
            }
        return {
            **entry,
            "executed": executed,
            "labels": labels,
            "recorded_collision_frames": int(np.count_nonzero(trajectory["collision"])),
            "remove": bool(executed["contact_frames"] or labels["contact_frames"]),
        }
    finally:
        episode.trajectory.close()


def stored_episodes(source: Path):
    """流式读取官方 parquet，跨文件拼接 episode，不解码视频。"""
    import pyarrow.dataset as arrow_dataset

    dataset = arrow_dataset.dataset(sorted(map(str, (source / "data").rglob("*.parquet"))))
    current, state_parts, action_parts, offset = None, [], [], 0
    for batch in dataset.scanner(
        columns=["episode_index", "index", "observation.state", "action"], use_threads=False
    ).to_batches():
        indices = batch.column("index").to_numpy()
        if not np.array_equal(indices, np.arange(offset, offset + len(indices))):
            raise ValueError("source frames are not in contiguous global index order")
        offset += len(indices)
        episodes = batch.column("episode_index").to_numpy()
        states = batch.column("observation.state").flatten().to_numpy().reshape(len(episodes), -1)
        actions = batch.column("action").flatten().to_numpy().reshape(len(episodes), -1)
        boundaries = np.r_[0, np.flatnonzero(np.diff(episodes)) + 1, len(episodes)]
        for start, end in pairwise(boundaries):
            episode = int(episodes[start])
            if current is not None and episode != current:
                yield current, np.concatenate(state_parts), np.concatenate(action_parts)
                state_parts, action_parts = [], []
            current = episode
            state_parts.append(states[start:end])
            action_parts.append(actions[start:end])
    if current is not None:
        yield current, np.concatenate(state_parts), np.concatenate(action_parts)


def filtered_manifest(document: dict, removed: set[int]) -> dict:
    """按旧全局 episode 顺序过滤源清单，保留动作表示契约。"""
    filtered = json.loads(json.dumps(document))
    offset = 0
    for name, episodes in document["sources"].items():
        filtered["sources"][name] = [
            episode for index, episode in enumerate(episodes, offset) if index not in removed
        ]
        offset += len(episodes)
    return filtered
