"""仅从过去的机器人状态提取短时运动，不读取历史图像或未来动作。"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.converters import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE
from torch import Tensor

from welding_path_vla.core.geometry import quaternion_to_matrix, rotation_to_6d_rows

# 每段包含关节速度 6、世界系 TCP 速度 3、相对旋转变化率 6、时间间隔 1、有效位 1。
MOTION_INTERVAL_DIM = 17
MOTION_FEATURES = "observation.motion_features"
HISTORY_TIMES = "observation.state_history_times"
DEFAULT_HISTORY_OFFSETS = (-8 / 30, -4 / 30, 0.0)


def state_history_deltas(config: Any) -> dict[str, list[float]]:
    """只扩展状态采样；图像、动作和当前时间戳保持原来的采样契约。"""
    values = config if isinstance(config, dict) else vars(config)
    if not values.get("use_state_history", False):
        return {}
    return {OBS_STATE: list(values.get("state_history_offsets_s", DEFAULT_HISTORY_OFFSETS))}


@dataclass
@ProcessorStepRegistry.register(name="welding_state_history")
class StateHistoryProcessorStep(ProcessorStep):
    """在归一化和相对动作转换前提取运动，然后恢复当前帧的 13D 状态。

    Attributes:
        offsets_s: 等间隔采集数据中的历史采样时刻，单位秒，末项必须为零。
        velocity_scales: 关节、TCP 平移、旋转变化率的固定尺度，分别为 rad/s、m/s、1/s。
            仅用于改善输入量级，不裁剪速度；无需额外标签或数据集统计。
    """

    offsets_s: tuple[float, ...] = DEFAULT_HISTORY_OFFSETS
    velocity_scales: tuple[float, float, float] = (0.1, 0.01, 0.1)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """计算相邻历史采样点的速度；起始补帧和过大的在线时间间隔不参与计算。"""
        result = transition.copy()
        observation = dict(transition[TransitionKey.OBSERVATION])
        state = observation[OBS_STATE].float()
        # 单条离线样本为 H×13，DataLoader 和在线缓冲输出为 B×H×13。
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if state.ndim != 3 or state.shape[1:] != (len(self.offsets_s), 13):
            raise ValueError("state history requires BxHx13 raw states and matching offsets")
        offsets = state.new_tensor(self.offsets_s)
        times = observation.pop(HISTORY_TIMES, offsets.expand(state.shape[0], -1)).to(state)
        padding = transition[TransitionKey.COMPLEMENTARY_DATA][f"{OBS_STATE}_is_pad"]
        padding = padding.to(device=state.device, dtype=torch.bool).reshape(state.shape[:2])
        dt = times.diff(dim=-1)
        valid = ~padding[:, :-1] & ~padding[:, 1:] & (dt > 1e-6)
        valid &= dt <= 2 * offsets.diff() + 1e-6
        elapsed = dt.clamp_min(1e-6)[..., None]
        joints = state[:, 1:, :6] - state[:, :-1, :6]
        translation = state[:, 1:, 6:9] - state[:, :-1, 6:9]
        rotation = quaternion_to_matrix(state[..., 9:13])
        relative = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]
        identity = torch.eye(3, device=state.device, dtype=state.dtype)
        # 旋转矩阵消除 q/-q 歧义；这是 6D 旋转变化率，不冒充欧拉角速度。
        angular = rotation_to_6d_rows(relative) - rotation_to_6d_rows(identity)
        joint_scale, tcp_scale, rotation_scale = self.velocity_scales
        features = torch.cat(
            (
                joints / elapsed / joint_scale,
                translation / elapsed / tcp_scale,
                angular / elapsed / rotation_scale,
                dt[..., None],
                valid[..., None].float(),
            ),
            dim=-1,
        )
        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        observation[MOTION_FEATURES] = features.flatten(1)
        observation[OBS_STATE] = state[:, -1]
        result[TransitionKey.OBSERVATION] = observation
        return result

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        """历史特征已有物理尺度，不进入数据集 Normalizer；当前 state schema 不变。"""
        return features

    def get_config(self) -> dict[str, Any]:
        """把采样时刻与物理尺度一同保存，部署不依赖训练 YAML 的默认值。"""
        return asdict(self)


@dataclass
class RobotStateHistory:
    """在线因果状态缓冲，保存真实观测时间，动作块执行期间同样更新。"""

    offsets_s: tuple[float, ...]
    samples: deque[tuple[float, np.ndarray]] = field(default_factory=deque)

    def record(self, timestamp: float, state: np.ndarray) -> None:
        """记录观测副本；重复时间替换，时间回退视为新 episode。"""
        if self.samples and timestamp < self.samples[-1][0]:
            self.samples.clear()
        if self.samples and timestamp == self.samples[-1][0]:
            self.samples.pop()
        self.samples.append((timestamp, np.asarray(state, dtype=np.float32).copy()))
        # 留出一个最大采样间隔的余量，容纳异步实机观测的轻微抖动。
        oldest = timestamp + self.offsets_s[0] - max(np.diff(self.offsets_s))
        while len(self.samples) > 1 and self.samples[1][0] < oldest:
            self.samples.popleft()

    def observation(self) -> dict[str, Tensor]:
        """每个目标时刻取其之前最近的观测，缺失位置补首帧并标为无效。

        Returns:
            带 batch=1 的历史状态、相对当前帧的实际时间和 padding mask。
            时间先在 float64 中相减，避免 UNIX 时间戳转 FP32 丢失子秒精度。
        """
        times = np.asarray([sample[0] for sample in self.samples], dtype=np.float64)
        targets = times[-1] + np.asarray(self.offsets_s)
        indices = np.searchsorted(times, targets + 1e-7, side="right") - 1
        padded = indices < 0
        indices = indices.clip(0)
        states = np.stack([self.samples[index][1] for index in indices])
        return {
            OBS_STATE: torch.from_numpy(states)[None],
            HISTORY_TIMES: torch.tensor(times[indices] - times[-1], dtype=torch.float32)[None],
            f"{OBS_STATE}_is_pad": torch.from_numpy(padded)[None],
        }
