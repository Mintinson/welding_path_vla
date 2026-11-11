"""在 LeRobot 原生 rollout 中复用焊接模型的整块推理与 ServoP。"""

from time import perf_counter

import torch
from lerobot.rollout import BaseStrategyConfig
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.inference.base import InferenceEngine
from lerobot.scripts import lerobot_rollout

from welding_path_vla.dataset.actions import ABSOLUTE_ACTION_NAMES
from welding_path_vla.policies.action_processors import (
    RelativeEEActionsProcessorStep,
    connect_relative_processors,
)
from welding_path_vla.policies.base import Observation
from welding_path_vla.policies.factory import get_policy_pipeline
from welding_path_vla.policies.runtime import LeRobotRuntime
from welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen import TrajVLAQwenConfig
from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
    TrajectoryVLAConfig,
)
from welding_path_vla.policies.vla_adapter.configuration_vla_adapter import VLAAdapterConfig
from welding_path_vla.policies.welding_prompt import configure_welding_prompt_builder

from .adapter import ElfinWelding, ElfinWeldingConfig

native_context_builder = lerobot_rollout.build_rollout_context


class WeldingInferenceEngine(InferenceEngine):
    """适配官方观测格式，逐点消费项目运行时的绝对动作队列。"""

    def __init__(self, runtime: LeRobotRuntime, task: str, robot: ElfinWelding, keys: list[str]):
        self.runtime, self.task, self.robot = runtime, task, robot
        self.order = [ABSOLUTE_ACTION_NAMES.index(key.removesuffix(".pos")) for key in keys]

    def start(self) -> None:
        """推理在官方调用线程内运行。"""

    def stop(self) -> None:
        """ServoP 由机器人的断开流程停止。"""

    def reset(self) -> None:
        self.runtime.reset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None
        observation = Observation.from_state(
            perf_counter(),
            {name: obs_frame[f"observation.images.{name}"] for name in self.robot.cameras},
            obs_frame["observation.state"],
            self.robot.task,
            self.task,
        )
        return torch.from_numpy(self.runtime.select_action(observation)[self.order])


def build_elfin_rollout_context(config, shutdown_event):
    """沿用官方 context，仅为 Elfin 接好整块解码和高频伺服。"""
    if not isinstance(config.robot, ElfinWeldingConfig):
        return native_context_builder(config, shutdown_event)
    if not config.robot.active:
        raise ValueError("Elfin rollout 需要 --robot.active=true")
    if not isinstance(config.strategy, BaseStrategyConfig) or not isinstance(
        config.inference, SyncInferenceConfig
    ):
        raise ValueError("Elfin rollout 请使用 --strategy.type=base --inference.type=sync")
    if config.interpolation_multiplier != 1:
        raise ValueError("Elfin 已在 ServoP 线程插值, 请使用 --interpolation_multiplier=1")

    ctx = native_context_builder(config, shutdown_event)
    robot = ctx.hardware.robot_wrapper.inner
    try:
        policy = ctx.policy.policy
        preprocessor, postprocessor = ctx.policy.preprocessor, ctx.policy.postprocessor
        if isinstance(
            policy.config, (TrajectoryVLAConfig, TrajVLAQwenConfig, VLAAdapterConfig)
        ) or any(isinstance(step, RelativeEEActionsProcessorStep) for step in preprocessor.steps):
            connect_relative_processors(preprocessor, postprocessor, require_saved=True)
            configure_welding_prompt_builder(preprocessor)
            prepare = getattr(policy, "prepare_for_inference", None)
            if callable(prepare):
                prepare()
            spec = get_policy_pipeline(policy.config.type).spec
            runtime = LeRobotRuntime(policy, preprocessor, postprocessor, config.device, spec)
            ctx.policy.inference = WeldingInferenceEngine(
                runtime, config.task, robot, ctx.data.ordered_action_keys
            )
        # 官方回初始位按同名关节键构造动作，Elfin 的观测和 TCP 动作键不同。
        ctx.hardware.initial_position = None
        robot.start_servo(config.fps)
        return ctx
    except BaseException:
        robot.disconnect()
        raise


def install_rollout_support() -> None:
    """让 LeRobot 0.6.0 的原生 CLI 采用 Elfin 兼容入口。"""
    lerobot_rollout.build_rollout_context = build_elfin_rollout_context
