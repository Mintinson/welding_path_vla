#!/usr/bin/env python3
"""运行一圈八字或 LeRobot checkpoint，复用同一 ServoP 链路。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import yaml
from common import cli, output_json
from draccus.parsers.decoding import decode
from lerobot.robots.config import RobotConfig
from lerobot_robot_welding import BlackCamera, ElfinWelding

from welding_path_vla.core.config import AppConfig, PolicyConfig
from welding_path_vla.core.geometry import (
    quaternion_to_matrix,
    rotation_to_6d_rows,
)
from welding_path_vla.dataset.actions import ABSOLUTE_ACTION_NAMES
from welding_path_vla.policies.base import Observation
from welding_path_vla.robot.cps_servo import SERVO_LOOKAHEAD_S


@dataclass
class ElfinControlConfig:
    """连接地址默认取录制配置，execute 显式开启运动。"""

    record_config: Path = Path("configs/real_record.yaml")
    app_config: Path | None = None
    checkpoint: str | None = None
    host: str | None = None
    execute: bool = False
    duration_s: float = 26.0
    task: str | None = None


def figure_eight_offsets(steps: int) -> np.ndarray:
    """按弧长采样世界系八字，X/Y/Z 总行程为 170/180/100 mm。"""
    phase = np.linspace(0, 2 * np.pi, 4097)
    curve = np.column_stack(
        (0.085 * np.sin(phase), 0.090 * np.sin(2 * phase), 0.050 * (np.cos(3 * phase) - 1))
    )
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    fraction = np.linspace(0, 1, steps + 1)
    # 两端各用 10% 时间平滑起停，其余部分按等弧长速度采样。
    ramp = 0.1
    half = np.minimum(fraction, 1 - fraction)
    edge = np.minimum(half, ramp)
    distance = (edge - ramp / np.pi * np.sin(np.pi * edge / ramp)) / 2
    distance += np.maximum(half - ramp, 0)
    progress = np.where(fraction <= 0.5, distance, 1 - ramp - distance) / (1 - ramp)
    return np.column_stack([np.interp(progress * lengths[-1], lengths, axis) for axis in curve.T])


def figure_eight_policy(
    start: np.ndarray, steps: int, policy: PolicyConfig
) -> Callable[[Observation], np.ndarray]:
    """生成固定轨迹的世界系动作块，返回与 checkpoint 相同的可执行前缀。"""
    positions = start[:3] + figure_eight_offsets(steps)
    plan = np.column_stack((positions, np.broadcast_to(start[3:], (len(positions), 6))))
    first = 0

    def predict(observation: Observation) -> np.ndarray:
        nonlocal first
        indices = np.minimum(np.arange(first + 1, first + 1 + policy.action_horizon), steps)
        first += policy.action_steps
        return plan[indices][: policy.action_steps]

    return predict


@cli
def main(config: ElfinControlConfig) -> None:
    """从录制配置连接设备，从项目配置选择动作块节拍和策略。"""
    settings = yaml.safe_load(config.record_config.read_text(encoding="utf-8"))
    recorded = decode(RobotConfig, settings["robot"])
    app = AppConfig.load(config.app_config or recorded.scene_config)
    robot = ElfinWelding(replace(recorded, host=config.host or recorded.host, active=True))
    robot.task = app.task
    task = config.task or settings["dataset"]["single_task"]
    action_hz = recorded.fps / app.policy.action_stride
    steps = round(config.duration_s * action_hz)
    checkpoint = config.checkpoint or app.policy.checkpoint
    horizon = app.policy.action_horizon
    if checkpoint:
        from welding_path_vla.policies.factory import get_policy_pipeline
        from welding_path_vla.policies.runtime import LeRobotRuntime

        runtime = LeRobotRuntime.from_pretrained(
            checkpoint,
            app.policy.device,
            get_policy_pipeline(app.policy.family).spec,
            action_steps=app.policy.action_steps,
            inference_steps=app.deployment.inference_steps,
        )
        runtime.reset()
        horizon = runtime.policy.config.chunk_size

    with ThreadPoolExecutor(max_workers=1) as pool, robot:
        initial = robot.policy_observation(task)
        position, quaternion = initial.state[6:9], initial.state[9:13]
        start = np.r_[position, rotation_to_6d_rows(quaternion_to_matrix(quaternion))]
        predict = (
            runtime.predict_absolute_chunk
            if checkpoint
            else figure_eight_policy(start, steps, app.policy)
        )
        if config.execute:
            robot.start_servo(action_hz)

        def predict_next() -> np.ndarray:
            return predict(robot.policy_observation(task))

        pending = pool.submit(predict_next)
        tick = perf_counter()
        played = chunks = 0
        while played < steps:
            targets = pending.result()[: steps - played]
            if not len(targets):
                raise ValueError("策略返回了空动作块")
            played += len(targets)
            if played < steps:
                pending = pool.submit(predict_next)
            for target in targets:
                sleep(max(0.0, tick - perf_counter()))
                if config.execute:
                    robot.send_action(dict(zip(ABSOLUTE_ACTION_NAMES, target, strict=True)))
                tick = max(tick + 1 / action_hz, perf_counter())
            chunks += 1
        if config.execute:
            robot.wait_for_actions()
            sleep(SERVO_LOOKAHEAD_S)
        else:
            sleep(max(0.0, tick - perf_counter()))
        output_json(
            {
                "host": robot.config.host,
                "execute": config.execute,
                "cameras": {
                    name: "black" if isinstance(camera, BlackCamera) else "device"
                    for name, camera in robot.cameras.items()
                },
                "task": initial.instruction,
                "state_size": initial.state.size,
                "checkpoint": checkpoint,
                "chunks": chunks,
                "action_hz": action_hz,
                "action_horizon": horizon,
                "action_steps": app.policy.action_steps,
            }
        )


if __name__ == "__main__":
    main()
