"""Trajectory-VLA 在 robosuite 环境中的闭环部署入口。"""

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.simulation_rollout import SimulationRolloutReport, deploy_episodes
from welding_path_vla.policies.trajectory_vla.runtime import TrajectoryVLARuntime


def deploy_simulation(config: AppConfig, checkpoint: str) -> list[SimulationRolloutReport]:
    """加载一次模型并连续运行多个带诊断信息的 episode。"""
    runtime = TrajectoryVLARuntime.from_pretrained(checkpoint, config.policy.device)
    return deploy_episodes(config, runtime)


__all__ = ["deploy_simulation"]
