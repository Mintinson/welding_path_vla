"""ACT 的 robosuite 闭环部署入口。"""

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.act.runtime import ACTRuntime
from welding_path_vla.policies.simulation_rollout import (
    SimulationRolloutReport,
    deploy_episodes,
    rollout_episode,
)


def deploy_simulation(config: AppConfig, checkpoint: str) -> list[SimulationRolloutReport]:
    """加载 ACT，并连续运行配置指定的仿真 episode。"""
    runtime = ACTRuntime.from_pretrained(checkpoint, config.policy.device)
    return deploy_episodes(config, runtime)


__all__ = [
    "SimulationRolloutReport",
    "deploy_simulation",
    "rollout_episode",
]
