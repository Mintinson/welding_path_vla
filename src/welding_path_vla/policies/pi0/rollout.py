"""π0 仿真部署入口。"""

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.pi0.runtime import load
from welding_path_vla.policies.simulation_rollout import SimulationRolloutReport, deploy_episodes


def deploy_simulation(config: AppConfig, checkpoint: str) -> list[SimulationRolloutReport]:
    """加载一次 π0，并连续执行多个带诊断信息的仿真 episode。"""
    return deploy_episodes(config, load(checkpoint, config.policy.device))


__all__ = ["deploy_simulation"]
