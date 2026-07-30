"""SmolVLA 在 robosuite 焊接环境中的闭环部署入口。"""

from __future__ import annotations

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.simulation_rollout import SimulationRolloutReport, deploy_episodes
from welding_path_vla.policies.smolvla.runtime import SmolVLARuntime


def deploy_simulation(config: AppConfig, checkpoint: str) -> list[SimulationRolloutReport]:
    """加载一次 SmolVLA，并连续运行多个带诊断信息的仿真 episode。"""
    runtime = SmolVLARuntime.from_pretrained(checkpoint, config.policy.device)
    return deploy_episodes(config, runtime)


__all__ = ["deploy_simulation"]
