#!/usr/bin/env python3
"""打开 MuJoCo 交互窗口检查当前焊接场景。"""

import time

import numpy as np
from common import cli

from welding_path_vla.core.config import AppConfig


@cli
def main(config: AppConfig) -> None:
    """显示由配置和固定随机种子确定的场景。"""
    import mujoco.viewer

    from welding_path_vla.simulation import WeldingSimulation

    simulation = WeldingSimulation(config)
    simulation.randomize_workpiece(np.random.default_rng(0))
    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    simulation.close()


if __name__ == "__main__":
    main()
