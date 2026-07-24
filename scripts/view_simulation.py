#!/usr/bin/env python3
"""打开 MuJoCo 交互窗口检查当前焊接场景。"""

from __future__ import annotations

import argparse
import time

import numpy as np
from common import load_config

from welding_path_vla.core.config import DEFAULT_CONFIG


def main() -> None:
    """加载配置并显示确定性随机场景。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    import mujoco.viewer

    from welding_path_vla.simulation import WeldingSimulation

    simulation = WeldingSimulation(load_config(arguments.config))
    simulation.randomize_workpiece(np.random.default_rng(0))
    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    simulation.close()


if __name__ == "__main__":
    main()
