#!/usr/bin/env python3
"""打开 MuJoCo 交互窗口检查 robosuite 焊接场景。"""

import os
import time

import mujoco.viewer
import numpy as np
from common import cli

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation import WeldingEnv


@cli
def main(config: AppConfig) -> None:
    """显示由配置和固定随机种子确定的场景。"""
    if os.environ.get("DISPLAY"):
        import glfw

        # Wayland 不支持 MuJoCo viewer 使用的窗口位置接口，并可能触发
        # libdecor / OpenGL 0x502 提示；桌面环境已有 XWayland，显式使用 X11。
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)

    simulation = WeldingEnv(config, seed=0, camera_observations=False)
    simulation.randomize_workpiece(np.random.default_rng(0))
    with mujoco.viewer.launch_passive(simulation.mj_model, simulation.mj_data) as viewer:
        viewer.opt.geomgroup[0] = 0
        viewer.opt.sitegroup[5] = 0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    simulation.close()


if __name__ == "__main__":
    main()
