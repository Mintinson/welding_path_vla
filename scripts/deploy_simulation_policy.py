#!/usr/bin/env python3
"""把训练好的策略部署到项目 MuJoCo 环境并记录评估轨迹。"""

import os

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.deployment import deploy_simulation


@cli
def main(config: AppConfig) -> None:
    """使用配置中的 checkpoint 执行策略 rollout。"""
    os.environ.setdefault("MUJOCO_GL", config.camera.offscreen_backend)
    reports = deploy_simulation(config)
    output_json([report.as_dict() for report in reports])


if __name__ == "__main__":
    main()
