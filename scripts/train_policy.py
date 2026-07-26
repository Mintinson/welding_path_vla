#!/usr/bin/env python3
"""根据统一 YAML 启动 LeRobot 策略训练。"""

from dataclasses import dataclass

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.training import TrainingRequest


@dataclass
class TrainArguments(AppConfig):
    """策略配置和非破坏性预览开关。"""

    dry_run: bool = False


@cli
def main(config: TrainArguments) -> None:
    """预览训练计划或启动训练。"""
    request = TrainingRequest(config.policy, config.training)
    if config.dry_run:
        output_json(request.plan())
        return
    print(f"training output: {request.run()}")


if __name__ == "__main__":
    main()
