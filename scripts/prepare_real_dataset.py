#!/usr/bin/env python3
"""整理原生录制的真机 LeRobot 数据，使其兼容现有 relative action 训练。"""

from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.dataset.real_capture import prepare_real_dataset


@dataclass
class PrepareRealArguments(AppConfig):
    """真机数据路径及复用的策略动作配置。"""

    dataset: Path = Path("datasets/real/elfin_demo")
    require_task_labels: bool = True
    reference_dataset: Path | None = None


@cli
def main(config: PrepareRealArguments) -> None:
    """补齐元信息和统计；不接触机器人，也不重写视频。"""
    output_json(
        prepare_real_dataset(
            config.dataset,
            config.policy.action_horizon,
            config.policy.action_stride,
            config.require_task_labels,
            config.reference_dataset,
        )
    )


if __name__ == "__main__":
    main()
