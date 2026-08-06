#!/usr/bin/env python3
"""将已经转换完成的 LeRobot 数据集上传到 Hugging Face Hub。"""

from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from welding_path_vla.dataset.export_lerobot import upload_lerobot_dataset


@dataclass
class UploadArguments:
    """LeRobot Dataset Hub 上传参数。

    Attributes:
        dataset: 已完成转换的本地 LeRobot 数据集目录。
        repo_id: Hugging Face Dataset 仓库标识。
        private: 新建仓库时是否设为私有。
        attempts: 网络连接失败时的总尝试次数。
        retry_wait_s: 网络连接失败后的等待秒数。
    """

    dataset: Path = Path("datasets/weldpath_lerobot_relative_v1")
    repo_id: str = "huayan/weldpath_relative_v1"
    private: bool = True
    attempts: int = 10
    retry_wait_s: float = 60.0


@cli
def main(config: UploadArguments) -> None:
    """验证并上传现有数据集，不重新执行格式转换。"""
    hub_url = upload_lerobot_dataset(
        config.dataset,
        config.repo_id,
        private=config.private,
        attempts=config.attempts,
        retry_wait_s=config.retry_wait_s,
    )
    output_json({"dataset": str(config.dataset), "hub_url": hub_url})


if __name__ == "__main__":
    main()
