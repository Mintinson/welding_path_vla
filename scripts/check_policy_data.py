#!/usr/bin/env python3
"""检查训练数据能否被所选策略正确读取。"""

from dataclasses import asdict

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.act.data_adapter import validate_dataset


@cli
def main(config: AppConfig) -> None:
    """输出 ACT 所需 schema 与当前数据集规模。"""
    if config.policy.family != "act":
        raise ValueError("data check is currently implemented for ACT")
    output_json(asdict(validate_dataset(config.training)))


if __name__ == "__main__":
    main()
