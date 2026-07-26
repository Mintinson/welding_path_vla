#!/usr/bin/env python3
"""显示策略、训练和部署配置。"""

from common import cli, output_json

from welding_path_vla.core.config import AppConfig


@cli
def main(config: AppConfig) -> None:
    """输出策略相关配置和就绪状态。"""
    resolved = config.as_dict()
    output_json(
        {
            "policy": resolved["policy"],
            "training": resolved["training"],
            "policy_evaluation": resolved["policy_evaluation"],
            "deployment": resolved["deployment"],
            "training_ready": bool(config.training.dataset_repo_id),
            "deployment_ready": bool(config.policy.checkpoint),
        }
    )


if __name__ == "__main__":
    main()
