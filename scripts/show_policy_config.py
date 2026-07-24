#!/usr/bin/env python3
"""显示策略、训练和部署配置。"""

from __future__ import annotations

import argparse

from common import load_config, output_json

from welding_path_vla.core.config import DEFAULT_CONFIG


def main() -> None:
    """加载配置并显示训练和部署就绪状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    resolved = config.as_dict()
    output_json(
        {
            "policy": resolved["policy"],
            "training": resolved["training"],
            "deployment": resolved["deployment"],
            "training_ready": bool(config.training.dataset_repo_id),
            "deployment_ready": bool(config.policy.checkpoint),
        }
    )


if __name__ == "__main__":
    main()
