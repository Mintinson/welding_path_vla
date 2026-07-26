#!/usr/bin/env python3
"""在 LeRobot 留出 episode 上离线评估策略 checkpoint。"""

from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.factory import get_policy_pipeline


@dataclass
class PolicyEvaluationArguments(AppConfig):
    """策略评估配置与报告路径。"""

    output: Path | None = None


@cli
def main(config: PolicyEvaluationArguments) -> None:
    """加载对应策略 pipeline 并输出可复查的 JSON 指标。"""
    if config.policy.checkpoint is None:
        raise ValueError("policy.checkpoint is required")
    report = get_policy_pipeline(config.policy.family).evaluate(config, config.policy.checkpoint)
    output_json(report.as_dict(), config.output)


if __name__ == "__main__":
    main()
