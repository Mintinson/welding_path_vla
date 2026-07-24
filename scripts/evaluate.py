#!/usr/bin/env python3
"""评估单条 episode 或聚合原始数据集。"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, output_json

from welding_path_vla.core.config import DEFAULT_CONFIG
from welding_path_vla.evaluation.adapters import (
    trace_from_raw_episode,
    trace_from_real_robot_log,
)
from welding_path_vla.evaluation.evaluator import aggregate_reports, evaluate_trace


def evaluate_episode(arguments: argparse.Namespace) -> None:
    """评估一条仿真或真机 episode。"""
    config = load_config(arguments.config)
    trace = (
        trace_from_real_robot_log(arguments.episode)
        if arguments.source == "real"
        else trace_from_raw_episode(
            arguments.episode,
            config,
            assume_reference_task=arguments.assume_reference_task,
        )
    )
    output_json(evaluate_trace(trace, config.evaluation).as_dict(), arguments.output)


def evaluate_dataset(arguments: argparse.Namespace) -> None:
    """聚合原始仿真数据集的评估结果。"""
    config = load_config(arguments.config)
    paths = sorted((Path(arguments.dataset) / "episodes").glob("episode_*"))
    reports = [
        evaluate_trace(
            trace_from_raw_episode(path, config, arguments.assume_reference_task),
            config.evaluation,
        )
        for path in paths
    ]
    output_json(aggregate_reports(reports).as_dict(), arguments.output)


def parser() -> argparse.ArgumentParser:
    """创建 episode/dataset 两种评估子命令。"""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    episode = commands.add_parser("episode")
    episode.add_argument("--episode", required=True)
    episode.add_argument("--source", choices=("raw", "real"), default="raw")
    episode.add_argument("--config", default=DEFAULT_CONFIG)
    episode.add_argument("--assume-reference-task", action="store_true")
    episode.add_argument("--output")
    episode.set_defaults(handler=evaluate_episode)
    dataset = commands.add_parser("dataset")
    dataset.add_argument("--dataset", required=True)
    dataset.add_argument("--config", default=DEFAULT_CONFIG)
    dataset.add_argument("--assume-reference-task", action="store_true")
    dataset.add_argument("--output")
    dataset.set_defaults(handler=evaluate_dataset)
    return root


def main() -> None:
    """解析评估子命令并执行。"""
    arguments = parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
