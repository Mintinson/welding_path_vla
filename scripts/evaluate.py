#!/usr/bin/env python3
"""评估单条 episode 或聚合原始数据集。"""

from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from welding_path_vla.core.config import AppConfig
from welding_path_vla.evaluation.adapters import (
    trace_from_raw_episode,
    trace_from_real_robot_log,
)
from welding_path_vla.evaluation.evaluator import aggregate_reports, evaluate_trace


@dataclass
class EvaluationArguments(AppConfig):
    """论文指标和待评估数据的统一配置。"""

    mode: str = "episode"
    episode: Path | None = None
    source: str = "raw"
    assume_reference_task: bool = False
    output: Path | None = None


def evaluate_episode(config: EvaluationArguments) -> dict[str, object]:
    """评估一条仿真或真机 episode。"""
    if config.episode is None:
        raise ValueError("episode is required in episode mode")
    trace = (
        trace_from_real_robot_log(config.episode)
        if config.source == "real"
        else trace_from_raw_episode(
            config.episode,
            config,
            assume_reference_task=config.assume_reference_task,
        )
    )
    return evaluate_trace(trace, config.evaluation).as_dict()


def evaluate_dataset(config: EvaluationArguments) -> dict[str, object]:
    """聚合配置中原始数据集的评估结果。"""
    paths = sorted((Path(config.collection.dataset_root) / "episodes").glob("episode_*"))
    reports = [
        evaluate_trace(
            trace_from_raw_episode(path, config, config.assume_reference_task),
            config.evaluation,
        )
        for path in paths
    ]
    return aggregate_reports(reports).as_dict()


@cli
def main(config: EvaluationArguments) -> None:
    """根据 mode 运行单条或数据集评估。"""
    if config.mode not in {"episode", "dataset"} or config.source not in {"raw", "real"}:
        raise ValueError("mode/source must be episode|dataset and raw|real")
    report = evaluate_episode(config) if config.mode == "episode" else evaluate_dataset(config)
    output_json(report, config.output)


if __name__ == "__main__":
    main()
