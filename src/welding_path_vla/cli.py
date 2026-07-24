"""CLI 入口: 定义焊接 VLA 项目的所有子命令及参数解析。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2

from welding_path_vla.config import DEFAULT_CONFIG, AppConfig
from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.evaluation.trajectory_metrics import validate_episode


def load_config(arguments: argparse.Namespace) -> AppConfig:
    """从命令行参数加载配置, 支持--dataset 覆盖数据根目录。

    Args:
        arguments: 命令行解析结果。

    Returns:
        解析后的应用配置对象。
    """
    config = AppConfig.load(arguments.config)
    if getattr(arguments, "dataset", None):
        config.collection.dataset_root = arguments.dataset
    return config


def output_json(value: object, output: str | None = None) -> None:
    """向终端或指定文件输出 JSON。"""
    document = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    else:
        print(document)


def sim_view(arguments: argparse.Namespace) -> None:
    """启动 MuJoCo 可视化窗口, 查看当前仿真场景。

    随机化工件位姿后进入被动查看器循环, 用于调试场景布局。

    Args:
        arguments: 命令行参数, 含 --config。
    """
    import mujoco.viewer

    from welding_path_vla.simulation import WeldingSimulation

    simulation = WeldingSimulation(load_config(arguments))
    simulation.randomize_workpiece(__import__("numpy").random.default_rng(0))
    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    simulation.close()


def sim_collect(arguments: argparse.Namespace) -> None:
    """在仿真环境中采集演示数据集。

    采集有效数据后对各 episode 执行质量校验并输出统计结果。

    Args:
        arguments: 命令行参数, 含 --config、--dataset、--episodes。
    """
    config = load_config(arguments)
    if config.collection.headless:
        os.environ.setdefault("MUJOCO_GL", config.camera.offscreen_backend)
    from welding_path_vla.simulation.collector import collect_dataset

    paths = collect_dataset(config, arguments.episodes)
    counts = Counter(validate_episode(path, config).status.value for path in paths)
    print(json.dumps({"episodes": len(paths), "status": counts}, ensure_ascii=False, indent=2))


def sim_replay(arguments: argparse.Namespace) -> None:
    """回放已采集 episode 的全局和腕部摄像头录像。

    按策略频率逐帧播放, 按 ESC 退出。

    Args:
        arguments: 命令行参数, 含 --episode。
    """
    episode = EpisodeReader(arguments.episode)
    global_video = cv2.VideoCapture(str(episode.path / "global.mp4"))
    wrist_video = cv2.VideoCapture(str(episode.path / "wrist.mp4"))
    delay = max(1, round(1000 / episode.metadata["resolved_config"]["timing"]["policy_hz"]))
    while True:
        global_ok, global_frame = global_video.read()
        wrist_ok, wrist_frame = wrist_video.read()
        if not global_ok or not wrist_ok:
            break
        cv2.imshow("global", global_frame)
        cv2.imshow("wrist", wrist_frame)
        if cv2.waitKey(delay) == 27:
            break
    global_video.release()
    wrist_video.release()
    cv2.destroyAllWindows()


def data_validate(arguments: argparse.Namespace) -> None:
    """校验数据集中的所有 episode, 输出有效/无效统计。

    Args:
        arguments: 命令行参数, 含 --dataset。
    """
    root = Path(arguments.dataset)
    reports = [validate_episode(path) for path in sorted((root / "episodes").glob("episode_*"))]
    result = {
        "episodes": len(reports),
        "valid": sum(report.valid for report in reports),
        "status": Counter(report.status.value for report in reports),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def data_export(arguments: argparse.Namespace) -> None:
    """将原始数据集导出为 LeRobot 格式。

    Args:
        arguments: 命令行参数, 含 --dataset、--output、--repo-id。
    """
    from welding_path_vla.dataset.export_lerobot import export_lerobot

    output = export_lerobot(arguments.dataset, arguments.output, arguments.repo_id)
    print(output)


def evaluation_episode(arguments: argparse.Namespace) -> None:
    """评估一条 raw 或真机 episode 并输出 JSON。"""
    from welding_path_vla.evaluation.adapters import (
        trace_from_raw_episode,
        trace_from_real_robot_log,
    )
    from welding_path_vla.evaluation.evaluator import evaluate_trace

    config = load_config(arguments)
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


def evaluation_dataset(arguments: argparse.Namespace) -> None:
    """聚合 raw 数据集的 ESR、ICR 和连续轨迹指标。"""
    from welding_path_vla.evaluation.adapters import trace_from_raw_episode
    from welding_path_vla.evaluation.evaluator import aggregate_reports, evaluate_trace

    config = load_config(arguments)
    paths = sorted((Path(arguments.dataset) / "episodes").glob("episode_*"))
    reports = [
        evaluate_trace(
            trace_from_raw_episode(path, config, arguments.assume_reference_task),
            config.evaluation,
        )
        for path in paths
    ]
    output_json(aggregate_reports(reports).as_dict(), arguments.output)


def robot_show_config(arguments: argparse.Namespace) -> None:
    """打印机器人相关配置 (模型、安装、安全参数)。

    Args:
        arguments: 命令行参数, 含 --config。
    """
    config = load_config(arguments)
    resolved = config.as_dict()
    print(
        json.dumps(
            {
                "robot": resolved["robot"],
                "robot_mount": {
                    "position_m": resolved["scene"]["robot_base_position_m"],
                    "yaw_deg": resolved["scene"]["robot_base_yaw_deg"],
                },
                "real_robot": resolved["real_robot"],
                "safety": resolved["safety"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def policy_show_config(arguments: argparse.Namespace) -> None:
    """打印策略和训练配置, 含训练/部署就绪状态。

    Args:
        arguments: 命令行参数, 含 --config。
    """
    config = load_config(arguments)
    resolved = config.as_dict()
    print(
        json.dumps(
            {
                "policy": resolved["policy"],
                "training": resolved["training"],
                "deployment": resolved["deployment"],
                "training_ready": bool(config.training.dataset_repo_id),
                "deployment_ready": bool(config.policy.checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def policy_train(arguments: argparse.Namespace) -> None:
    """启动策略训练流程, 支持 dry-run 预览命令。

    Args:
        arguments: 命令行参数, 含 --config、--dry-run。
    """
    from welding_path_vla.policies.training import TrainingRequest

    config = load_config(arguments)
    command = TrainingRequest(config.policy, config.training).command()
    if arguments.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


def parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器, 注册所有子命令。

    子命令分组: sim (view/collect/replay)、data (validate/export-lerobot)、
    robot (show-config)、policy (show-config/train)。

    Returns:
        配置完成的参数解析器。
    """
    root = argparse.ArgumentParser(prog="welding-vla")
    commands = root.add_subparsers(dest="group", required=True)
    sim = commands.add_parser("sim")
    sim_commands = sim.add_subparsers(dest="command", required=True)
    view = sim_commands.add_parser("view")
    view.add_argument("--config", default=DEFAULT_CONFIG)
    view.set_defaults(handler=sim_view)
    collect = sim_commands.add_parser("collect")
    collect.add_argument("--config", default=DEFAULT_CONFIG)
    collect.add_argument("--dataset")
    collect.add_argument("--episodes", type=int)
    collect.set_defaults(handler=sim_collect)
    replay = sim_commands.add_parser("replay")
    replay.add_argument("--episode", required=True)
    replay.set_defaults(handler=sim_replay)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="command", required=True)
    validate = data_commands.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.set_defaults(handler=data_validate)
    export = data_commands.add_parser("export-lerobot")
    export.add_argument("--dataset", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--repo-id", default="huayan/weldpath_sim_v1")
    export.set_defaults(handler=data_export)

    evaluation = commands.add_parser("evaluation")
    evaluation_commands = evaluation.add_subparsers(dest="command", required=True)
    episode_evaluation = evaluation_commands.add_parser("episode")
    episode_evaluation.add_argument("--episode", required=True)
    episode_evaluation.add_argument("--source", choices=("raw", "real"), default="raw")
    episode_evaluation.add_argument("--config", default=DEFAULT_CONFIG)
    episode_evaluation.add_argument("--assume-reference-task", action="store_true")
    episode_evaluation.add_argument("--output")
    episode_evaluation.set_defaults(handler=evaluation_episode)
    dataset_evaluation = evaluation_commands.add_parser("dataset")
    dataset_evaluation.add_argument("--dataset", required=True)
    dataset_evaluation.add_argument("--config", default=DEFAULT_CONFIG)
    dataset_evaluation.add_argument("--assume-reference-task", action="store_true")
    dataset_evaluation.add_argument("--output")
    dataset_evaluation.set_defaults(handler=evaluation_dataset)

    robot = commands.add_parser("robot")
    robot_commands = robot.add_subparsers(dest="command", required=True)
    robot_config = robot_commands.add_parser("show-config")
    robot_config.add_argument("--config", default=DEFAULT_CONFIG)
    robot_config.set_defaults(handler=robot_show_config)

    policy = commands.add_parser("policy")
    policy_commands = policy.add_subparsers(dest="command", required=True)
    policy_config = policy_commands.add_parser("show-config")
    policy_config.add_argument("--config", default=DEFAULT_CONFIG)
    policy_config.set_defaults(handler=policy_show_config)
    train = policy_commands.add_parser("train")
    train.add_argument("--config", default=DEFAULT_CONFIG)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(handler=policy_train)
    return root


def main() -> None:
    """CLI 入口: 解析参数并派发到对应 handler。"""
    arguments = parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
