#!/usr/bin/env python3
"""预生成所有模型共用的、专家全轨迹可达的闭环评价场景。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from welding_path_vla.core.config import AppConfig  # noqa: I001
from welding_path_vla.core.config_files import compose_config, merge_config
from welding_path_vla.simulation import WeldingEnv
from welding_path_vla.simulation.evaluation_scenes import SCENE_BANK_VERSION
from welding_path_vla.simulation.task_sampling import (
    sample_feasible_trajectory,
    sample_task_config,
)

def broaden_randomization(config: AppConfig, bright: bool) -> None:
    """扩大空间与任务范围，并使用清晰、均衡的评价照明。

    Args:
        config: 从训练基准与任务 YAML 组合得到的场景配置。
        bright: 真时使用更亮的保留域，否则使用更暗的保留域。
    """
    randomization = config.randomization
    randomization.lighting_intensity_range = (
        [1.15, 1.30]
        if bright
        else [1.00, 1.15]
    )
    # 单灯加随机方向会产生大面积欠曝；固定双灯，仍保留强度、色温、位置和方向随机性。
    randomization.lighting_count_range = [2, 2]
    for name in (
        "xy_m", "yaw_deg", "initial_tcp_m", "work_angle_range_deg",
        "travel_angle_range_deg", "tool_roll_range_deg", "arc_start_range_deg",
        "arc_sweep_range_deg", "speed_range_mps", "curve_amplitude_range_m",
        "curve_frequency_range", "seam_length_range_m", "trihedral_size_range_m",
        "workpiece_color_jitter", "workpiece_finish_jitter",
        "lighting_azimuth_range_deg", "lighting_elevation_range_deg",
        "background_color_jitter",
    ):
        setattr(randomization, name, getattr(randomization, name) * 1.2)
    randomization.joint_degs = [value * 1.1 for value in randomization.joint_degs]
    randomization.max_sampling_attempts = max(30, randomization.max_sampling_attempts)
    randomization.task_group_size = 1
    config.validate()


def save_bank(path: Path, bank: dict) -> None:
    """原子写入清单，使中断后可从已完成场景继续。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """为每个任务采样 25 个互不重叠的保留域可行场景。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--task-dir", default="configs/tasks")
    parser.add_argument("--output", default="configs/evaluation_scenes_v2.json")
    parser.add_argument("--per-task", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--tasks", nargs="*", help="只生成指定任务: 用于小规模验证")
    args = parser.parse_args()
    base = AppConfig.load(args.base_config)
    task_paths = sorted(Path(args.task_dir).glob("*.yaml"))
    if args.tasks:
        task_paths = [path for path in task_paths if path.stem in args.tasks]
    if not task_paths or args.per_task < 1 or args.max_candidates < 1:
        raise ValueError("task list, per-task count and max-candidates must be positive")
    output = Path(args.output)
    if output.exists():
        bank = json.loads(output.read_text(encoding="utf-8"))
        if bank.get("version") != SCENE_BANK_VERSION or bank.get("complete"):
            raise ValueError(f"existing scene bank cannot be resumed: {output}")
    else:
        bank = {
            "version": SCENE_BANK_VERSION,
            "complete": False,
            "scenes": [],
        }
    existing = {(scene["task"]["task_id"], scene["episode"]) for scene in bank["scenes"]}

    for task_index, task_path in enumerate(task_paths):
        task_base = AppConfig.from_dict(merge_config(base.as_dict(), compose_config(task_path)))
        for episode in range(args.per_task):
            if (task_path.stem, episode) in existing:
                continue
            last_error = ""
            for candidate in range(args.max_candidates):
                seed = args.seed + task_index * 1_000_000 + episode * 1_000 + candidate
                config = AppConfig.from_dict(task_base.as_dict())
                broaden_randomization(config, bright=episode % 2 == 0)
                direction = "forward" if episode % 2 == 0 else "reverse"
                config = sample_task_config(config, np.random.default_rng(seed + 17), direction)
                simulation = None
                try:
                    simulation = WeldingEnv(config, seed)
                    sample = sample_feasible_trajectory(
                        simulation, config, np.random.default_rng(seed)
                    )
                    if not sample.initial_tcp_offset_applied:
                        raise RuntimeError("initial TCP offset was not applied")
                    state = simulation.scene_state()
                    scene = {
                        "task": state["task"],
                        "episode": episode,
                        "scene": state["scene"],
                        "robot": state["robot"],
                        "workpiece": state["workpiece"],
                    }
                    bank["scenes"].append(scene)
                    save_bank(output, bank)
                    print(f"{task_path.stem} {episode + 1}/{args.per_task} seed={seed}", flush=True)
                    break
                except (RuntimeError, ValueError) as error:
                    last_error = str(error)
                finally:
                    if simulation is not None:
                        simulation.close()
            else:
                raise RuntimeError(
                    f"no feasible scene for {task_path.stem} episode {episode} "
                    f"after {args.max_candidates} candidate seeds: {last_error}"
                )
    bank["scenes"].sort(key=lambda scene: (scene["task"]["task_id"], scene["episode"]))
    bank["complete"] = True
    save_bank(output, bank)
    print(f"Wrote {len(bank['scenes'])} feasible scenes to {output}", flush=True)


if __name__ == "__main__":
    main()
