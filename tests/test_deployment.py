import json
import re
from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies import simulation_rollout
from welding_path_vla.policies.simulation_rollout import (
    SimulationRolloutReport,
    checkpoint_experiment_name,
    deploy_episodes,
    deployment_output_dir,
    deployment_task_configs,
    rollout_metric_summary,
)
from welding_path_vla.simulation.task_sampling import TaskSamplingError


def test_checkpoint_experiment_name_supports_numbered_checkpoints() -> None:
    """输出目录应从 numbered checkpoint 提取训练实验名。"""
    checkpoint = "outputs/train_v2/xvla_l_joint/checkpoints/0040000/pretrained_model"
    assert checkpoint_experiment_name(checkpoint) == "xvla_l_joint"


def test_deployment_directory_is_derived_from_checkpoint_and_minute() -> None:
    """输出根目录应使用 checkpoint 的训练实验名和分钟级时间。"""
    config = AppConfig.load("configs/deploy/smolvla_pipe_top.yaml")
    output = deployment_output_dir(config)
    assert output.parent == Path("outputs/deploy") / "smolvla_weldpath_relative_v1"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}", output.name)


def test_deployment_can_load_all_task_modules() -> None:
    """批量部署应复用一个策略配置并展开所有任务 YAML。"""
    config = AppConfig.load("configs/deploy/trajectory_vla.yaml")
    config.deployment.mode = "random"
    tasks = deployment_task_configs(config)

    assert [item.task.task_id for item in tasks] == [
        "curve_plate",
        "l_joint",
        "pipe_bottom",
        "pipe_top",
        "trihedral_horizontal",
        "trihedral_vertical",
    ]
    assert all(item.policy.family == "trajectory_vla" for item in tasks)
    assert all(item.policy.checkpoint == config.policy.checkpoint for item in tasks)
    assert len({deployment_output_dir(item) for item in tasks}) == 1
    assert (
        next(item for item in tasks if item.task.task_id == "pipe_top").deployment.max_steps == 3300
    )


def test_deployment_include_tasks_accepts_task_ids() -> None:
    """部署过滤器只接受任务 ID。"""
    config = AppConfig.load("configs/deploy/trajectory_vla.yaml")
    config.deployment.include_tasks = ["pipe_top"]
    assert [item.task.task_id for item in deployment_task_configs(config)] == ["pipe_top"]


def test_scene_include_tasks_filters_scene_without_switching_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scene 模式传入 include_tasks 时只筛选场景，不切换为 random。"""
    config = AppConfig.load("configs/deploy/smolvla.yaml")
    config.policy.checkpoint = "checkpoint"
    config.deployment.mode = "scene"
    config.deployment.include_tasks = ["pipe_top"]
    config.deployment.output_root = str(tmp_path)
    path = tmp_path / "scenes.json"
    scene_config = config.as_dict()
    scenes = [
        {
            "task": {**scene_config["task"], "task_id": task},
            "episode": 0,
            "scene": {},
            "robot": {},
            "workpiece": {},
        }
        for task in ("l_joint", "pipe_top")
    ]
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "complete": True,
                "scenes": scenes,
            }
        ),
        encoding="utf-8",
    )
    config.deployment.scene_bank_path = str(path)
    calls: list[tuple[str, list[str], list[str]]] = []

    def fake_deploy_task_episodes(task_config, runtime, selected_scenes, run_root):
        calls.append(
            (
                task_config.deployment.mode,
                task_config.deployment.include_tasks,
                [scene["task"]["task_id"] for scene in selected_scenes],
            )
        )
        return []

    monkeypatch.setattr(simulation_rollout, "deploy_task_episodes", fake_deploy_task_episodes)
    monkeypatch.setattr(
        simulation_rollout,
        "deployment_task_configs",
        lambda config: pytest.fail("scene mode must not build random task configs"),
    )

    assert simulation_rollout.deploy_episodes(config, object()) == []
    assert calls == [("scene", ["pipe_top"], ["pipe_top"])]


def test_all_tasks_write_subdirectories_and_combined_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量部署应为每个任务建子目录，并在公共根目录写跨任务指标。"""
    config = AppConfig.load("configs/deploy/smolvla.yaml")
    config.deployment.mode = "random"
    config.deployment.output_root = str(tmp_path)
    config.deployment.episodes = 1

    def fake_rollout(
        task_config: AppConfig,
        runtime: object,
        episode: int,
        output: Path,
    ) -> SimulationRolloutReport:
        output.mkdir(parents=True)
        return SimulationRolloutReport(
            task=task_config.task.task_id,
            episode=episode,
            seed=task_config.deployment.seed,
            steps=1,
            completed=True,
            termination_reason="completed",
            collision=False,
            trace_path=str(output / "rollout.npz"),
            videos=(),
            diagnostics={},
            evaluation=None,
        )

    monkeypatch.setattr(simulation_rollout, "rollout_episode", fake_rollout)
    reports = deploy_episodes(config, object())

    assert len(reports) == 6
    root = next((tmp_path / "smolvla_weldpath_relative_v1").iterdir())
    assert (root / "metrics_summary.json").exists()
    assert not (root / "summary.json").exists()
    assert all(not (root / report.task / "metrics_summary.json").exists() for report in reports)
    assert all(not (root / report.task / "summary.json").exists() for report in reports)


def test_unreachable_staging_does_not_abort_remaining_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record scene setup failures separately and continue the simulation benchmark."""
    config = AppConfig.load("configs/xvla_a100.yaml")
    config.deployment.mode = "random"
    config.deployment.output_root = str(tmp_path)
    config.deployment.episodes = 2

    def fake_rollout(
        task_config: AppConfig,
        runtime: object,
        episode: int,
        output: Path,
    ) -> SimulationRolloutReport:
        output.mkdir()
        if episode == 0:
            raise TaskSamplingError("unreachable staging pose")
        return SimulationRolloutReport(
            task=task_config.task.task_id,
            episode=episode,
            seed=task_config.deployment.seed + episode,
            steps=1,
            completed=True,
            termination_reason="completed",
            collision=False,
            trace_path="",
            videos=(),
            diagnostics={},
            evaluation=None,
        )

    monkeypatch.setattr(simulation_rollout, "rollout_episode", fake_rollout)
    reports = simulation_rollout.deploy_task_episodes(config, object())
    assert [report.termination_reason for report in reports] == ["setup_failure", "completed"]
    root = deployment_output_dir(config) / config.task.task_id
    assert (
        root
        / "episode_000000"
        / "summary.json"
    ).exists()
    assert not (root / "summary.json").exists()
    assert not (root / "metrics_summary.json").exists()
    metrics = rollout_metric_summary(config, reports)
    assert metrics["metrics"]["complete_rate"] == 0.5


def test_fixed_scene_bank_uses_user_deployment_config_for_every_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scene bank 不包含 deployment，部署控制项应来自用户配置。"""
    model = AppConfig()
    model.policy.checkpoint = "checkpoint-a"
    model.deployment.output_root = str(tmp_path / "outputs")
    model.deployment.max_steps = 1
    model.evaluation.coverage_tolerance_m = 0.001
    path = tmp_path / "scenes.json"
    model.deployment.scene_bank_path = str(path)
    reference = AppConfig()
    reference.deployment.max_steps = 500
    reference.evaluation.coverage_tolerance_m = 0.005
    reference_values = reference.as_dict()
    scenes = [
        {
            "task": reference_values["task"],
            "episode": index,
            "scene": {
                "layout": reference_values["scene"],
                "camera": {"global_fovy_deg": reference.camera.global_fovy_deg},
                "lighting": {},
                "background": {},
            },
            "robot": {"joint_position_rad": [0.0] * 6, "joint_velocity_rad_s": [0.0] * 6},
            "workpiece": {
                "geometry": reference_values["workpiece"],
                "pose": {
                    "position_m": reference_values["scene"]["workpiece_position_m"],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "surface": {},
            },
        }
        for index in range(2)
    ]
    path.write_text(
        json.dumps({"version": 2, "complete": True, "scenes": scenes}),
        encoding="utf-8",
    )
    seen = []

    def fake_rollout(config, runtime, episode, output, scene):
        output.mkdir(parents=True)
        seen.append((episode, config.deployment.max_steps, config.evaluation.coverage_tolerance_m))
        return SimulationRolloutReport(
            "l_joint",
            episode,
            model.deployment.seed + episode,
            1,
            True,
            "completed",
            False,
            "",
            (),
            {},
            {"success": episode == 0, "coverage_ratio": 0.9,
             "direction_ratio": 1.0, "reached_start": episode == 0,
             "tracking_frames": 1, "cte_p95_m": 0.001},
        )

    output_calls = 0

    def unique_output(config):
        nonlocal output_calls
        output_calls += 1
        return tmp_path / "outputs" / "checkpoint-a" / f"run-{output_calls}"

    monkeypatch.setattr(simulation_rollout, "rollout_episode", fake_rollout)
    monkeypatch.setattr(simulation_rollout, "deployment_output_dir", unique_output)
    reports = deploy_episodes(model, object())
    assert seen == [(0, 1, 0.001), (1, 1, 0.001)]
    assert len(reports) == 2
    experiment = tmp_path / "outputs" / "checkpoint-a"
    run_roots = list(experiment.iterdir())
    assert len(run_roots) == 1
    assert output_calls == 1
    root = run_roots[0]
    assert (root / "metrics_summary.json").exists()
    assert not (root / "l_joint" / "metrics_summary.json").exists()
    assert not (root / "summary.json").exists()
    assert not (root / "l_joint" / "summary.json").exists()
    summary = json.loads((root / "metrics_summary.json").read_text())
    assert summary["metrics"]["success_rate"] == 0.5


def test_rollout_metric_summary_reports_arrival_timeout_and_tracking_cte() -> None:
    """批量指标使用全部 episode 分母，并单独报告到达率与超时率。"""
    reports = [
        SimulationRolloutReport(
            "l_joint", 0, 0, 10, True, "completed", False, "", (), {},
            {
                "success": True,
                "coverage_ratio": 1.0,
                "direction_ratio": 1.0,
                "reached_start": True,
                "tracking_frames": 8,
                "cte_p95_m": 0.002,
            },
        ),
        SimulationRolloutReport(
            "l_joint", 1, 1, 10, False, "timeout", False, "", (), {},
            {
                "success": False,
                "coverage_ratio": 0.0,
                "direction_ratio": 0.0,
                "reached_start": False,
                "tracking_frames": 0,
                "cte_p95_m": None,
            },
        ),
        SimulationRolloutReport(
            "l_joint", 2, 2, 0, False, "setup_failure", False, "", (), {}, None
        ),
        SimulationRolloutReport(
            "l_joint", 3, 3, 4, False, "collision", True, "", (), {},
            {
                "success": False,
                "coverage_ratio": 0.2,
                "direction_ratio": 0.5,
                "reached_start": True,
                "tracking_frames": 3,
                "cte_p95_m": 0.004,
            },
        ),
    ]

    config = AppConfig.load("configs/deploy/smolvla.yaml")
    summary = rollout_metric_summary(config, reports)

    assert list(summary) == [
        "checkpoint",
        "config_path",
        "metrics",
        "episode_groups",
    ]
    assert summary["checkpoint"] == config.policy.checkpoint
    assert summary["config_path"] == "configs/deploy/smolvla.yaml"
    metrics = summary["metrics"]
    assert metrics["episodes"] == 4
    assert metrics["arrival_episodes"] == 2
    assert metrics["arrival_rate"] == pytest.approx(1 / 2)
    assert metrics["timeout_episodes"] == 1
    assert metrics["timeout_rate"] == pytest.approx(1 / 4)
    assert metrics["collision_episodes"] == 1
    assert metrics["collision_rate"] == pytest.approx(1 / 4)
    assert metrics["coverage_ratio_mean"] == pytest.approx(0.3)
    assert metrics["cte_p95_mean_m"] == pytest.approx(0.003)
    assert metrics["cte_observed_episodes"] == 2
    assert summary["episode_groups"] == {
        "complete": [{"task": "l_joint", "episode": 0}],
        "success": [{"task": "l_joint", "episode": 0}],
        "timeout": [{"task": "l_joint", "episode": 1}],
        "collision": [{"task": "l_joint", "episode": 3}],
    }
