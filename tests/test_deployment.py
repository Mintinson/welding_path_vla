from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies import simulation_rollout
from welding_path_vla.policies.simulation_rollout import (
    SimulationRolloutReport,
    deploy_episodes,
    deployment_output_dir,
    deployment_task_configs,
)


def test_deployment_directory_is_derived_from_policy_and_task() -> None:
    """单任务输出目录应自动使用模型与稳定任务标识命名。"""
    config = AppConfig.load("configs/deploy/smolvla_pipe_top.yaml")
    assert deployment_output_dir(config) == Path("outputs/deploy/smolvla_pipe_top")


def test_deployment_can_load_all_task_modules() -> None:
    """批量部署应复用一个策略配置并展开所有任务 YAML。"""
    config = AppConfig.load("configs/deploy/trajectory_vla.yaml")
    config.deployment.run_all_tasks = True
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
    assert len({deployment_output_dir(item) for item in tasks}) == len(tasks)
    assert (
        next(item for item in tasks if item.task.task_id == "pipe_top").deployment.max_steps == 3300
    )


def test_deployment_can_keep_an_explicit_legacy_directory() -> None:
    """关闭自动命名后应继续支持调用方给出的完整目录。"""
    config = AppConfig.load("configs/deploy/smolvla.yaml")
    config.deployment.auto_log_dir = False
    config.deployment.log_dir = "outputs/deploy/custom_run"
    assert deployment_output_dir(config) == Path("outputs/deploy/custom_run")


def test_all_tasks_write_subdirectories_and_combined_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """批量部署应为每个任务建子目录，并在公共根目录写跨任务摘要。"""
    config = AppConfig.load("configs/deploy/smolvla.yaml")
    config.deployment.run_all_tasks = True
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
    assert all(
        (tmp_path / f"smolvla_{report.task}" / "summary.json").exists() for report in reports
    )
    assert (tmp_path / "smolvla_all_tasks_summary.json").exists()
