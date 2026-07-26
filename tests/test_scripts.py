import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    "collect_simulation_data.py",
    "view_simulation.py",
    "replay_episode.py",
    "validate_dataset.py",
    "export_lerobot.py",
    "evaluate.py",
    "show_robot_config.py",
    "show_policy_config.py",
    "train_policy.py",
    "check_policy_data.py",
    "evaluate_policy.py",
    "deploy_simulation_policy.py",
)


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help_is_directly_runnable(name: str) -> None:
    """每个公开脚本都应能独立解析帮助参数。"""
    path = Path("scripts") / name
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_draccus_loads_config_and_applies_nested_overrides() -> None:
    """配置文件与命令行覆盖应由 draccus 一次完成。"""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/show_policy_config.py",
            "--config_path=configs/act.yaml",
            "--training.steps=7",
            "--policy.action_steps=3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["policy"]["family"] == "act"
    assert config["policy"]["action_steps"] == 3
    assert config["training"]["steps"] == 7
