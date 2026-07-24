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
