from pathlib import Path

import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import frame_delta, yaw_degrees_to_matrix
from welding_path_vla.dataset.actions import build_action_chunk, build_relative_action_chunk
from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.simulation.collector import collect_episode


def test_episode_has_n_plus_one_states(tmp_path: Path) -> None:
    config = AppConfig.load("configs/default.yaml")
    config.collection.dataset_root = str(tmp_path)
    config.camera.width = 64
    config.camera.height = 48
    config.task.speed_mps = 1.0
    config.randomization.xy_m = 0
    config.randomization.z_m = 0
    config.randomization.yaw_deg = 0
    config.randomization.recovery_probability = 0
    episode_path = collect_episode(config, 0, 7)
    episode = EpisodeReader(episode_path)
    assert episode.state_count == episode.action_count + 1
    assert (episode_path / "global.mp4").exists()
    assert episode.metadata["quaternion_order"] == "wxyz"
    assert "collision_pairs" in episode.trajectory
    assert "command_delta_pose_world" in episode.trajectory
    assert "executed_delta_pose_world" in episode.trajectory
    assert "safe_command_position" in episode.trajectory
    assert episode.trajectory["episode_done"][-1]
    assert episode.metadata["initial_joint_offset_deg"] != [0.0] * 6
    expected_base = frame_delta(
        episode.trajectory["command_delta_pose_world"][0],
        yaw_degrees_to_matrix(config.scene.robot_base_yaw_deg),
    )
    np.testing.assert_allclose(episode.trajectory["command_delta_pose_base"][0], expected_base)
    assert episode.metadata["coordinate_frames"]["command_delta_pose_base"] == "robot_base"
    assert episode.metadata["episode_start"] == "collision_checked_staging_pose"
    chunk = build_action_chunk(episode, 0, horizon=4, source="executed_tcp")
    assert chunk.shape == (4, 6)
    relative = build_relative_action_chunk(episode, 0, horizon=4)
    assert relative.values.shape == (4, 9)
    assert relative.valid_mask.tolist() == [True] * 4
    compatible = build_relative_action_chunk(episode, 0, horizon=4, include_current=True)
    np.testing.assert_allclose(compatible.values[0], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-7)
    tail = build_relative_action_chunk(episode, episode.action_count - 2, horizon=4)
    assert tail.valid_mask.tolist() == [True, True, False, False]
