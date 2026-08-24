from pathlib import Path

import av
import numpy as np

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import frame_delta, yaw_degrees_to_matrix
from welding_path_vla.dataset.actions import (
    build_absolute_actions,
    build_delta_action,
    build_relative_actions,
)
from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.dataset.recorder import EpisodeRecorder
from welding_path_vla.evaluation.trajectory_metrics import report_from_arrays
from welding_path_vla.simulation.collector import collect_episode


def test_default_video_recorder_does_not_probe_unavailable_hardware_codec(
    tmp_path: Path,
    capfd,
) -> None:
    """LeRobot 录制器不应探测不可用的 V4L2 硬件编码器。"""
    config = AppConfig.load("configs/default.yaml")
    recorder = EpisodeRecorder(tmp_path, 0, config)
    recorder.abort()
    errors = capfd.readouterr().err
    assert "h264_v4l2m2m" not in errors
    assert "Failed to initialize VideoWriter" not in errors


def test_raw_video_is_browser_compatible_h264(tmp_path: Path, capfd) -> None:
    """原始视频应采用 VS Code 内置 Chromium 可播放的 H.264 格式。"""
    config = AppConfig.load("configs/default.yaml")
    recorder = EpisodeRecorder(tmp_path, 0, config)
    image = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
    recorder.video.append({"global": image, "wrist": image})
    recorder.video.finish()
    errors = capfd.readouterr().err
    assert "libx264" not in errors
    assert "Auto-inserting" not in errors
    assert "Starting second pass" not in errors
    with av.open(recorder.temporary_path / "global.mp4") as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.pix_fmt == "yuv420p"
        assert float(stream.average_rate) >= 30
    recorder.abort()


def test_episode_has_n_plus_one_states(tmp_path: Path) -> None:
    config = AppConfig.load("configs/default.yaml")
    config.collection.dataset_root = str(tmp_path)
    config.camera.width = 64
    config.camera.height = 48
    config.task.speed_mps = 0.04
    config.randomization.xy_m = 0
    config.randomization.z_m = 0
    config.randomization.yaw_deg = 0
    config.randomization.recovery_probability = 0
    episode_path = collect_episode(config, 0, 7)
    episode = EpisodeReader(episode_path)
    assert episode.state_count == episode.action_count + 1
    assert (episode_path / "global.mp4").exists()
    with av.open(episode_path / "global.mp4") as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == config.timing.policy_hz
        assert stream.frames == episode.state_count
    assert episode.metadata["quaternion_order"] == "wxyz"
    assert "collision_pairs" in episode.trajectory
    assert "command_delta_pose_world" in episode.trajectory
    assert "executed_delta_pose_world" in episode.trajectory
    assert "safe_command_position" in episode.trajectory
    assert episode.trajectory["episode_done"][-1]
    assert episode.metadata["initial_joint_offset_deg"] != [0.0] * 6
    assert len(episode.metadata["initial_joint_position_deg"]) == 6
    assert len(episode.metadata["initial_tcp_position_m"]) == 3
    assert episode.metadata["planning_max_ik_residual"] <= 0.005
    assert episode.metadata["quality"]["failure_reasons"] == []
    expected_base = frame_delta(
        episode.trajectory["command_delta_pose_world"][0],
        yaw_degrees_to_matrix(config.scene.robot_base_yaw_deg),
    )
    np.testing.assert_allclose(episode.trajectory["command_delta_pose_base"][0], expected_base)
    assert episode.metadata["coordinate_frames"]["command_delta_pose_base"] == "robot_base"
    assert episode.metadata["episode_start"] == "collision_checked_staging_pose"
    assert episode.metadata["instruction"] == config.task.instruction
    task_parameters = episode.metadata["task_parameters"]
    assert task_parameters["group_index"] == 0
    assert task_parameters["group_size"] == 10
    assert task_parameters["speed_mps"] == round(task_parameters["speed_mps"], 3)
    absolute = build_absolute_actions(episode)
    assert absolute.shape == (episode.action_count, 9)
    relative = build_relative_actions(episode, 0, horizon=4)
    assert relative.values.shape == (4, 9)
    assert relative.valid_mask.tolist() == [True] * 4
    tail = build_relative_actions(episode, episode.action_count - 2, horizon=4)
    assert tail.valid_mask.tolist() == [True, True, False, False]
    delta = build_delta_action(episode)
    assert delta.shape == (episode.action_count, 9)
    for index in (0, episode.action_count // 2, episode.action_count - 1):
        expected = build_relative_actions(episode, index, horizon=1).values[0]
        np.testing.assert_allclose(delta[index], expected, atol=1e-6)


def test_quality_report_explains_every_failed_condition() -> None:
    """质量报告应给出具体失败条件、碰撞帧数和几何对。"""
    trajectory = {
        "phase": np.array(["track", "track"]),
        "seam_progress": np.array([0.0, 0.5]),
        "cross_track_error": np.array([0.0, 0.02]),
        "orientation_error_deg": np.array([0.0, 8.0]),
        "ik_residual": np.array([0.0, 0.01]),
        "collision": np.array([False, True]),
        "collision_pairs": np.array(["", "torch_nozzle:plate_vertical"]),
    }

    report = report_from_arrays(trajectory, AppConfig().quality, recovery=False)

    assert set(report.failure_reasons) == {
        "incomplete_seam",
        "cross_track_mean",
        "cross_track_p95",
        "cross_track_max",
        "orientation_p95",
        "orientation_max",
        "collision",
        "ik_residual",
    }
    assert report.collision_frames == 1
    assert report.collision_pairs == ("torch_nozzle:plate_vertical",)
