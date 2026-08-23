import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from welding_path_vla.core.config import AppConfig, LeRobotExportConfig
from welding_path_vla.dataset.export_lerobot import export_lerobot
from welding_path_vla.dataset.rerender import (
    LOW_DIMENSIONAL_COLUMNS,
    parquet_table,
    render_episode_videos,
    rerender_lerobot_dataset,
    video_frame_count,
)
from welding_path_vla.simulation.models import WorkpieceObject


def file_hash(path: Path) -> str:
    """计算文件内容摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_video(path: Path, frame_count: int, size: tuple[int, int]) -> None:
    """写入便于识别是否被替换的纯色测试视频。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), 30 + index, dtype=np.uint8))
    writer.release()


def write_reconstructable_dataset(root: Path) -> Path:
    """创建包含完整仿真重建信息的最小 raw 数据集。"""
    config = AppConfig.load("configs/trihedral_horizontal.yaml")
    config.camera.width = 64
    config.camera.height = 48
    episode = root / "episodes/episode_000000"
    episode.mkdir(parents=True)
    workpiece_position = np.asarray(config.scene.workpiece_position_m)
    workpiece_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    seam = WorkpieceObject(config).seam(workpiece_position, np.eye(3))
    tcp_positions = np.vstack(
        [
            [0.0, 0.0, 1.0],
            seam.sample(0.05).position,
            seam.sample(0.95).position,
        ]
    )
    joint_positions = np.tile(np.radians(config.robot.initial_joint_deg), (3, 1))
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    np.savez_compressed(
        episode / "trajectory.npz",
        timestamp=np.arange(3) / config.timing.policy_hz,
        joint_position=joint_positions,
        joint_velocity=np.zeros_like(joint_positions),
        tcp_position=tcp_positions,
        tcp_quaternion_wxyz=quaternions,
        command_delta_pose_seam=np.zeros((2, 6)),
        safe_command_position=tcp_positions[1:],
        safe_command_quaternion_wxyz=quaternions[1:],
    )
    metadata = {
        "seed": 7,
        "instruction": config.task.instruction,
        "quality": {"status": "valid_success"},
        "workpiece_position": workpiece_position.tolist(),
        "workpiece_quaternion_wxyz": workpiece_quaternion.tolist(),
        "resolved_config": config.as_dict(),
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = {"dataset": root.name, "format": "weldpath_raw_v1", "seed": 7}
    (root / "dataset.json").write_text(json.dumps(summary), encoding="utf-8")
    write_video(episode / "global.mp4", 3, (64, 48))
    write_video(episode / "wrist.mp4", 3, (64, 48))
    return episode


def read_first_last(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取视频首尾帧。"""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames[0], frames[-1]


def test_raw_rerender_only_replaces_videos(tmp_path: Path) -> None:
    """raw 重渲染应保留轨迹和元数据，并体现焊缝颜色变化。"""
    episode = write_reconstructable_dataset(tmp_path / "weldpath_test_raw_v2")
    trajectory_hash = file_hash(episode / "trajectory.npz")
    metadata_hash = file_hash(episode / "metadata.json")

    frames = render_episode_videos(episode, show_progress=False)

    assert frames == 3
    assert file_hash(episode / "trajectory.npz") == trajectory_hash
    assert file_hash(episode / "metadata.json") == metadata_hash
    assert video_frame_count(episode / "global.mp4") == 3
    assert video_frame_count(episode / "wrist.mp4") == 3
    first, last = read_first_last(episode / "global.mp4")
    assert np.mean(np.abs(first.astype(float) - last.astype(float))) > 0.01


def test_raw_rerender_accepts_legacy_include_current(tmp_path: Path) -> None:
    """raw 重渲染应兼容旧数据中已废弃的策略字段。"""
    episode = write_reconstructable_dataset(tmp_path / "weldpath_legacy_raw_v2")
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resolved_config"]["policy"]["include_current"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert render_episode_videos(episode, show_progress=False) == 3


def test_lerobot_rerender_preserves_low_dimensional_data(tmp_path: Path) -> None:
    """LeRobot 原地重建只能改变视觉列及其统计量。"""
    raw = tmp_path / "weldpath_test_raw_v2"
    episode = write_reconstructable_dataset(raw)
    output = tmp_path / "weldpath_lerobot"
    export_lerobot(
        raw,
        output,
        "test/rerender",
        LeRobotExportConfig(),
        action_horizon=2,
        show_progress=False,
    )
    before = parquet_table(output, "data", LOW_DIMENSIONAL_COLUMNS)
    trajectory_hash = file_hash(episode / "trajectory.npz")

    report = rerender_lerobot_dataset(
        output,
        raw_dataset_glob=str(tmp_path / "*_raw_v2"),
        show_progress=False,
    )

    after = parquet_table(output, "data", LOW_DIMENSIONAL_COLUMNS)
    assert report.dataset_type == "lerobot"
    assert report.episodes == 1
    assert report.frames == 3
    assert before.equals(after)
    assert file_hash(episode / "trajectory.npz") == trajectory_hash
    assert not output.with_name(f"{output.name}_before_rerender").exists()
