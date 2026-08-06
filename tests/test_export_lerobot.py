import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from welding_path_vla.core.config import LeRobotExportConfig
from welding_path_vla.dataset.export_lerobot import (
    export_lerobot,
    export_lerobot_many,
    upload_lerobot_dataset,
    valid_episode_paths,
)


def write_video(path: Path, frame_count: int, value: int) -> None:
    """写入用于转换测试的小尺寸视频。"""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), value + index, dtype=np.uint8))
    writer.release()


def write_raw_episode(root: Path, index: int, valid: bool = True) -> Path:
    """创建满足导出器字段要求的最小原始 episode。"""
    path = root / "episodes" / f"episode_{index:06d}"
    path.mkdir(parents=True)
    action_count = 3
    positions = np.column_stack(
        (np.linspace(0, 0.03, action_count + 1), np.zeros((action_count + 1, 2)))
    )
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (action_count + 1, 1))
    np.savez_compressed(
        path / "trajectory.npz",
        command_delta_pose_seam=np.zeros((action_count, 6)),
        joint_position=np.zeros((action_count + 1, 6)),
        tcp_position=positions,
        tcp_quaternion_wxyz=quaternions,
        safe_command_position=positions[1:],
        safe_command_quaternion_wxyz=quaternions[1:],
    )
    metadata = {
        "instruction": "Weld along the test seam.",
        "quality": {"status": "valid_success" if valid else "invalid_simulation"},
        "resolved_config": {
            "camera": {"height": 24, "width": 32},
            "timing": {"policy_hz": 20},
            "policy": {"action_source": "safe_command"},
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_video(path / "global.mp4", action_count + 1, 20)
    write_video(path / "wrist.mp4", action_count + 1, 80)
    return path


def test_episode_range_is_inclusive_and_filters_invalid(tmp_path: Path) -> None:
    write_raw_episode(tmp_path, 2)
    write_raw_episode(tmp_path, 3, valid=False)
    write_raw_episode(tmp_path, 4)
    selected = valid_episode_paths(tmp_path, start_episode=3, end_episode=4)
    assert [path.name for path in selected] == ["episode_000004"]


def test_video_export_can_resume_and_skip_completed_episodes(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "lerobot"
    write_raw_episode(source, 2)
    write_raw_episode(source, 4)
    (source / "dataset.json").write_text(
        json.dumps({"dataset": "test", "format": "raw_v1", "seed": 7}),
        encoding="utf-8",
    )
    options = LeRobotExportConfig(start_episode=2, end_episode=2)
    first = export_lerobot(source, output, "test/welding", options, action_horizon=2)
    assert first.exported_episodes == 1
    assert not (output / "images").exists()

    options.incremental = True
    options.end_episode = 4
    second = export_lerobot(source, output, "test/welding", options, action_horizon=2)
    third = export_lerobot(source, output, "test/welding", options, action_horizon=2)
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "meta/welding_path_vla_export.json").read_text(encoding="utf-8")
    )
    assert second.exported_episodes == 1
    assert second.skipped_episodes == 1
    assert third.exported_episodes == 0
    assert third.skipped_episodes == 2
    assert info["total_episodes"] == 2
    assert info["features"]["observation.images.global"]["dtype"] == "video"
    assert info["features"]["observation.state"]["names"] == [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_qw",
        "tcp_qx",
        "tcp_qy",
        "tcp_qz",
    ]
    assert info["features"]["action"]["names"][:3] == ["x", "y", "z"]
    assert manifest["action_representation"]["type"] == "relative_action"
    assert manifest["action_representation"]["horizon"] == 2
    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(stats["action"]["mean"][:3], [0.015, 0, 0], atol=1e-3)
    assert len(next(iter(manifest["sources"].values()))) == 2

    with pytest.raises(FileExistsError):
        export_lerobot(source, output, "test/welding", LeRobotExportConfig(), action_horizon=2)


def test_image_storage_is_explicit_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "images"
    write_raw_episode(source, 0)
    options = LeRobotExportConfig(save_images=True)
    export_lerobot(source, output, "test/welding-images", options, action_horizon=2)
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.images.global"]["dtype"] == "image"
    assert not (output / "videos").exists()


def test_default_video_export_uses_bounded_parallel_camera_encoding(tmp_path: Path) -> None:
    """默认关闭流式编码，并保留 LeRobot 的双相机并行编码。"""
    source = tmp_path / "raw"
    output = tmp_path / "videos"
    write_raw_episode(source, 0)
    options = LeRobotExportConfig()
    assert not options.streaming_encoding
    assert options.parallel_video_encoding
    export_lerobot(source, output, "test/welding-parallel", options, action_horizon=2)
    videos = list((output / "videos").rglob("*.mp4"))
    assert len(videos) == 2
    assert not (output / "images").exists()


def test_sequential_export_combines_multiple_raw_datasets(tmp_path: Path) -> None:
    """单 writer 应按顺序把多个源写入同一个 LeRobot 数据集。"""
    sources = [tmp_path / "raw_a", tmp_path / "raw_b"]
    for index, source in enumerate(sources):
        write_raw_episode(source, index)
        (source / "dataset.json").write_text(
            json.dumps({"dataset": source.name, "format": "raw_v1", "seed": index}),
            encoding="utf-8",
        )

    output = tmp_path / "combined"
    options = LeRobotExportConfig()
    report = export_lerobot_many(
        sources,
        output,
        "test/welding-combined",
        options,
        action_horizon=2,
    )

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "meta/welding_path_vla_export.json").read_text(encoding="utf-8")
    )
    assert report.exported_episodes == 2
    assert info["total_episodes"] == 2
    assert len(manifest["sources"]) == 2

    write_raw_episode(sources[0], 10)
    write_raw_episode(sources[1], 11)
    incremental = LeRobotExportConfig(
        incremental=True,
        start_episode=10,
        end_episode=11,
    )
    resumed = export_lerobot_many(
        sources,
        output,
        "test/welding-combined",
        incremental,
        action_horizon=2,
    )
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert resumed.exported_episodes == 2
    assert info["total_episodes"] == 4


def test_export_can_push_finalized_dataset_to_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub 开关应在转换完成后调用 LeRobot 官方上传接口。"""
    source = tmp_path / "raw"
    output = tmp_path / "lerobot"
    write_raw_episode(source, 0)
    calls = []

    def record_push(dataset: LeRobotDataset, **kwargs: object) -> None:
        calls.append((dataset.root, kwargs))

    monkeypatch.setattr(LeRobotDataset, "push_to_hub", record_push)
    options = LeRobotExportConfig(
        push_to_hub=True,
        hub_private=True,
    )
    report = export_lerobot_many(
        [source],
        output,
        "test/welding-upload",
        options,
        action_horizon=2,
    )

    assert report.hub_url == "https://huggingface.co/datasets/test/welding-upload"
    assert calls == [
        (
            output,
            {
                "tags": ["lerobot", "robotics", "welding"],
                "private": True,
            },
        )
    ]


def test_existing_dataset_upload_retries_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅上传入口应重试连接错误，而不重新转换数据。"""
    source = tmp_path / "raw"
    output = tmp_path / "lerobot"
    write_raw_episode(source, 0)
    export_lerobot(source, output, "test/welding-retry", action_horizon=2)
    calls = 0

    def flaky_push(dataset: LeRobotDataset, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectTimeout("temporary timeout")

    monkeypatch.setattr(LeRobotDataset, "push_to_hub", flaky_push)
    monkeypatch.setattr("welding_path_vla.dataset.export_lerobot.sleep", lambda _: None)
    hub_url = upload_lerobot_dataset(
        output,
        "test/welding-retry",
        attempts=3,
        retry_wait_s=0,
    )

    assert calls == 3
    assert hub_url == "https://huggingface.co/datasets/test/welding-retry"
