import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import av
import cv2
import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from welding_path_vla.core.config import LeRobotExportConfig
from welding_path_vla.dataset.export_lerobot import (
    export_lerobot,
    export_lerobot_many,
    upload_lerobot_dataset,
    valid_episode_paths,
)
from welding_path_vla.dataset.migrate_task_parameters import migrate_task_parameters
from welding_path_vla.dataset.video_stats import write_raw_video_stats
from welding_path_vla.policies.data import phase_frame_indices


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


def write_h264_video(path: Path, frame_count: int, value: int) -> None:
    """写入无 B 帧的 H.264 测试视频，满足 packet passthrough 契约。"""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20, options={"bf": "0"})
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), value + index, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def h264_packet_hashes(path: Path) -> list[str]:
    """返回视频 packet 的哈希，用于断言未重新编码。"""
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        return [
            sha256(bytes(packet)).hexdigest()
            for packet in container.demux(stream)
            if packet.dts is not None
        ]


def write_passthrough_episode(root: Path, index: int, format_name: str) -> Path:
    """创建包含 H.264 双相机视频的 raw episode。"""
    path = write_raw_episode(root, index)
    for name, value in (("global", 20), ("wrist", 80)):
        write_h264_video(path / f"{name}.mp4", frame_count=4, value=value)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_format"] = format_name
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def raw_statistics(global_value: float = 20, wrist_value: float = 80) -> dict:
    """创建不依赖被测代码计算的 raw 0--255 双相机统计。"""
    values = {"global": global_value, "wrist": wrist_value}
    return {
        name: {
            "count": np.array([2304], dtype=np.int64),
            "min": np.full(3, value - 2, dtype=np.float64),
            "max": np.full(3, value + 2, dtype=np.float64),
            "mean": np.full(3, value, dtype=np.float64),
            "std": np.ones(3, dtype=np.float64),
            "q01": np.full(3, value - 1.8, dtype=np.float64),
            "q10": np.full(3, value - 1.5, dtype=np.float64),
            "q50": np.full(3, value, dtype=np.float64),
            "q90": np.full(3, value + 1.5, dtype=np.float64),
            "q99": np.full(3, value + 1.8, dtype=np.float64),
        }
        for name, value in values.items()
    }


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
        phase=np.array(["approach", "track", "retreat"]),
        joint_position=np.zeros((action_count + 1, 6)),
        tcp_position=positions,
        tcp_quaternion_wxyz=quaternions,
        safe_command_position=positions[1:],
        safe_command_quaternion_wxyz=quaternions[1:],
    )
    metadata = {
        "instruction": "Weld along the test seam.",
        "direction": "reverse" if index % 2 else "forward",
        "task_parameters": {
            "direction": "reverse" if index % 2 else "forward",
            "speed_mps": 0.02 + index * 0.001,
            "work_angle_deg": 45 + index,
            "travel_angle_deg": 10 + index,
            "tool_roll_deg": -20 + index,
        },
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


def test_export_writes_phase_index_aligned_to_lerobot_frames(tmp_path: Path) -> None:
    """导出后 phase sidecar 应按 LeRobot 全局 frame 顺序保存。"""
    source, output = tmp_path / "raw", tmp_path / "lerobot"
    write_raw_episode(source, 0)

    export_lerobot(source, output, "test/welding-phase", action_horizon=2, show_progress=False)

    with np.load(output / "meta/phase_index.npz", allow_pickle=False) as sidecar:
        assert sidecar["phase"].tolist() == [0, 1, 2]
    mapping = phase_frame_indices(output / "meta/phase_index.npz")
    assert {name: frames.tolist() for name, frames in mapping.items()} == {
        "approach": [0],
        "track": [1],
        "retreat": [2],
    }


def test_stored_video_stats_skip_decode_and_preserve_h264_packets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v2 sidecar 应直接进入 writer，且 H.264 packet 不得重新编码。"""
    source, output = tmp_path / "raw", tmp_path / "lerobot"
    episode = write_passthrough_episode(source, 0, "weldpath_raw_v2")
    expected = raw_statistics()
    write_raw_video_stats(
        episode,
        expected,
        action_count=3,
        image_shapes={"global": (24, 32, 3), "wrist": (24, 32, 3)},
    )
    source_hashes = {name: h264_packet_hashes(episode / f"{name}.mp4")[:3] for name in expected}
    monkeypatch.setattr(
        "welding_path_vla.dataset.export_lerobot.compute_video_stats",
        lambda *_: (_ for _ in ()).throw(AssertionError("v2 must not decode")),
    )

    export_lerobot(
        source,
        output,
        "test/welding-stored-stats",
        LeRobotExportConfig(video_passthrough=True),
        action_horizon=2,
    )

    stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    episode_stats = pq.read_table(next((output / "meta" / "episodes").rglob("*.parquet")))
    for name, key in (
        ("global", "observation.images.global"),
        ("wrist", "observation.images.wrist"),
    ):
        for field, value in expected[name].items():
            normalized = value if field == "count" else (value / 255).reshape(3, 1, 1)
            np.testing.assert_allclose(stats[key][field], normalized)
            np.testing.assert_allclose(
                episode_stats[f"stats/{key}/{field}"].to_pylist()[0], normalized
            )
        target = next(path for path in (output / "videos").rglob("*.mp4") if key in str(path))
        assert h264_packet_hashes(target) == source_hashes[name]


def test_backfilled_v1_video_stats_skip_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """带有效回填 sidecar 的 v1 episode 必须复用统计而非解码。"""
    source, output = tmp_path / "raw", tmp_path / "lerobot"
    episode = write_passthrough_episode(source, 0, "weldpath_raw_v1")
    write_raw_video_stats(
        episode,
        raw_statistics(),
        action_count=3,
        image_shapes={"global": (24, 32, 3), "wrist": (24, 32, 3)},
    )
    monkeypatch.setattr(
        "welding_path_vla.dataset.export_lerobot.compute_video_stats",
        lambda *_: (_ for _ in ()).throw(AssertionError("backfilled v1 must not decode")),
    )

    export_lerobot(
        source,
        output,
        "test/welding",
        LeRobotExportConfig(video_passthrough=True),
        action_horizon=2,
    )


def test_legacy_video_stats_decode_each_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有 sidecar 的 v1 episode 应仅通过解码回退为双相机生成统计。"""
    source, output = tmp_path / "raw", tmp_path / "lerobot"
    write_passthrough_episode(source, 0, "weldpath_raw_v1")
    calls = []

    def record_decode(path: Path, frame_count: int) -> dict:
        calls.append((path.parent.name.rsplit("-", 1)[1], frame_count))
        return {
            **raw_statistics()[calls[-1][0]],
        }

    monkeypatch.setattr(
        "welding_path_vla.dataset.export_lerobot.compute_video_stats", record_decode
    )
    export_lerobot(
        source,
        output,
        "test/welding-legacy-stats",
        LeRobotExportConfig(video_passthrough=True),
        action_horizon=2,
    )
    assert calls == [("global", 3), ("wrist", 3)]


def test_invalid_v2_video_stats_fail_before_metadata(tmp_path: Path) -> None:
    """损坏的 v2 sidecar 必须在 LeRobot 创建任何元数据前失败。"""
    source, output = tmp_path / "raw", tmp_path / "lerobot"
    episode = write_passthrough_episode(source, 0, "weldpath_raw_v2")
    (episode / "video_stats.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="format or algorithm"):
        export_lerobot(
            source,
            output,
            "test/welding-invalid-stats",
            LeRobotExportConfig(video_passthrough=True),
            action_horizon=2,
        )

    assert not (output / "meta").exists()


def test_incomplete_v2_video_stats_fail_before_metadata(tmp_path: Path) -> None:
    """v2 sidecar 缺少 canonical 字段时不得创建 LeRobot 元数据。"""
    source, output = tmp_path / "raw", tmp_path / "lerobot"
    episode = write_passthrough_episode(source, 0, "weldpath_raw_v2")
    statistics = raw_statistics()
    statistics["global"].pop("q99")

    with pytest.raises(ValueError, match="fields"):
        write_raw_video_stats(
            episode,
            statistics,
            action_count=3,
            image_shapes={"global": (24, 32, 3), "wrist": (24, 32, 3)},
        )

    assert not (output / "meta").exists()


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

    stale_image = output / "images/observation.images.global/episode-000001/frame-000000.png"
    stale_image.parent.mkdir(parents=True)
    stale_image.write_bytes(b"interrupted temporary frame")
    options.incremental = True
    options.end_episode = 4
    second = export_lerobot(source, output, "test/welding", options, action_horizon=2)
    stale_image.parent.mkdir(parents=True)
    stale_image.write_bytes(b"interrupted temporary frame")
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
    assert not (output / "images").exists()
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


def test_export_preserves_numeric_task_parameters(tmp_path: Path) -> None:
    """新导出应保留方向和焊接参数，同时维持原有语言任务语义。"""
    source = tmp_path / "raw"
    output = tmp_path / "lerobot"
    write_raw_episode(source, 0)
    export_lerobot(source, output, "test/welding-parameters", action_horizon=2)

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["task.direction"] == {
        "dtype": "int64",
        "shape": [1],
        "names": ["direction"],
    }
    assert info["features"]["task.parameters"]["names"] == [
        "welding_speed_mps",
        "work_angle_deg",
        "travel_angle_deg",
        "tool_roll_deg",
    ]

    dataset = LeRobotDataset("test/welding-parameters", root=output)
    sample = dataset.hf_dataset[0]
    assert int(sample["task.direction"]) == 0
    np.testing.assert_allclose(sample["task.parameters"], [0.02, 45, 10, -20])
    assert dataset.meta.tasks.index.tolist() == ["Weld along the test seam."]


def strip_task_parameters(root: Path) -> None:
    """把测试导出退化成迁移前的旧 LeRobot schema。"""
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path).drop(["task.direction", "task.parameters"])
        pq.write_table(table, path, compression="snappy")
    for path in (root / "meta/episodes").rglob("*.parquet"):
        table = pq.read_table(path)
        columns = [name for name in table.column_names if name.startswith("stats/task.")]
        pq.write_table(table.drop(columns), path, compression="snappy")
    for filename, parent in (("info.json", "features"), ("stats.json", None)):
        path = root / "meta" / filename
        document = json.loads(path.read_text(encoding="utf-8"))
        values = document[parent] if parent else document
        values.pop("task.direction")
        values.pop("task.parameters")
        path.write_text(json.dumps(document), encoding="utf-8")


def test_existing_dataset_can_add_task_parameters_in_place(tmp_path: Path) -> None:
    """旧数据应只重写 Parquet 和元数据，不重新编码视频或替换语言任务。"""
    source = tmp_path / "raw"
    output = tmp_path / "lerobot"
    write_raw_episode(source, 0)
    write_raw_episode(source, 1)
    (source / "dataset.json").write_text(
        json.dumps({"dataset": "test", "format": "raw_v1", "seed": 7}),
        encoding="utf-8",
    )
    export_lerobot(source, output, "test/welding-migration", action_horizon=2)
    videos_before = {path: path.stat().st_mtime_ns for path in (output / "videos").rglob("*.mp4")}
    strip_task_parameters(output)

    report = migrate_task_parameters(output, [source])
    verified = migrate_task_parameters(output, [source], verify_only=True)

    dataset = LeRobotDataset("test/welding-migration", root=output)
    forward = dataset.hf_dataset[0]
    reverse_index = next(
        index
        for index, episode_index in enumerate(dataset.hf_dataset["episode_index"])
        if episode_index == 1
    )
    reverse = dataset.hf_dataset[reverse_index]
    assert report.episodes == 2
    assert report.data_files_changed == 1
    assert report.metadata_files_changed == 1
    assert verified.data_files_changed == 0
    assert int(forward["task.direction"]) == 0
    assert int(reverse["task.direction"]) == 1
    np.testing.assert_allclose(reverse["task.parameters"], [0.021, 46, 11, -19])
    assert dataset.meta.tasks.index.tolist() == ["Weld along the test seam."]
    assert videos_before == {
        path: path.stat().st_mtime_ns for path in (output / "videos").rglob("*.mp4")
    }


def test_default_video_export_uses_bounded_streaming_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认直接流式写视频，不经过临时 PNG。"""
    from lerobot.datasets import dataset_writer

    source = tmp_path / "raw"
    output = tmp_path / "videos"
    write_raw_episode(source, 0)
    written_images = []
    write_image = dataset_writer.write_image

    def record_write_image(*args: object, **kwargs: object) -> None:
        written_images.append(args[1])
        write_image(*args, **kwargs)

    monkeypatch.setattr(dataset_writer, "write_image", record_write_image)
    options = LeRobotExportConfig()
    assert options.streaming_encoding
    assert options.encoder_queue_maxsize == 30
    assert options.encoder_threads == 4
    assert options.video_codec == "libsvtav1"
    assert options.video_preset == "12"
    export_lerobot(source, output, "test/welding-parallel", options, action_horizon=2)
    videos = list((output / "videos").rglob("*.mp4"))
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert len(videos) == 2
    assert info["features"]["observation.images.global"]["info"]["video.codec"] == "av1"
    assert not written_images
    assert not (output / "images").exists()


def test_video_export_releases_native_memory_after_each_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每条视频写完后都应归还 SVT-AV1 在线程堆中保留的空闲内存。"""
    source = tmp_path / "raw"
    output = tmp_path / "videos"
    write_raw_episode(source, 0)
    write_raw_episode(source, 1)
    releases = []
    monkeypatch.setattr(
        "welding_path_vla.dataset.export_lerobot.release_encoder_memory",
        lambda: releases.append(True),
    )

    export_lerobot(source, output, "test/welding-memory", action_horizon=2)

    assert releases == [True, True]


def test_export_cli_uses_base_config_by_default(tmp_path: Path) -> None:
    """未指定配置路径时，导出任务应读取 base.yaml。"""
    source = tmp_path / "raw"
    output = tmp_path / "videos"
    write_raw_episode(source, 0)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_lerobot.py",
            f"--dataset={source}",
            f"--output={output}",
            "--policy.action_horizon=2",
            "--policy.action_steps=2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    video = info["features"]["observation.images.global"]["info"]
    assert video["video.codec"] == "av1"
    assert video["video.preset"] == "12"


def test_export_cli_hides_dependency_progress_and_encoder_banner(tmp_path: Path) -> None:
    """CLI 应只显示项目 episode 进度，不混入 Map 和 SVT 原生日志。"""
    source = tmp_path / "raw"
    output = tmp_path / "videos"
    write_raw_episode(source, 0)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_lerobot.py",
            f"--dataset={source}",
            f"--output={output}",
            "--policy.action_horizon=2",
            "--policy.action_steps=2",
            "--lerobot_export.video_codec=h264",
            "--lerobot_export.video_preset=veryfast",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Exporting raw" in result.stderr
    assert "Map:" not in result.stderr
    assert "Svt[" not in result.stderr
    assert "libx264" not in result.stderr
    assert "Auto-inserting" not in result.stderr
    assert "Starting second pass" not in result.stderr


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
    with np.load(output / "meta/phase_index.npz", allow_pickle=False) as sidecar:
        assert sidecar["phase"].tolist() == [0, 1, 2] * 4


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
