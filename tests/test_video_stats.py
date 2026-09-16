import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import welding_path_vla.dataset.backfill_raw_video_stats as backfill_module
import welding_path_vla.dataset.repair_video_stats as repair_module
import welding_path_vla.dataset.video_stats as video_stats_module
from welding_path_vla.dataset.raw_schema import RAW_DATASET_FORMAT_V1, RAW_DATASET_FORMAT_V2
from welding_path_vla.dataset.video_stats import (
    ActionAlignedVideoStats,
    compute_video_stats,
    read_raw_video_stats,
    write_raw_video_stats,
)


def backfill_episode(root: Path, index: int) -> Path:
    """创建可供回填流程实际解码的 v1 原始 episode。"""
    episode = root / f"episode_{index:06d}"
    episode.mkdir()
    (episode / "metadata.json").write_text(
        json.dumps({"dataset_format": RAW_DATASET_FORMAT_V1, "instruction": "keep label"}),
        encoding="utf-8",
    )
    np.savez(episode / "trajectory.npz", command_delta_pose_seam=np.zeros((2, 6)))
    for camera, value in (("global", 10), ("wrist", 20)):
        video = episode / f"{camera}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (4, 2))
        for frame in (value, value + 1, 250):
            writer.write(np.full((2, 4, 3), frame, dtype=np.uint8))
        writer.release()
    return episode


def source_hashes(episode: Path) -> dict[str, bytes]:
    """读取迁移不得改写的 v1 源文件字节。"""
    return {
        path.name: path.read_bytes()
        for path in episode.glob("*")
        if path.name != "video_stats.json"
    }


def canonical_statistics(value: float = 1.0) -> dict[str, np.ndarray]:
    """创建符合 v2 契约的单相机统计。"""
    return {
        "count": np.array([16], dtype=np.int64),
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


def test_compute_video_stats_uses_action_frames_only(tmp_path: Path):
    """解码回退应忽略 terminal frame 并保留 raw 0--255 数值。"""
    path = tmp_path / "global.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (4, 2))
    for value in (10, 20, 250):
        writer.write(np.full((2, 4, 3), value, dtype=np.uint8))
    writer.release()

    stats = compute_video_stats(path, frame_count=2)

    assert stats["count"].tolist() == [16]
    assert set(stats) == {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert np.all(stats["mean"] < 50)
    assert np.all(stats["max"] < 50)
    assert np.all(stats["min"] >= 0)


def test_compute_video_stats_produces_canonical_quantiles(tmp_path: Path):
    """直方图边界误差不能让解码统计违反严格 sidecar 契约。"""
    path = tmp_path / "global.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (4, 2))
    for value in (10, 11, 250):
        writer.write(np.full((2, 4, 3), value, dtype=np.uint8))
    writer.release()

    write_raw_video_stats(tmp_path, {"global": compute_video_stats(path, frame_count=2)}, 2)

    read_raw_video_stats(tmp_path, ("global",), action_count=2)


def test_action_aligned_stats_exclude_terminal_frame(tmp_path: Path):
    accumulator = ActionAlignedVideoStats(("global", "wrist"))
    frames = [np.full((24, 32, 3), value, np.uint8) for value in (10, 20, 250)]
    for frame in frames:
        accumulator.append({"global": frame, "wrist": frame})
    stats = accumulator.finish(action_count=2)
    assert stats["global"]["mean"].tolist() == [15.0, 15.0, 15.0]
    assert stats["global"]["max"].tolist() == [20, 20, 20]
    write_raw_video_stats(tmp_path, stats, action_count=2)
    read_raw_video_stats(tmp_path, ("global", "wrist"), action_count=2)


def test_action_aligned_stats_reject_invalid_rgb_frame():
    accumulator = ActionAlignedVideoStats(("global",))
    with pytest.raises(ValueError, match="RGB"):
        accumulator.append({"global": np.zeros((2, 2), dtype=np.uint8)})


def test_action_aligned_stats_reject_invalid_rgb_dtype():
    accumulator = ActionAlignedVideoStats(("global",))
    with pytest.raises(ValueError, match="RGB"):
        accumulator.append({"global": np.zeros((2, 2, 3), dtype=np.float32)})


def test_action_aligned_stats_reject_mismatched_action_count():
    accumulator = ActionAlignedVideoStats(("global",))
    accumulator.append({"global": np.zeros((2, 2, 3), dtype=np.uint8)})
    with pytest.raises(ValueError, match="action"):
        accumulator.finish(action_count=2)


def test_action_aligned_stats_reject_mismatched_action_and_state_counts():
    accumulator = ActionAlignedVideoStats(("global",))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    accumulator.append({"global": frame})
    accumulator.append({"global": frame})
    accumulator.append({"global": frame})
    with pytest.raises(ValueError, match="action"):
        accumulator.finish(action_count=1)


def test_raw_video_stats_round_trip_and_validate_contract(tmp_path):
    stats = {"global": canonical_statistics()}
    path = write_raw_video_stats(tmp_path, stats, action_count=2)
    assert path.name == "video_stats.json"
    loaded = read_raw_video_stats(tmp_path, ("global",), action_count=2)
    for key in stats["global"]:
        np.testing.assert_array_equal(loaded["global"][key], stats["global"][key])

    payload = json.loads(path.read_text())
    payload["algorithm"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="algorithm"):
        read_raw_video_stats(tmp_path, ("global",), action_count=2)


def test_raw_video_stats_reject_missing_camera_and_non_finite_values(tmp_path):
    stats = {"global": canonical_statistics()}
    write_raw_video_stats(tmp_path, stats, action_count=2)
    with pytest.raises(ValueError, match="camera"):
        read_raw_video_stats(tmp_path, ("global", "wrist"), action_count=2)
    bad = {"global": canonical_statistics()}
    bad["global"]["mean"] = np.array([np.nan, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        write_raw_video_stats(tmp_path, bad, action_count=2)


@pytest.mark.parametrize(
    "field, value",
    [("mean", np.zeros(2)), ("min", np.zeros((3, 1))), ("count", np.zeros(3, dtype=np.int64))],
)
def test_raw_video_stats_reject_wrong_field_shape_or_dtype(tmp_path, field, value):
    stats = {"global": canonical_statistics()}
    stats["global"][field] = value
    with pytest.raises(ValueError, match="shape or dtype"):
        write_raw_video_stats(tmp_path, stats, action_count=2)


def test_raw_video_stats_reject_incomplete_fields_and_atomic_write(tmp_path):
    incomplete = {"global": canonical_statistics()}
    incomplete["global"].pop("q99")
    with pytest.raises(ValueError, match="fields"):
        write_raw_video_stats(tmp_path, incomplete, action_count=2)
    stats = {"global": canonical_statistics()}
    path = write_raw_video_stats(tmp_path, stats, action_count=2)
    assert not (tmp_path / ".video_stats.json.tmp").exists()
    assert path.exists()


def test_raw_video_stats_reader_rejects_incomplete_and_malformed_fields(tmp_path):
    """读取 v2 sidecar 时也必须拒绝缺失字段和非 int64 样本数。"""
    path = write_raw_video_stats(tmp_path, {"global": canonical_statistics()}, action_count=2)
    payload = json.loads(path.read_text())
    payload["stats"]["global"].pop("q99")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields"):
        read_raw_video_stats(tmp_path, ("global",), action_count=2)

    write_raw_video_stats(tmp_path, {"global": canonical_statistics()}, action_count=2)
    payload = json.loads(path.read_text())
    payload["stats"]["global"]["count"] = [1.5]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="int64"):
        read_raw_video_stats(tmp_path, ("global",), action_count=2)


def test_raw_video_stats_reader_rejects_unordered_quantiles(tmp_path):
    """v2 canonical quantile 必须在每个通道内单调递增。"""
    path = write_raw_video_stats(tmp_path, {"global": canonical_statistics()}, action_count=2)
    payload = json.loads(path.read_text())
    payload["stats"]["global"]["q10"] = [-1.0, -1.0, -1.0]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="ordered"):
        read_raw_video_stats(tmp_path, ("global",), action_count=2)


def test_raw_video_stats_replaces_sidecar_atomically(tmp_path, monkeypatch):
    stats = {"global": canonical_statistics()}
    replacements = []
    original_replace = video_stats_module.os.replace

    def record_replace(source, destination):
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(video_stats_module.os, "replace", record_replace)
    write_raw_video_stats(tmp_path, stats, action_count=2)
    assert len(replacements) == 1
    assert replacements[0][1] == tmp_path / "video_stats.json"
    assert replacements[0][0] == tmp_path / ".video_stats.json.tmp"


def test_raw_video_stats_exclusive_write_refuses_existing_sidecar(tmp_path):
    """非覆盖写入必须保留并拒绝已出现的 sidecar。"""
    path = tmp_path / "video_stats.json"
    existing = b'{"invalid": true}'
    path.write_bytes(existing)

    with pytest.raises(FileExistsError):
        write_raw_video_stats(tmp_path, {"global": canonical_statistics()}, 2, overwrite=False)

    assert path.read_bytes() == existing


def repair_episode(root: Path, index: int, format_name: str) -> Path:
    """创建只包含 repair 所需元数据和视频占位符的原始 episode。"""
    episode = root / f"episode_{index:06d}"
    episode.mkdir()
    (episode / "metadata.json").write_text(json.dumps({"dataset_format": format_name}))
    np.savez(episode / "trajectory.npz", command_delta_pose_seam=np.zeros((2, 6)))
    for camera in ("global", "wrist"):
        (episode / f"{camera}.mp4").write_bytes(camera.encode())
    return episode


def test_repair_mixed_raw_formats_reuses_v2_sidecar(tmp_path, monkeypatch):
    """v2 sidecar episode 不解码，v1 episode 仍逐相机解码回退。"""
    v2 = repair_episode(tmp_path, 0, RAW_DATASET_FORMAT_V2)
    v1 = repair_episode(tmp_path, 1, RAW_DATASET_FORMAT_V1)
    write_raw_video_stats(v2, {name: canonical_statistics(20.0) for name in ("global", "wrist")}, 2)
    decoded = []

    def decode(path, frame_count, terminal_frame=False):
        decoded.append((Path(path).name, frame_count, terminal_frame))
        return canonical_statistics(10.0)

    monkeypatch.setattr(repair_module, "compute_video_stats", decode)
    checkpoint = repair_module.repair_video_stats((v2, v1), tmp_path / "repair.json")

    assert decoded == [("global.mp4", 2, True), ("wrist.mp4", 2, True)]
    np.testing.assert_array_equal(
        checkpoint[str(v2)]["statistics"]["global"]["mean"],
        np.full((3, 1, 1), 20.0 / 255),
    )
    assert checkpoint[str(v2)]["statistics"]["global"]["count"].tolist() == [16]


def test_repair_reuses_backfilled_v1_sidecar(tmp_path, monkeypatch):
    """v1 已回填有效 sidecar 时同样避免不必要的视频解码。"""
    episode = repair_episode(tmp_path, 0, RAW_DATASET_FORMAT_V1)
    write_raw_video_stats(
        episode,
        {name: canonical_statistics(20.0) for name in ("global", "wrist")},
        2,
    )
    monkeypatch.setattr(
        repair_module,
        "compute_video_stats",
        lambda *args, **kwargs: pytest.fail("backfilled v1 episode must not decode"),
    )

    checkpoint = repair_module.repair_video_stats((episode,), tmp_path / "repair.json")

    np.testing.assert_array_equal(
        checkpoint[str(episode)]["statistics"]["wrist"]["max"],
        np.full((3, 1, 1), 22.0 / 255),
    )


def test_repair_checkpoint_sidecar_replacement_invalidates_only_that_episode(tmp_path, monkeypatch):
    """替换一个 sidecar 只重算其所属 episode，其他检查点保持复用。"""
    episodes = tuple(repair_episode(tmp_path, index, RAW_DATASET_FORMAT_V2) for index in range(2))
    for episode in episodes:
        write_raw_video_stats(
            episode,
            {name: canonical_statistics(10.0) for name in ("global", "wrist")},
            2,
        )
    computed = []
    original = repair_module.compute_episode

    def record_compute(episode):
        computed.append(episode.path)
        return original(episode)

    monkeypatch.setattr(repair_module, "compute_episode", record_compute)
    path = tmp_path / "repair.json"
    repair_module.repair_video_stats(episodes, path)
    assert computed == list(episodes)

    computed.clear()
    repair_module.repair_video_stats(episodes, path)
    assert computed == []

    write_raw_video_stats(
        episodes[0],
        {name: canonical_statistics(30.0) for name in ("global", "wrist")},
        2,
    )
    repair_module.repair_video_stats(episodes, path)
    assert computed == [episodes[0]]


def test_backfill_dry_run_preserves_v1_sources_and_requires_apply(tmp_path):
    """默认预演只报告待回填 episode，不创建 sidecar 或修改 v1 文件。"""
    episode = backfill_episode(tmp_path, 0)
    before = source_hashes(episode)

    report = backfill_module.backfill_raw_video_stats((tmp_path,), workers=1, apply=False)

    assert report.as_dict() == {"backfilled": 0, "dry_run": 1, "refused": 0, "skipped": 0}
    assert not (episode / "video_stats.json").exists()
    assert source_hashes(episode) == before


def test_backfill_apply_writes_atomic_sidecar_and_preserves_v1_label(tmp_path):
    """显式 apply 仅以原子 sidecar 写入补充统计，不修改 v1 标签或源字节。"""
    episode = backfill_episode(tmp_path, 0)
    before = source_hashes(episode)
    report = backfill_module.backfill_raw_video_stats((tmp_path,), workers=1, apply=True)

    assert report.as_dict() == {
        "backfilled": 1,
        "dry_run": 0,
        "refused": 0,
        "skipped": 0,
    }
    assert not list(episode.glob(".video_stats.json.*.tmp"))
    assert (
        json.loads((episode / "metadata.json").read_text())["dataset_format"]
        == RAW_DATASET_FORMAT_V1
    )
    assert source_hashes(episode) == before
    read_raw_video_stats(episode, ("global", "wrist"), action_count=2)


def test_backfill_resume_skips_valid_sidecars(tmp_path, monkeypatch):
    """已存在且有效的 sidecar 是恢复点，重跑时不再解码或写入。"""
    episode = backfill_episode(tmp_path, 0)
    backfill_module.backfill_raw_video_stats((tmp_path,), workers=1, apply=True)
    monkeypatch.setattr(
        backfill_module,
        "compute_video_stats",
        lambda *args, **kwargs: pytest.fail("valid sidecar must be skipped"),
    )

    report = backfill_module.backfill_raw_video_stats((tmp_path,), workers=1, apply=True)

    assert report.as_dict() == {"backfilled": 0, "dry_run": 0, "refused": 0, "skipped": 1}
    assert (episode / "video_stats.json").exists()


def test_backfill_refuses_invalid_sidecar_without_overwrite(tmp_path, monkeypatch):
    """残缺或错误的已有 sidecar 必须拒绝，避免回填掩盖数据问题。"""
    episode = backfill_episode(tmp_path, 0)
    sidecar = episode / "video_stats.json"
    sidecar.write_text("{}", encoding="utf-8")
    before = sidecar.read_bytes()
    monkeypatch.setattr(
        backfill_module,
        "compute_video_stats",
        lambda *args, **kwargs: pytest.fail("invalid sidecar must not be replaced"),
    )

    report = backfill_module.backfill_raw_video_stats((tmp_path,), workers=1, apply=True)

    assert report.as_dict() == {"backfilled": 0, "dry_run": 0, "refused": 1, "skipped": 0}
    assert sidecar.read_bytes() == before


def test_backfill_refuses_sidecar_created_during_decode(tmp_path, monkeypatch):
    """解码期间新出现的无效 sidecar 必须保留，不能被回填覆盖。"""
    episode = backfill_episode(tmp_path, 0)
    sidecar = episode / "video_stats.json"
    existing = b'{"invalid": true}'

    def decode(*args, **kwargs):
        sidecar.write_bytes(existing)
        return canonical_statistics()

    monkeypatch.setattr(backfill_module, "compute_video_stats", decode)

    report = backfill_module.backfill_raw_video_stats((tmp_path,), workers=1, apply=True)

    assert report.as_dict() == {"backfilled": 0, "dry_run": 0, "refused": 1, "skipped": 0}
    assert sidecar.read_bytes() == existing


def test_backfill_uses_multiple_workers_for_independent_episodes(tmp_path):
    """多个独立 v1 episode 可由多个进程回填，且不会遗漏任何 sidecar。"""
    episodes = [backfill_episode(tmp_path, index) for index in range(2)]

    report = backfill_module.backfill_raw_video_stats((tmp_path,), workers=2, apply=True)

    assert report.as_dict() == {"backfilled": 2, "dry_run": 0, "refused": 0, "skipped": 0}
    for episode in episodes:
        read_raw_video_stats(episode, ("global", "wrist"), action_count=2)


def test_backfill_cli_help_and_dry_run(tmp_path):
    """独立 CLI 可显示 apply 说明，且默认预演不写入 sidecar。"""
    episode = backfill_episode(tmp_path, 0)
    script = Path(__file__).parents[1] / "scripts" / "backfill_raw_video_stats.py"

    help_result = subprocess.run(
        [sys.executable, script, "--help"], text=True, capture_output=True, check=True
    )
    dry_run = subprocess.run(
        [sys.executable, script, "--raw-glob", str(tmp_path), "--workers", "1"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--apply" in help_result.stdout
    assert json.loads(dry_run.stdout) == {"backfilled": 0, "dry_run": 1, "refused": 0, "skipped": 0}
    assert not (episode / "video_stats.json").exists()
