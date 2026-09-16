import json

import numpy as np
import pytest

import welding_path_vla.dataset.video_stats as video_stats_module
from welding_path_vla.dataset.video_stats import (
    ActionAlignedVideoStats,
    read_raw_video_stats,
    write_raw_video_stats,
)


def test_action_aligned_stats_exclude_terminal_frame():
    accumulator = ActionAlignedVideoStats(("global", "wrist"))
    frames = [np.full((24, 32, 3), value, np.uint8) for value in (10, 20, 250)]
    for frame in frames:
        accumulator.append({"global": frame, "wrist": frame})
    stats = accumulator.finish(action_count=2)
    assert stats["global"]["mean"].tolist() == [15.0, 15.0, 15.0]
    assert stats["global"]["max"].tolist() == [20, 20, 20]


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
    stats = {"global": {key: np.array(value, dtype=np.float64) for key, value in {
        "mean": [1.0, 2.0, 3.0], "min": [0, 0, 0], "max": [4, 5, 6]}.items()}}
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
    stats = {"global": {key: np.array(value, dtype=np.float64) for key, value in {
        "mean": [1.0, 2.0, 3.0], "min": [0, 0, 0], "max": [4, 5, 6]}.items()}}
    write_raw_video_stats(tmp_path, stats, action_count=2)
    with pytest.raises(ValueError, match="camera"):
        read_raw_video_stats(tmp_path, ("global", "wrist"), action_count=2)
    bad = {"global": {"mean": np.array([np.nan, 2.0, 3.0]), "min": np.zeros(3), "max": np.ones(3)}}
    with pytest.raises(ValueError, match="finite"):
        write_raw_video_stats(tmp_path, bad, action_count=2)


@pytest.mark.parametrize("field, value", [("mean", np.zeros(2)), ("min", np.zeros((3, 1))), ("max", np.zeros(3, dtype=np.int32))])
def test_raw_video_stats_reject_wrong_field_shape_or_dtype(tmp_path, field, value):
    stats = {"global": {"mean": np.zeros(3), "min": np.zeros(3), "max": np.ones(3)}}
    stats["global"][field] = value
    with pytest.raises(ValueError, match="shape or dtype"):
        write_raw_video_stats(tmp_path, stats, action_count=2)


def test_raw_video_stats_reject_incomplete_fields_and_atomic_write(tmp_path):
    incomplete = {"global": {"mean": np.zeros(3), "min": np.zeros(3)}}
    with pytest.raises(ValueError, match="fields"):
        write_raw_video_stats(tmp_path, incomplete, action_count=2)
    stats = {"global": {"mean": np.zeros(3), "min": np.zeros(3), "max": np.ones(3)}}
    path = write_raw_video_stats(tmp_path, stats, action_count=2)
    assert not (tmp_path / ".video_stats.json.tmp").exists()
    assert path.exists()


def test_raw_video_stats_replaces_sidecar_atomically(tmp_path, monkeypatch):
    stats = {"global": {"mean": np.zeros(3), "min": np.zeros(3), "max": np.ones(3)}}
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
