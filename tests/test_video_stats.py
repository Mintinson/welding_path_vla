import json

import numpy as np
import pytest

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


def test_raw_video_stats_round_trip_and_validate_contract(tmp_path):
    stats = {"global": {"mean": np.array([1.0, 2.0, 3.0]), "min": np.array([0, 0, 0]), "max": np.array([4, 5, 6])}}
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
    stats = {"global": {"mean": np.array([1.0, 2.0, 3.0]), "max": np.array([4, 5, 6])}}
    write_raw_video_stats(tmp_path, stats, action_count=2)
    with pytest.raises(ValueError, match="camera"):
        read_raw_video_stats(tmp_path, ("global", "wrist"), action_count=2)
    bad = {"global": {"mean": np.array([np.nan, 2.0, 3.0]), "min": np.array([0, 0, 0]), "max": np.array([4, 5, 6])}}
    with pytest.raises(ValueError, match="finite"):
        write_raw_video_stats(tmp_path, bad, action_count=2)
