"""接触审计不允许旧过滤器忽略浅接触，并保留 episode 删除映射。"""

import runpy

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from welding_path_vla.dataset.contact_audit import (
    filtered_manifest,
    robot_workpiece_contacts,
    stored_episodes,
)


def test_shallow_contact_without_force_and_static_robot_base():
    """静态机器人底座与工件浅接触也要命中，桌面接触不算机器人接触。"""
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="table" type="box" size="1 1 .05" pos="0 0 -.05"/>
      <body name="robot"><geom name="tip" type="sphere" size=".001"
             pos="0 0 .01099"/></body>
      <body name="workpiece"><freejoint/><geom name="plate" type="box"
             size=".1 .1 .01"/></body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    contacts = robot_workpiece_contacts(model, data, {2}, {1})
    assert len(contacts) == 1
    assert contacts[0][:2] == ("tip", "plate")
    assert np.isclose(contacts[0][2], 1e-5)
    assert robot_workpiece_contacts(model, data, {2}, set()) == []
    data.qpos[2] = -0.02
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    assert robot_workpiece_contacts(model, data, {2}, {1}) == []


def test_filter_manifest_global_mapping():
    """删除跨任务的旧全局索引，保留动作表示契约和原始清单。"""
    manifest = {
        "sources": {"a": ["a0", "a1"], "b": ["b0", "b1"]},
        "action_representation": {"horizon": 30},
    }
    filtered = filtered_manifest(manifest, {1, 2})
    assert filtered["sources"] == {"a": ["a0"], "b": ["b1"]}
    assert manifest["sources"]["a"] == ["a0", "a1"]
    assert filtered["action_representation"] == manifest["action_representation"]


def test_stored_episodes_cross_files(tmp_path):
    """跨 parquet 文件拼接一条 episode 时，数值与顺序必须保持一致。"""
    data = tmp_path / "data"
    data.mkdir()
    for file_index, (indices, episodes) in enumerate([([0, 1], [0, 0]), ([2, 3], [0, 1])]):
        pq.write_table(
            pa.table(
                {
                    "index": indices,
                    "episode_index": episodes,
                    "observation.state": [[float(i), 0.0] for i in indices],
                    "action": [[float(i)] for i in indices],
                }
            ),
            data / f"file-{file_index:03d}.parquet",
        )
    result = list(stored_episodes(tmp_path))
    assert [entry[0] for entry in result] == [0, 1]
    np.testing.assert_equal(result[0][1], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_equal(result[1][2], [[3.0]])


def test_video_remux_preserves_images_and_rejects_nonkeyframe(tmp_path):
    """保留视频逐像素不变，不能从依赖旧帧的非关键帧开始裁剪。"""
    import av
    from test_export_lerobot import write_h264_video

    from welding_path_vla.dataset.video_filter import remux_video_ranges

    source, destination = tmp_path / "source.mp4", tmp_path / "filtered.mp4"
    write_h264_video(source, 6, 20)
    remux_video_ranges(source, destination, [(0, 6)], 20)
    with av.open(str(source)) as original, av.open(str(destination)) as filtered:
        for first, second in zip(original.decode(video=0), filtered.decode(video=0), strict=True):
            np.testing.assert_equal(first.to_ndarray(), second.to_ndarray())
    with pytest.raises(ValueError, match="keyframe"):
        remux_video_ranges(source, tmp_path / "invalid.mp4", [(1, 4)], 20)


@pytest.mark.parametrize("lossless", [False, True])
def test_official_cleanup_keeps_phase_manifest_and_relative_stats(tmp_path, lossless):
    """官方删除 API 的数值重排与本项目 phase、relative-action 契约保持一致。"""
    import json

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from test_export_lerobot import write_raw_episode

    from welding_path_vla.dataset.export_lerobot import export_lerobot
    from welding_path_vla.dataset.phase_index import load_phase_index, phase_index_path

    raw, source, destination, report = [tmp_path / name for name in ("raw", "v5", "v6", "report")]
    paths = [write_raw_episode(raw, index) for index in range(3)]
    export_lerobot(raw, source, "test/audit-source", action_horizon=2, show_progress=False)
    metadata = LeRobotDatasetMetadata("test/audit-source", root=source)
    manifest = json.loads((source / "meta/welding_path_vla_export.json").read_text())
    results = [
        {
            "old_episode_index": index,
            "remove": index == 1,
            "raw_path": str(path),
            "from_index": index * 3,
            "to_index": (index + 1) * 3,
        }
        for index, path in enumerate(paths)
    ]
    original_stats = (source / "meta/stats.json").read_bytes()
    report.mkdir()
    cleanup = runpy.run_path("scripts/clean_contact_episodes.py")["create_cleaned_dataset"]
    cleanup(source, destination, report, results, manifest, "test/audit-cleaned", lossless)
    cleaned = LeRobotDatasetMetadata("test/audit-cleaned", root=destination)
    assert (cleaned.total_episodes, cleaned.total_frames) == (2, 6)
    assert metadata.total_episodes == 3
    assert (source / "meta/stats.json").read_bytes() == original_stats
    assert load_phase_index(phase_index_path(destination)).tolist() == [0, 1, 2, 0, 1, 2]
    filtered = json.loads((destination / "meta/welding_path_vla_export.json").read_text())
    assert list(filtered["sources"].values()) == [["episode_000000", "episode_000002"]]
    stats = json.loads((destination / "meta/stats.json").read_text())
    # 绝对目标 x 最小为 .01；relative-action 最小为相对当前 TCP 的 .01。
    assert np.isclose(stats["action"]["max"][0], 0.02)
