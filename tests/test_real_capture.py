"""用严格只读的假 CPS 和双 RGB 相机检验原生采集链路。"""

import json
from hashlib import sha256
from itertools import pairwise

import av
import numpy as np
import pytest
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot_robot_welding import adapter as capture

from welding_path_vla.core.config import PolicyConfig, SceneConfig, TrainingConfig
from welding_path_vla.dataset.export_lerobot import MANIFEST_PATH
from welding_path_vla.dataset.real_capture import prepare_real_dataset
from welding_path_vla.dataset.record_conditions import (
    ConditionedRecordConfig,
    record_with_conditions,
)
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.robot.cps_reader import CPSStateReader


class ReadOnlyCPS:
    """只实现读取所需 API，其他任何调用直接失败。"""

    def __init__(self):
        self.calls = []
        self.sequence = 0
        self.tcp = [10, 20, 30, 0, 0, 0]
        self.ucs = [200, 100, 50, 0, 0, 90]
        self.error = 0

    def __getattr__(self, name):
        raise AssertionError(f"不允许调用 {name}")

    def HRIF_Connect(self, box, host, port):
        self.calls.append("connect")
        return 0

    def HRIF_DisConnect(self, box):
        self.calls.append("disconnect")
        return 0

    def HRIF_ReadActPos(self, box, robot, result):
        self.calls.append("read")
        self.sequence += 1
        # 使用官方接口的字符串输出格式，TCP/UCS 偏移与实时位姿分开返回。
        result.extend(
            map(
                str,
                [
                    self.sequence,
                    90,
                    0,
                    0,
                    0,
                    0,
                    100 + self.sequence,
                    200,
                    300,
                    0,
                    0,
                    0,
                    *self.tcp,
                    *self.ucs,
                ],
            )
        )
        return self.error


class RGBCamera:
    """用固定通道值区分两个相机，排查 RGB/BGR 或相机交换。"""

    def __init__(self, red):
        self.is_connected = False
        self.image = np.full((24, 32, 3), [red, 30, 10], dtype=np.uint8)

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def read_latest(self):
        return self.image


def test_cps_units_frames_and_calibration_changes():
    """独立检查毫米/度转换、world 外参和会话内标定固定要求。"""
    cps = ReadOnlyCPS()
    reader = CPSStateReader(
        cps, SceneConfig(robot_base_position_m=[1, 2, 3], robot_base_yaw_deg=90)
    )
    reader.connect("127.0.0.1", 10003)
    state = reader.read()
    # 非零 TCP/UCS 偏移不重复叠加；单次采样只读取一次，禁止额外正运动学查询。
    assert cps.calls == ["connect", "read"]
    np.testing.assert_allclose(state.values[:2], [np.pi / 180, np.pi / 2])
    np.testing.assert_allclose(state.values[6:9], [0.8, 2.101, 3.3])
    np.testing.assert_allclose(state.values[9:13], [2**-0.5, 0, 0, 2**-0.5])
    cps.tcp[0] += 1
    with pytest.raises(RuntimeError, match="TCP/UCS"):
        reader.read()
    reader.close()
    assert cps.calls[-1] == "disconnect"
    reader.connect("127.0.0.1", 10003)
    reader.read()  # 新会话允许使用修改后的有效标定。
    assert reader.tcp[0] == cps.tcp[0]
    cps.error = 42
    with pytest.raises(RuntimeError, match="CPS error 42"):
        reader.read()
    reader.close()


def make_robot(monkeypatch, root):
    """仅替换设备 I/O，生产状态转换与原生录制循环保持真实。"""
    cps = ReadOnlyCPS()
    cameras = {"global": RGBCamera(100), "wrist": RGBCamera(200)}
    monkeypatch.setattr(capture, "CPSClient", lambda: cps)
    monkeypatch.setattr(capture, "make_cameras_from_configs", lambda config: cameras)
    config = capture.ElfinWeldingConfig(calibration_dir=root.parent / "cal")
    for camera in config.cameras.values():
        camera.width, camera.height = 32, 24
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config

    robot = make_robot_from_config(config)
    robot.dataset_root = root
    teleop = make_teleoperator_from_config(
        capture.ElfinPassiveConfig(calibration_dir=root.parent / "cal")
    )
    return robot, teleop, cps


def test_partial_connect_failure_closes_devices(monkeypatch, tmp_path):
    """相机连接失败时关闭已连接设备，不向机械臂发送停止指令。"""
    root = tmp_path / "failed"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "robot_type": "elfin_welding"}))
    robot, _, cps = make_robot(monkeypatch, root)

    def fail():
        raise OSError("USB camera unavailable")

    monkeypatch.setattr(robot.cameras["wrist"], "connect", fail)
    with pytest.raises(OSError, match="USB"):
        robot.connect()
    assert not robot.is_connected and not robot.cameras["global"].is_connected
    assert cps.calls[-1] == "disconnect"
    assert not capture.CAPTURE_SESSIONS


def test_failed_sdk_connect_closes_client(monkeypatch, tmp_path):
    """SDK 建链失败可能已经创建通信资源，仍执行只读断开清理。"""
    root = tmp_path / "failed-sdk"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "robot_type": "elfin_welding"}))
    robot, _, cps = make_robot(monkeypatch, root)
    monkeypatch.setattr(cps, "HRIF_Connect", lambda *args: 42)
    with pytest.raises(RuntimeError, match="HRIF_Connect failed: CPS error 42"):
        robot.connect()
    assert cps.calls == ["disconnect"]
    assert not robot.is_connected


def test_native_sampling_reuses_successor_without_slow_loop(monkeypatch, tmp_path, caplog):
    """SDK 每次耗时 3ms 时，连续样本只查询一次，原生循环仍能保持 30Hz。"""
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts import lerobot_record

    from welding_path_vla.robot import cps_reader

    root = tmp_path / "timing"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "robot_type": "elfin_welding"}))
    robot, teleop, cps = make_robot(monkeypatch, root)
    now = [0.0]

    def advance(seconds):
        now[0] += seconds

    original_read = cps.HRIF_ReadActPos

    def delayed_read(*args):
        advance(0.003)
        return original_read(*args)

    monkeypatch.setattr(cps, "HRIF_ReadActPos", delayed_read)
    monkeypatch.setattr(capture, "perf_counter", lambda: now[0])
    monkeypatch.setattr(capture, "sleep", advance)
    monkeypatch.setattr(cps_reader, "perf_counter", lambda: now[0])
    monkeypatch.setattr(lerobot_record.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(lerobot_record, "precise_sleep", advance)

    def record(config):
        robot.connect()
        teleop.connect()
        events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
        count = [0]

        def count_action(action):
            count[0] += 1
            events["exit_early"] = count[0] == 4
            return action

        monkeypatch.setattr(robot, "send_action", count_action)
        try:
            processors = make_default_processors()
            lerobot_record.record_loop(
                robot, events, 30, *processors, teleop=teleop, control_time_s=5
            )
        finally:
            robot.disconnect()
            teleop.disconnect()

    original_loop = lerobot_record.record_loop
    monkeypatch.setattr(lerobot_record, "record", record)
    record_with_conditions(
        ConditionedRecordConfig(
            robot=robot.config,
            teleop=teleop.config,
            dataset=DatasetRecordConfig(root=root),
            require_task_labels=False,
        )
    )
    rows = [
        row
        for path in (root / "meta/real_sessions").glob("*.jsonl")
        for line in path.read_text().splitlines()
        if (row := json.loads(line))["kind"] == "sample"
    ]
    assert len(rows) == 4
    assert cps.calls.count("read") == 6  # 标定一次、首个观测一次、四个后继各一次。
    assert lerobot_record.record_loop is original_loop
    for previous, current in pairwise(rows):
        assert previous["target_request_s"] == current["source_request_s"]
        assert current["source_request_s"] - previous["source_request_s"] == pytest.approx(1 / 30)
    assert not any("Record loop is running slower" in row.message for row in caplog.records)


def test_pairing_boundaries_preserve_images_and_clear_on_failure(monkeypatch, tmp_path):
    """首帧使用本回合快照；图像缓冲复用及异常退出均不污染下一回合。"""
    root = tmp_path / "boundaries"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "robot_type": "elfin_welding"}))
    robot, teleop, cps = make_robot(monkeypatch, root)
    robot.connect()
    teleop.connect()
    try:
        for red in (100, 200):
            robot.cameras["global"].image[:] = [red, 30, 10]
            with pytest.raises(RuntimeError, match="模拟循环中断"), robot.observation_pairs():
                primed_sequence = cps.sequence
                robot.cameras["global"].image[:] = [red + 1, 30, 10]
                observation = robot.get_observation()
                action = teleop.get_action()
                assert observation["joint_1"] == pytest.approx(np.deg2rad(primed_sequence))
                assert observation["global"][0, 0, 0] == red
                assert action["y"] == pytest.approx(observation["tcp_y"] - 0.001, abs=1e-6)
                raise RuntimeError("模拟循环中断")
            assert robot.source is None and robot.target is None
            cps.sequence += 10  # 模拟保存或重置期间机械臂继续运动。
    finally:
        robot.disconnect()
        teleop.disconnect()


def run_frames(robot, teleop, dataset=None, *, discard=False):
    """在原生循环内通过真实 events 结束，避免依赖运行时间决定样本数。"""
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    original = robot.send_action
    frames = []

    def count_action(action):
        frames.append(action)
        if len(frames) == 4:
            events["exit_early"] = True
            events["rerecord_episode"] = discard
        return original(action)

    robot.send_action = count_action
    try:
        teleop_processor, robot_processor, observation_processor = make_default_processors()
        with robot.observation_pairs():
            record_loop(
                robot,
                events,
                30,
                teleop_processor,
                robot_processor,
                observation_processor,
                dataset=dataset,
                teleop=teleop,
                control_time_s=5,
                single_task="Follow seam",
            )
    finally:
        robot.send_action = original
    return events


def test_native_record_discard_reset_resume_and_prepare(monkeypatch, tmp_path):
    """验证原生回合缓冲、恢复、实测后继动作和训练契约的完整链路。"""
    root = tmp_path / "recorded"
    robot, teleop, cps = make_robot(monkeypatch, root)
    from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features

    teleop_processor, _, observation_processor = make_default_processors()
    features = {
        **aggregate_pipeline_dataset_features(
            teleop_processor, create_initial_features(action=robot.action_features)
        ),
        **aggregate_pipeline_dataset_features(
            observation_processor, create_initial_features(observation=robot.observation_features)
        ),
    }
    dataset = LeRobotDataset.create(
        "test/real",
        fps=30,
        root=root,
        robot_type=robot.name,
        features=features,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast"),
    )
    try:
        robot.connect()
        teleop.connect()
        events = run_frames(robot, teleop, dataset, discard=True)
        assert events["rerecord_episode"]
        dataset.clear_episode_buffer()
        run_frames(robot, teleop, dataset)
        dataset.save_episode()
        run_frames(robot, teleop)  # 原生命令的重置阶段仍读取，但不写数据。
        run_frames(robot, teleop, dataset)
        dataset.save_episode()
    finally:
        dataset.finalize()
        robot.disconnect()
        teleop.disconnect()
    assert cps.calls[-1] == "disconnect"
    assert not capture.CAPTURE_SESSIONS

    # 实际恢复官方 writer，再录制一个回合；新会话必须重新读取有效标定。
    resumed = LeRobotDataset.resume(
        "test/real",
        root=root,
        streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast"),
    )
    try:
        robot.connect()
        teleop.connect()
        run_frames(robot, teleop, resumed)
        resumed.save_episode()
    finally:
        resumed.finalize()
        robot.disconnect()
        teleop.disconnect()

    import pyarrow.parquet as pq

    tables = [pq.read_table(file) for file in sorted((root / "data").rglob("*.parquet"))]
    rows = [row for table in tables for row in table.to_pylist()]
    assert len(rows) == 12
    for name, red in (("global", 100), ("wrist", 200)):
        video = next((root / f"videos/observation.images.{name}").rglob("*.mp4"))
        with av.open(str(video)) as container:
            image = next(container.decode(video=0)).to_ndarray(format="rgb24")
        assert image.shape == (24, 32, 3)
        np.testing.assert_allclose(image.mean(axis=(0, 1)), [red, 30, 10], atol=5)
    for row in rows:
        assert len(row["observation.state"]) == 13 and len(row["action"]) == 9
        # Base 的 X 增加 1mm；默认 -90deg 的安装旋转将其映射到 world -Y。
        assert row["action"][1] == pytest.approx(row["observation.state"][7] - 0.001, abs=1e-6)
    media = {
        path: sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.suffix in {".parquet", ".mp4"}
    }
    report = prepare_real_dataset(root, action_horizon=2, require_task_labels=False)
    assert report["episodes"] == 3 and report["frames"] == 12
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in media.items())
    manifest = json.loads((root / MANIFEST_PATH).read_text())
    assert manifest["action_source"] == "next_measured_tcp"
    statistics = json.loads((root / "meta/stats.json").read_text())["action"]
    np.testing.assert_allclose(statistics["mean"][:3], [0.0015, 0, 0], atol=1e-6)
    checked = validate_dataset(
        TrainingConfig(dataset_repo_id="test/real", dataset_root=str(root)),
        PolicyConfig(action_horizon=2),
    )
    assert checked.frames == 12 and checked.episodes == 3
    reopened = LeRobotDataset("test/real", root=root, video_backend="pyav")
    sample = reopened[0]
    assert sample["observation.state"].shape == (13,)
    assert sample["action"].shape == (9,)
    assert sample["observation.images.global"].shape == (3, 24, 32)
    assert sample["observation.images.wrist"].shape == (3, 24, 32)
    from welding_path_vla.dataset.rerender import rerender_lerobot_dataset

    with pytest.raises(ValueError, match="真机图像"):
        rerender_lerobot_dataset(root)
    assert prepare_real_dataset(root, action_horizon=2, require_task_labels=False) == report
    with pytest.raises(ValueError, match="至少需要"):
        prepare_real_dataset(root, action_horizon=30, require_task_labels=False)
