"""验证显式任务条件通过原生录制、恢复及训练提示链路保存。"""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import draccus
import numpy as np
import pytest
import yaml
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_record
from test_real_capture import make_robot

from welding_path_vla.core.config import PolicyConfig, TrainingConfig
from welding_path_vla.dataset.real_capture import prepare_real_dataset
from welding_path_vla.dataset.record_conditions import (
    ConditionedRecordConfig,
    RecordingConditions,
    record_with_conditions,
    recording_conditions,
)
from welding_path_vla.policies.data import validate_dataset
from welding_path_vla.policies.welding_prompt import WeldingPromptBuilder


def test_condition_config_decoding_and_optional_labels():
    """配置与命令行可以覆盖方向和任意自定义标量，空配置不虚构标签。"""
    assert RecordingConditions().values() == {}
    config = draccus.parse(
        RecordingConditions,
        args=["--direction=reverse", "--extra={fixture: jig_a, trial: 2, dry_run: true, gap: 0.3}"],
    )
    values = config.values()
    assert values["task.direction"].tolist() == [1]
    assert config.features()["task.direction"]["dtype"] == "int64"
    assert values["task.extra.fixture"] == "jig_a"
    assert values["task.extra.trial"].tolist() == [2]
    assert values["task.extra.dry_run"].tolist() == [True]
    np.testing.assert_allclose(values["task.extra.gap"], [0.3])
    assert "task.parameters" not in values


@pytest.mark.parametrize(
    "conditions",
    [
        RecordingConditions(direction="sideways"),
        RecordingConditions(parameters={"welding_speed_mps": 0.02}),
        RecordingConditions(extra={"gap": float("nan")}),
        RecordingConditions(extra={"nested": {"value": 1}}),
        RecordingConditions(extra={"bad/name": 1}),
    ],
)
def test_invalid_conditions_fail_before_recording(conditions, monkeypatch):
    """错误标签应在接机或创建数据之前报错。"""
    called = []
    monkeypatch.setattr(lerobot_record, "record", lambda config: called.append(config))
    with pytest.raises(ValueError), recording_conditions(conditions):
        lerobot_record.record(None)
    assert not called


def test_condition_hooks_restore_after_failure():
    """包装只对当前调用生效，即使录制异常也恢复原生函数。"""
    aggregate = lerobot_record.aggregate_pipeline_dataset_features
    build_frame = lerobot_record.build_dataset_frame
    with (
        pytest.raises(RuntimeError),
        recording_conditions(RecordingConditions(direction="forward")),
    ):
        raise RuntimeError("模拟中断")
    assert lerobot_record.aggregate_pipeline_dataset_features is aggregate
    assert lerobot_record.build_dataset_frame is build_frame


@pytest.mark.parametrize("with_parameters", [False, True])
def test_native_record_conditions_resume_and_training(monkeypatch, tmp_path, with_parameters):
    """原生 record 写入不同方向回合，并能直接读回、整理和生成训练提示。"""
    root = tmp_path / "conditioned"
    # 模拟适配器曾指向旧目录；原生 dataset.root 必须决定本次全部输出的位置。
    old_root = tmp_path / "old_capture_root"
    robot, teleop, cps = make_robot(monkeypatch, old_root)
    if with_parameters:
        for camera in robot.config.cameras.values():
            camera.width, camera.height = 640, 480
        for camera in robot.cameras.values():
            camera.image = np.full((480, 640, 3), camera.image[0, 0], dtype=np.uint8)
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda config: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda config: teleop)
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda: (None, events))
    original = robot.send_action
    count = [0]
    frames_per_episode = 32 if with_parameters else 4
    horizon = 30 if with_parameters else 2

    def send_action(action):
        count[0] += 1
        if count[0] % frames_per_episode == 0:
            events["exit_early"] = True
        return original(action)

    monkeypatch.setattr(robot, "send_action", send_action)
    parameters = (
        {
            "welding_speed_mps": 0.02,
            "work_angle_deg": 45.0,
            "travel_angle_deg": 10.0,
            "tool_roll_deg": -20.0,
        }
        if with_parameters
        else {}
    )
    config = ConditionedRecordConfig(
        robot=robot.config,
        teleop=teleop.config,
        dataset=DatasetRecordConfig(
            repo_id="test/conditions",
            root=root,
            single_task="Follow seam",
            num_episodes=1,
            episode_time_s=5,
            fps=30,
            push_to_hub=False,
            streaming_encoding=True,
            encoder_threads=1,
            rgb_encoder=(
                RGBEncoderConfig(vcodec="av1")
                if with_parameters
                else RGBEncoderConfig(vcodec="h264", preset="fast")
            ),
        ),
        play_sounds=False,
        task_conditions=RecordingConditions(
            direction="forward",
            parameters=parameters,
            extra={"fixture": "jig_a", "trial": 1, "dry_run": True, "gap": 0.3},
        ),
    )
    record_with_conditions(config)
    assert cps.calls[-1] == "disconnect"
    assert (root / "meta/real_capture_contract.json").is_file()
    assert len(list((root / "meta/real_sessions").glob("*.jsonl"))) == 1
    assert not old_root.exists()
    config = replace(
        config,
        resume=True,
        task_conditions=replace(
            config.task_conditions,
            direction="reverse",
            extra={"fixture": "jig_b", "trial": 2, "dry_run": False, "gap": 0.4},
        ),
    )
    events.update(exit_early=False, stop_recording=False, rerecord_episode=False)
    record_with_conditions(config)
    reference = Path("datasets/weldpath_lerobot_relative_small")
    comparison = reference if with_parameters and reference.exists() else None
    prepare_real_dataset(
        root,
        action_horizon=horizon,
        reference_dataset=comparison,
    )
    fields = (
        ("direction", "welding_speed", "work_angle", "travel_angle", "tool_roll")
        if with_parameters
        else ("direction",)
    )
    if comparison is not None:
        import pyarrow.parquet as pq

        expected = pq.read_schema(next((reference / "data").rglob("*.parquet")))
        actual = pq.read_schema(next((root / "data").rglob("*.parquet")))
        for item in expected:
            assert actual.field(item.name).type == item.type
        expected_manifest = json.loads(
            (reference / "meta/welding_path_vla_export.json").read_text()
        )
        actual_manifest = json.loads((root / "meta/welding_path_vla_export.json").read_text())
        assert (
            actual_manifest["action_representation"] == expected_manifest["action_representation"]
        )
        with pytest.raises(ValueError, match="动作契约"):
            prepare_real_dataset(root, action_horizon=2, reference_dataset=reference)
        import av

        reference_info = json.loads((reference / "meta/info.json").read_text())
        actual_info = json.loads((root / "meta/info.json").read_text())
        for key in ("observation.images.global", "observation.images.wrist"):
            assert actual_info["features"][key]["info"] == reference_info["features"][key]["info"]
            with av.open(str(next((root / "videos" / key).rglob("*.mp4")))) as container:
                # 解码器实现可能名为 libdav1d，按码流类型 ID 判断是否为 AV1。
                assert container.streams.video[0].codec_context.codec.id == av.Codec("av1", "r").id
                assert next(container.decode(video=0)).to_ndarray(format="rgb24").shape == (
                    480,
                    640,
                    3,
                )
    report = validate_dataset(
        TrainingConfig(dataset_repo_id=config.dataset.repo_id, dataset_root=str(root)),
        PolicyConfig(action_horizon=horizon, welding_prompt_fields=fields),
    )
    assert report.frames == 2 * frames_per_episode and report.episodes == 2
    reopened = LeRobotDataset(config.dataset.repo_id, root=root, video_backend="pyav")
    for index, direction, fixture in ((0, 0, "jig_a"), (frames_per_episode, 1, "jig_b")):
        sample = reopened[index]
        assert sample["observation.state"].shape == (13,)
        assert sample["action"].shape == (9,)
        assert sample["task.direction"].item() == direction
        assert sample["task.extra.fixture"] == fixture
        assert sample["task.extra.trial"].item() == direction + 1
        assert sample["task.extra.dry_run"].item() == (direction == 0)
        assert sample["task.extra.gap"].item() == pytest.approx(0.3 + direction * 0.1)
        if with_parameters:
            np.testing.assert_allclose(sample["task.parameters"], [0.02, 45, 10, -20])
        else:
            assert "task.parameters" not in sample
        prompt = WeldingPromptBuilder(fields=fields).complementary_data(sample)["task"]
        assert f"Move in the {('forward', 'reverse')[direction]} direction." in prompt
        assert "welding speed" in prompt if with_parameters else "welding speed" not in prompt

    # 改变字段名/类型时沿用原生恢复检查，须在接机前拒绝，不能给旧回合补虚构标签。
    calls_before = list(cps.calls)
    bad = replace(config, task_conditions=RecordingConditions(direction="forward"))
    with pytest.raises(ValueError):
        record_with_conditions(bad)
    assert cps.calls == calls_before
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_episodes"] == 2


def test_record_yaml_uses_native_fields_and_condition_extension():
    """默认 YAML 经薄配置扩展解码，其余字段仍是原生类型。"""
    config = draccus.decode(
        ConditionedRecordConfig, yaml.safe_load(Path("configs/real_record.yaml").read_text())
    )
    assert isinstance(config.dataset, DatasetRecordConfig)
    assert config.robot.type == "elfin_welding"
    assert config.require_task_labels
    assert config.task_conditions.direction in ("forward", "reverse")
    assert "task.parameters" not in config.task_conditions.values()
    assert config.dataset.rgb_encoder.vcodec == "libsvtav1"
    assert "capture_root" not in draccus.encode(config.robot)


def test_native_default_root_and_binding_restore_after_connect_failure(monkeypatch, tmp_path):
    """省略 root 时跟随原生时间戳目录，连接失败也恢复临时适配。"""
    from lerobot.datasets import dataset_metadata
    from lerobot_robot_welding import ElfinWelding

    from welding_path_vla.dataset import record_conditions

    robot, teleop, _ = make_robot(monkeypatch, tmp_path / "old_root")
    cache = tmp_path / "cache"
    monkeypatch.setattr(dataset_metadata, "HF_LEROBOT_HOME", cache)
    monkeypatch.setattr(record_conditions, "HF_LEROBOT_HOME", cache)
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda config: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda config: teleop)

    def fail_connect(*args):
        raise RuntimeError("模拟设备连接失败")

    monkeypatch.setattr(robot.reader, "connect", fail_connect)
    connect, loop = ElfinWelding.connect, lerobot_record.record_loop
    config = ConditionedRecordConfig(
        robot=robot.config,
        teleop=teleop.config,
        dataset=DatasetRecordConfig(
            repo_id="test/default", single_task="Follow seam", push_to_hub=False
        ),
        require_task_labels=False,
        play_sounds=False,
    )
    with pytest.raises(RuntimeError, match="模拟设备连接失败"):
        record_with_conditions(config)
    assert config.dataset.repo_id.startswith("test/default_")
    assert robot.dataset_root == cache / config.dataset.repo_id
    assert (robot.dataset_root / "meta/info.json").is_file()
    assert not (tmp_path / "old_root").exists()
    assert ElfinWelding.connect is connect
    assert lerobot_record.record_loop is loop


def test_record_cli_overrides_yaml_without_connecting(monkeypatch):
    """用户命令经过公开入口解析，方向和自定义字段正确传给薄封装。"""
    from welding_path_vla.dataset import record_conditions

    captured = []
    monkeypatch.setattr(record_conditions, "record_with_conditions", captured.append)
    monkeypatch.syspath_prepend("scripts")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/record_real.py",
            "--config_path=configs/real_record.yaml",
            "--robot.host=192.168.56.101",
            "--task_conditions.direction=reverse",
            "--task_conditions.extra={fixture: jig_a, trial: 1, dry_run: true}",
        ],
    )
    runpy.run_path("scripts/record_real.py", run_name="__main__")
    config = captured[0]
    assert config.task_conditions.direction == "reverse"
    assert config.task_conditions.extra == {"fixture": "jig_a", "trial": 1, "dry_run": True}
    assert config.robot.host == "192.168.56.101"


def test_formal_record_requires_only_direction_before_native_call(monkeypatch):
    """正式采集仅要求方向，parameters 和 extra 均可省略。"""
    from lerobot_robot_welding import ElfinPassiveConfig, ElfinWeldingConfig

    called = []
    monkeypatch.setattr(lerobot_record, "record", called.append)
    config = ConditionedRecordConfig(
        robot=ElfinWeldingConfig(),
        teleop=ElfinPassiveConfig(),
        dataset=DatasetRecordConfig(),
    )
    with pytest.raises(ValueError, match=r"task\.direction"):
        record_with_conditions(config)
    assert not called
    record_with_conditions(
        replace(config, task_conditions=RecordingConditions(direction="forward"))
    )
    assert len(called) == 1
    record_with_conditions(replace(config, require_task_labels=False))
    assert len(called) == 2
