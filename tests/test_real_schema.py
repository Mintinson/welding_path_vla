"""对照服务器训练子集检查真机数据契约，不依赖该大数据集才能运行测试。"""

import json
from pathlib import Path

import pytest

from welding_path_vla.dataset.export_lerobot import (
    ABSOLUTE_ACTION_NAMES,
    GLOBAL_IMAGE,
    OBSERVATION_STATE_NAMES,
    WRIST_IMAGE,
)
from welding_path_vla.dataset.real_capture import validate_real_schema
from welding_path_vla.dataset.task_parameters import TASK_FEATURES


def info():
    """复现已核对的参考 schema；测试不复制真实数值或标签。"""
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [13],
                "names": list(OBSERVATION_STATE_NAMES),
            },
            "action": {"dtype": "float32", "shape": [9], "names": list(ABSOLUTE_ACTION_NAMES)},
            **{
                key: {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {"video.codec": "av1", "video.pix_fmt": "yuv420p"},
                }
                for key in (GLOBAL_IMAGE, WRIST_IMAGE)
            },
            **{
                key: {**value, "shape": list(value["shape"])}
                for key, value in TASK_FEATURES.items()
            },
        },
    }


def test_server_schema_matches_full_real_labels(tmp_path):
    """完整字段与视频格式匹配参考，硬件来源可以不同，不要求 phase。"""
    reference = tmp_path / "reference"
    (reference / "meta").mkdir(parents=True)
    (reference / "meta/info.json").write_text(json.dumps(info()))
    actual = info()
    actual["robot_type"] = "elfin_welding"
    validate_real_schema(actual, reference_dataset=reference)
    assert not (reference / "meta/phase_index.npz").exists()


@pytest.mark.parametrize("field,value", [("video.codec", "h264"), ("video.pix_fmt", "yuv444p")])
def test_reference_video_format_must_match(tmp_path, field, value):
    """编码不同不能再仅凭张量兼容就判定与参考格式一致。"""
    reference = tmp_path / "reference"
    (reference / "meta").mkdir(parents=True)
    (reference / "meta/info.json").write_text(json.dumps(info()))
    actual = info()
    actual["features"][GLOBAL_IMAGE]["info"][field] = value
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        validate_real_schema(actual, reference_dataset=reference)


def test_missing_direction_rejected():
    """方向是唯一必填任务标签；缺失时不使用默认值填充。"""
    actual = info()
    actual["features"].pop("task.direction")
    with pytest.raises(ValueError, match=r"task\.direction"):
        validate_real_schema(actual)
    validate_real_schema(actual, require_task_labels=False)


@pytest.mark.parametrize("with_reference", [False, True])
def test_direction_only_is_valid(tmp_path, with_reference):
    """即使参考集含工艺参数和自定义标签，真机也只强制提供方向。"""
    reference = tmp_path / "reference"
    expected = info()
    expected["features"]["task.extra.fixture"] = {
        "dtype": "string",
        "shape": [1],
        "names": ["fixture"],
    }
    (reference / "meta").mkdir(parents=True)
    (reference / "meta/info.json").write_text(json.dumps(expected))
    actual = info()
    actual["features"].pop("task.parameters")
    validate_real_schema(actual, reference_dataset=reference if with_reference else None)
    assert "task.parameters" not in actual["features"]


@pytest.mark.parametrize("require_task_labels", [False, True])
def test_provided_optional_parameters_are_validated(require_task_labels):
    """可选参数一旦提供，仍须满足固定四维契约。"""
    actual = info()
    actual["features"]["task.parameters"]["shape"] = [3]
    with pytest.raises(ValueError, match=r"task\.parameters\.shape"):
        validate_real_schema(actual, require_task_labels=require_task_labels)


@pytest.mark.parametrize(
    "field,value", [("dtype", "float64"), ("shape", [12]), ("names", ["wrong"])]
)
def test_incorrect_state_contract_rejected(field, value):
    """不仅检查维度，还检查类型与坐标字段名。"""
    actual = info()
    actual["features"]["observation.state"][field] = value
    with pytest.raises(ValueError, match=r"observation\.state"):
        validate_real_schema(actual)


def test_reference_resolution_and_fps_are_checked(tmp_path):
    """显式对照参考数据时拒绝分辨率或 FPS 不一致。"""
    reference = tmp_path / "reference"
    (reference / "meta").mkdir(parents=True)
    (reference / "meta/info.json").write_text(json.dumps(info()))
    actual = info()
    actual["features"][WRIST_IMAGE]["shape"] = [24, 32, 3]
    with pytest.raises(ValueError, match=r"observation\.images\.wrist"):
        validate_real_schema(actual, reference_dataset=reference)
    actual = info()
    actual["fps"] = 25
    with pytest.raises(ValueError, match="fps"):
        validate_real_schema(actual, reference_dataset=reference)


def test_actual_reference_dataset_when_available():
    """本机真实训练子集可直接作为 schema 校验依据。"""
    reference = Path("datasets/weldpath_lerobot_relative_small")
    if not reference.exists():
        pytest.skip("未附带服务器训练子集")
    actual = info()
    reference_info = json.loads((reference / "meta/info.json").read_text())
    for key in (GLOBAL_IMAGE, WRIST_IMAGE):
        actual["features"][key]["info"] = reference_info["features"][key]["info"].copy()
    actual["features"].update(
        {
            key: value
            for key, value in reference_info["features"].items()
            if key not in actual["features"]
        }
    )
    validate_real_schema(actual, reference_dataset=reference)
    actual["features"].pop("task.parameters")
    validate_real_schema(actual, reference_dataset=reference)
