"""为原生录制的真机 LeRobot 数据补齐本项目训练契约。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as arrow_dataset

from welding_path_vla.dataset.actions import ABSOLUTE_ACTION_NAMES
from welding_path_vla.dataset.export_lerobot import (
    ACTION_REPRESENTATION,
    ACTION_STORAGE,
    GLOBAL_IMAGE,
    MANIFEST_PATH,
    OBSERVATION_STATE_NAMES,
    WRIST_IMAGE,
    recompute_relative_action_stats,
)
from welding_path_vla.dataset.task_parameters import TASK_DIRECTION, TASK_FEATURES, TASK_PARAMETERS

CAPTURE_CONTRACT = Path("meta/real_capture_contract.json")


def validate_real_schema(
    info: dict[str, Any],
    require_task_labels: bool = True,
    reference_dataset: str | Path | None = None,
) -> None:
    """核对真机特征契约，可额外使用服务器训练集的 info.json 做逐字段对照。

    Args:
        info: 待检查数据集的原生 meta/info.json 内容。
        require_task_labels: 正式数据必须包含方向；工艺参数和自定义标签可省略。
        reference_dataset: 可选参考数据集；检查 dtype、shape、names、FPS 和视频编码信息，
            不要求硬件来源、episode 数量或可选 phase sidecar 相同。
    """
    expected: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (13,),
            "names": list(OBSERVATION_STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(ABSOLUTE_ACTION_NAMES),
        },
    }
    expected.update(
        {
            key: value
            for key, value in TASK_FEATURES.items()
            if key in info["features"] or (key == TASK_DIRECTION and require_task_labels)
        }
    )
    reference = None
    if reference_dataset is not None:
        reference = json.loads((Path(reference_dataset) / "meta/info.json").read_text())
        for key in ("codebase_version", "fps"):
            if info.get(key) != reference[key]:
                raise ValueError(f"{key} 与参考数据集不一致")
        # 参考集中的可选任务标签不强制补齐；已提供的字段仍须匹配参考定义。
        expected.update(
            {
                key: value
                for key, value in reference["features"].items()
                if not key.startswith("task.")
                or key in info["features"]
                or (key == TASK_DIRECTION and require_task_labels)
            }
        )
    for key, definition in expected.items():
        actual = info["features"].get(key)
        if actual is None:
            raise ValueError(f"缺少训练字段 {key}; 不允许用默认值补充未知标签")
        for field in ("dtype", "shape", "names"):
            value, target = actual.get(field), definition[field]
            if field == "shape":
                value, target = tuple(value or ()), tuple(target)
            if value != target:
                raise ValueError(f"{key}.{field} 与训练契约不一致: {value!r} != {target!r}")
    for key in (GLOBAL_IMAGE, WRIST_IMAGE):
        image = info["features"].get(key, {})
        shape = image.get("shape", [])
        if image.get("dtype") not in {"video", "image"} or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"{key} 必须是 HxWx3 RGB 视频或图像")
        if reference is not None:
            for name, target in reference["features"][key].get("info", {}).items():
                value = image.get("info", {}).get(name)
                if value != target:
                    raise ValueError(f"{key}.{name} 与参考视频格式不一致: {value!r} != {target!r}")


def prepare_real_dataset(
    root: str | Path,
    action_horizon: int = 30,
    action_stride: int = 1,
    require_task_labels: bool = True,
    reference_dataset: str | Path | None = None,
) -> dict[str, int | str]:
    """校验真机数值数据并重算 relative action 统计，不改写视频或动作。

    Args:
        root: 原生 lerobot-record 保存的数据集根目录。
        action_horizon: 与训练配置一致的预测长度。
        action_stride: 预测块中相邻动作的帧间隔。
        require_task_labels: 正式数据要求方向标签；其余标签仅在提供时校验。
        reference_dataset: 可选的参考数据集，用于校验实际训练 schema。

    Returns:
        已整理的数据集位置、回合数、帧数和预测长度。
    """
    root = Path(root)
    if action_horizon < 1 or action_stride < 1:
        raise ValueError("action_horizon 和 action_stride 必须为正数")
    info = json.loads((root / "meta/info.json").read_text())
    if not info["total_episodes"]:
        raise ValueError("数据集中尚无已保存的 episode")
    contract = json.loads((root / CAPTURE_CONTRACT).read_text())
    if (
        info["robot_type"] != "elfin_welding"
        or contract["action_source"] != "next_measured_tcp"
        or info["fps"] != contract["fps"]
    ):
        raise ValueError("数据集不是本项目的真机实测后继目标采集结果")
    validate_real_schema(info, require_task_labels, reference_dataset)
    representation = {
        "type": ACTION_REPRESENTATION,
        "storage": ACTION_STORAGE,
        "frame": "prediction_tcp",
        "rotation": "rotation_6d_rows",
        "horizon": action_horizon,
        "stride": action_stride,
    }
    if reference_dataset is not None:
        reference_manifest = json.loads((Path(reference_dataset) / MANIFEST_PATH).read_text())
        if reference_manifest["action_representation"] != representation:
            raise ValueError("动作契约与参考数据集不一致, 请核对 horizon/stride 及表示方式")

    # 逐批扫描数值列，不解码或重复编码视频，不将整个数据集载入内存。
    files = sorted((root / "data").rglob("*.parquet"))
    task_columns = [key for key in TASK_FEATURES if key in info["features"]]
    columns = ["episode_index", "frame_index", "observation.state", "action", *task_columns]
    scanner = arrow_dataset.dataset(files, format="parquet").scanner(columns=columns)
    lengths: dict[int, int] = {}
    count = 0
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            episode, frame = int(row["episode_index"]), int(row["frame_index"])
            state, action = row["observation.state"], row["action"]
            if frame != lengths.get(episode, 0):
                raise ValueError("episode 内 frame_index 不连续, 请检查中断的录制")
            values = np.asarray([*state, *action])
            if not np.isfinite(values).all() or not np.isclose(np.linalg.norm(state[9:13]), 1):
                raise ValueError("状态或动作含无效数值/四元数")
            if TASK_DIRECTION in row and np.asarray(row[TASK_DIRECTION]).item() not in (0, 1):
                raise ValueError("task.direction 必须是 forward=0 或 reverse=1")
            if TASK_PARAMETERS in row:
                parameters = np.asarray(row[TASK_PARAMETERS], dtype=float)
                if parameters.shape != (4,) or not np.isfinite(parameters).all():
                    raise ValueError("task.parameters 必须为四个明确提供的有限数值")
            lengths[episode] = frame + 1
            count += 1
    if count != info["total_frames"] or len(lengths) != info["total_episodes"]:
        raise ValueError("录制帧数与 LeRobot 元数据不一致")
    required_frames = 1 + (action_horizon - 1) * action_stride
    if not any(length >= required_frames for length in lengths.values()):
        raise ValueError(
            f"没有足够长的回合计算 relative action 统计, 至少需要 {required_frames} 帧"
        )

    manifest_path = root / MANIFEST_PATH
    manifest_path.unlink(missing_ok=True)
    recompute_relative_action_stats(root, action_horizon, action_stride)
    manifest = {
        "format_version": 2,
        "source_kind": "real_passive",
        "action_source": "next_measured_tcp",
        "action_representation": representation,
        "capture_contract": str(CAPTURE_CONTRACT),
        "session_logs": "meta/real_sessions/*.jsonl",
        "video_storage": {"mode": "encoded"},
        "total_frames": count,
        "total_episodes": len(lengths),
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(manifest_path)
    return {
        "dataset": str(root),
        "episodes": len(lengths),
        "frames": count,
        "action_horizon": action_horizon,
        "action_stride": action_stride,
    }
