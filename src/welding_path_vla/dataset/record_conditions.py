"""在原生录制流程中附加用户指定的任务条件，不复制录制循环。"""

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.scripts import lerobot_record
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.utils.constants import HF_LEROBOT_HOME

from welding_path_vla.dataset.task_parameters import (
    DIRECTION_CODES,
    TASK_DIRECTION,
    TASK_FEATURES,
    TASK_PARAMETER_NAMES,
    TASK_PARAMETERS,
)


@dataclass
class RecordingConditions:
    """一次运行内固定、由用户保证与实际示范一致的任务条件。

    Attributes:
        direction: forward/reverse；未填写时不生成方向标签。
        parameters: 仿真已有的四项参数，名称和单位见 TASK_PARAMETER_NAMES。
            使用时需全部明确填写，空字典表示未知，不写入 task.parameters。
        extra: 任意自定义名称的数值、布尔或文本，保存为 task.extra.<名称>。
            这些字段仅保存标签，不会自动成为训练提示或机器人指令。
    """

    direction: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def values(self) -> dict[str, Any]:
        """按仿真契约编码已知参数；自定义字段保留标量类型。"""
        values: dict[str, Any] = {}
        if self.direction is not None:
            if self.direction not in DIRECTION_CODES:
                raise ValueError("direction 必须是 forward 或 reverse")
            values[TASK_DIRECTION] = np.asarray([DIRECTION_CODES[self.direction]], dtype=np.int64)
        if self.parameters:
            if set(self.parameters) != set(TASK_PARAMETER_NAMES):
                raise ValueError(f"parameters 必须完整填写 {TASK_PARAMETER_NAMES}, 未知时请留空")
            parameters = np.asarray(
                [self.parameters[name] for name in TASK_PARAMETER_NAMES], dtype=np.float32
            )
            if not np.isfinite(parameters).all():
                raise ValueError("parameters 必须是有限数值")
            values[TASK_PARAMETERS] = parameters
        for name, value in self.extra.items():
            if not name or "/" in name:
                raise ValueError("extra 字段名不能为空或包含 /")
            if isinstance(value, str):
                encoded = value
            elif type(value) in (bool, int, float):
                dtype = {bool: np.bool_, int: np.int64, float: np.float32}[type(value)]
                encoded = np.asarray([value], dtype=dtype)
                if not np.isfinite(encoded).all():
                    raise ValueError(f"extra.{name} 必须是有限数值")
            else:
                raise ValueError(f"extra.{name} 仅支持数值、布尔或文本标量")
            values[f"task.extra.{name}"] = encoded
        return values

    def features(self) -> dict[str, dict]:
        """构造原生 Dataset schema，保证方向和参数与仿真字段完全一致。"""
        features = {}
        for key, value in self.values().items():
            if key in TASK_FEATURES:
                features[key] = dict(TASK_FEATURES[key])
            else:
                features[key] = {
                    "dtype": "string" if isinstance(value, str) else str(value.dtype),
                    "shape": (1,),
                    "names": [key.removeprefix("task.extra.")],
                }
        return features


@dataclass
class ConditionedRecordConfig(RecordConfig):
    """原生 RecordConfig 的薄扩展，其余配置保持原生含义。

    Attributes:
        require_task_labels: 正式训练数据仅要求方向；参数和自定义标签可省略。
            仅联调时可关闭方向检查。
    """

    task_conditions: RecordingConditions = field(default_factory=RecordingConditions)
    require_task_labels: bool = True


@contextmanager
def recording_conditions(conditions: RecordingConditions):
    """仅在本次录制内给原生 schema 和每帧数据增加标签。

    LeRobot 0.6.0 的 RecordConfig 没有任意附加字段入口；将字段直接塞入
    observation_features 会改变 13D 状态。这里只适配录制模块引用的两个
    数据组装函数，退出时恢复，不修改安装文件或 Dataset 写入/回合管理逻辑。
    """
    values, features = conditions.values(), conditions.features()
    if not values:
        yield
        return
    aggregate = lerobot_record.aggregate_pipeline_dataset_features
    build_frame = lerobot_record.build_dataset_frame

    def with_features(*args, **kwargs):
        """原生 schema 上添加任务字段，使恢复录制也执行原生兼容性检查。"""
        return {**aggregate(*args, **kwargs), **features}

    def with_conditions(ds_features, sample, prefix):
        """每个观测周期只附加一次条件，状态与动作维度保持不变。"""
        frame = build_frame(ds_features, sample, prefix)
        return {**frame, **values} if prefix == "observation" else frame

    lerobot_record.aggregate_pipeline_dataset_features = with_features
    lerobot_record.build_dataset_frame = with_conditions
    try:
        yield
    finally:
        lerobot_record.aggregate_pipeline_dataset_features = aggregate
        lerobot_record.build_dataset_frame = build_frame


@contextmanager
def consecutive_recording(dataset: DatasetRecordConfig):
    """从原生数据配置绑定日志目录，并管理各段循环的观测配对边界。"""
    from lerobot_robot_welding.adapter import ElfinWelding

    original = lerobot_record.record_loop
    connect = ElfinWelding.connect

    def with_dataset(robot, *args, **kwargs):
        """连接时 repo_id 已由原生创建流程加上时间戳，默认目录也随之确定。"""
        robot.dataset_root = Path(dataset.root or HF_LEROBOT_HOME / dataset.repo_id)
        return connect(robot, *args, **kwargs)

    def with_pairs(robot, *args, **kwargs):
        """仅适配本项目只读 Robot，其余设备沿用原生行为。"""
        if isinstance(robot, ElfinWelding):
            with robot.observation_pairs():
                return original(robot, *args, **kwargs)
        return original(robot, *args, **kwargs)

    lerobot_record.record_loop = with_pairs
    ElfinWelding.connect = with_dataset
    try:
        yield
    finally:
        lerobot_record.record_loop = original
        ElfinWelding.connect = connect


def record_with_conditions(config: ConditionedRecordConfig):
    """调用原生 record，连接、重录、保存、恢复和清理均由 LeRobot 完成。"""
    if config.require_task_labels and config.task_conditions.direction is None:
        raise ValueError(
            "正式采集缺少 task.direction; 请明确填写 forward 或 reverse。"
            "仅联调测试可设置 --require_task_labels=false"
        )
    # Draccus 会将 preset 的数字字符串解码为 int；统一为仿真导出使用的字符串表示。
    encoder = config.dataset.rgb_encoder
    if encoder.vcodec == "libsvtav1" and encoder.preset is not None:
        encoder.preset = str(encoder.preset)
    native = RecordConfig(
        **{item.name: getattr(config, item.name) for item in fields(RecordConfig)}
    )
    with recording_conditions(config.task_conditions), consecutive_recording(config.dataset):
        return lerobot_record.record(native)
