"""Elfin5-Pro 的 LeRobot 被动采集和实测动作标签。"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep, time
from typing import Any, TextIO
from uuid import uuid4

import numpy as np
from cps_client import CPSClient
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from welding_path_vla.core.config import AppConfig
from welding_path_vla.core.geometry import (
    quaternion_to_matrix,
    rotation_to_6d_rows,
)
from welding_path_vla.dataset.actions import ABSOLUTE_ACTION_NAMES
from welding_path_vla.dataset.export_lerobot import MANIFEST_PATH, OBSERVATION_STATE_NAMES
from welding_path_vla.dataset.real_capture import CAPTURE_CONTRACT
from welding_path_vla.robot.cps_reader import CapturedState, CPSStateReader


# 原生命令在同一进程依次调用 Robot 和 Teleoperator，用 ID 共享只读采集对象。
CAPTURE_SESSIONS: dict[str, ElfinWelding] = {}


@RobotConfig.register_subclass("elfin_welding")
@dataclass
class ElfinWeldingConfig(RobotConfig):
    """被动采集配置。

    Attributes:
        scene_config: 复用项目 YAML 中的机器人基座到 world 安装外参。
        fps: 首对观测的目标采样频率，必须等于 dataset.fps；后续节拍由原生循环控制。
        bridge_id: 本进程内与 ElfinPassive 对应的唯一会话 ID。
    """

    scene_config: str = "configs/base.yaml"
    host: str = "192.168.56.101"
    port: int = 10003
    box_id: int = 0
    rbt_id: int = 0
    fps: int = 30
    bridge_id: str = "elfin"
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "global": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
            "wrist": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480),
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fps <= 0 or set(self.cameras) != {"global", "wrist"}:
            raise ValueError("需要正数 fps 和 global/wrist 两路相机")


@TeleoperatorConfig.register_subclass("elfin_passive")
@dataclass
class ElfinPassiveConfig(TeleoperatorConfig):
    """仅提供实测动作标签，不控制机械臂或启用拖动模式。"""

    bridge_id: str = "elfin"


@dataclass
class CapturedObservation:
    """一次采样的观测快照。

    Attributes:
        state: 本次 SDK 查询返回的状态及主机时间。
        values: 13D 状态字段与双相机独立图像快照。
        camera_read_times: 主线程各路取帧完成时间，并非曝光时间。
    """

    state: CapturedState
    values: dict[str, Any]
    camera_read_times: dict[str, float]


class ElfinWelding(Robot):
    """连续观测配对成实测后继动作；每个原生循环独立处理首尾。"""

    config_class = ElfinWeldingConfig
    name = "elfin_welding"

    def __init__(self, config: ElfinWeldingConfig):
        super().__init__(config)
        self.config = config
        app = AppConfig.load(config.scene_config)
        self.scene, self.task = app.scene, app.task
        self.reader = CPSStateReader(
            CPSClient(),
            self.scene,
            config.box_id,
            config.rbt_id,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self.connected = False
        self.source: CapturedObservation | None = None
        self.target: CapturedObservation | None = None
        self.log: TextIO | None = None
        # 录制时绑定原生数据目录；None 表示只读预览，不读写数据集文件。
        self.dataset_root: Path | None = None

    @property
    def observation_features(self) -> dict:
        """与仿真导出保持相同的 13D 状态顺序和双相机命名。"""
        return {
            **{name: float for name in OBSERVATION_STATE_NAMES},
            **{name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()},
        }

    @property
    def action_features(self) -> dict:
        """保存世界系绝对 TCP 位置和 rotation-6D。"""
        return {name: float for name in ABSOLUTE_ACTION_NAMES}

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return self.reader.tcp is not None

    def calibrate(self) -> None:
        """工具标定由控制器维护，采集端不修改。"""

    def configure(self) -> None:
        """不改变示教器、Servo、override 或其他控制模式。"""

    def connect(self, calibrate: bool = True) -> None:
        """连接机器人与双相机，不改变控制模式。"""
        if self.connected or self.config.bridge_id in CAPTURE_SESSIONS:
            raise RuntimeError("采集会话已连接或 bridge_id 已被使用")
        root = self.dataset_root
        if root is not None:
            info = json.loads((root / "meta/info.json").read_text())
            if info["fps"] != self.config.fps or info["robot_type"] != self.name:
                raise ValueError("dataset.root 中的机器人类型必须一致, 且 robot.fps = dataset.fps")
        try:
            self.reader.connect(self.config.host, self.config.port)
            self.reader.read()
            for camera in self.cameras.values():
                camera.connect()
            if root is not None:
                self.start_session_log(root)
            self.connected = True
            CAPTURE_SESSIONS[self.config.bridge_id] = self
        except BaseException:
            self.disconnect()
            raise

    def start_session_log(self, root: Path) -> None:
        """录制连接后保存有效标定及契约；预览连接不调用此方法。"""
        directory = root / "meta/real_sessions"
        directory.mkdir(exist_ok=True)
        self.log = (directory / f"{uuid4().hex}.jsonl").open("x")
        assert self.reader.tcp is not None and self.reader.ucs is not None
        self.write_log(
            {
                "kind": "session",
                "wall_time_unix_s": time(),
                "clock": "host_perf_counter_seconds_not_exposure_time",
                "tcp_mm_deg": self.reader.tcp.tolist(),
                "ucs_mm_deg": self.reader.ucs.tolist(),
                "world_from_base_position_m": self.scene.robot_base_position_m,
                "world_from_base_yaw_deg": self.scene.robot_base_yaw_deg,
                "pose_source": "read_act_pos_base_tcp",
                "sampling_mode": "consecutive_observations",
            }
        )
        contract = {
            "format_version": 1,
            "fps": self.config.fps,
            "action_source": "next_measured_tcp",
            "state_names": list(OBSERVATION_STATE_NAMES),
            "action_names": list(ABSOLUTE_ACTION_NAMES),
        }
        (root / CAPTURE_CONTRACT).write_text(json.dumps(contract, indent=2))
        # 恢复录制会改变样本与统计，必须重新整理后才可用于训练。
        (root / MANIFEST_PATH).unlink(missing_ok=True)

    def write_log(self, record: dict[str, Any]) -> None:
        """记录原始采集时序；包含未保存的重置或重录周期。"""
        if self.log is not None:
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()

    def read_observation(self) -> CapturedObservation:
        """一次 SDK 查询配合两路最新图像；复制图像以保留上一周期快照。"""
        state = self.reader.read()
        observation: dict[str, Any] = dict(zip(OBSERVATION_STATE_NAMES, state.values, strict=True))
        camera_read_times = {}
        for name, camera in self.cameras.items():
            observation[name] = camera.read_latest().copy()
            # 此时间只是主线程取帧完成时间，不冒充曝光或后台接收时间。
            camera_read_times[name] = perf_counter()
        return CapturedObservation(state, observation, camera_read_times)

    @contextmanager
    def observation_pairs(self):
        """每次原生录制/重置循环独立预采样，退出时丢弃未配对的末观测。

        首个观测在循环开始前准备，等待一次以保证首帧也有真实后继。
        后续节拍完全由原生 record_loop 控制，不在每帧额外等待或重复查询。
        """
        if not self.connected:
            raise RuntimeError("机械臂尚未连接")
        try:
            self.target = self.read_observation()
            sleep(max(0.0, self.target.state.request_time + 1 / self.config.fps - perf_counter()))
            yield
        finally:
            self.source = self.target = None

    def get_observation(self) -> dict:
        """按原生循环返回已配对的上一帧观测。"""
        if self.target is None:
            raise RuntimeError("请通过 record-real 入口启动连续观测配对")
        self.source, self.target = self.target, self.read_observation()
        return self.source.values

    def measured_action(self) -> dict[str, float]:
        """复用本周期已读取的后继状态，首尾样本均不补零或重复查询。"""
        source, target = self.source, self.target
        if source is None or target is None:
            raise RuntimeError("必须先调用 get_observation 再请求实测动作")
        values = target.state.values
        action = [*values[6:9], *rotation_to_6d_rows(quaternion_to_matrix(values[9:13]))]
        self.write_log(
            {
                "kind": "sample",
                "source_request_s": source.state.request_time,
                "source_receive_s": source.state.receive_time,
                "camera_read_s": source.camera_read_times,
                "target_request_s": target.state.request_time,
                "target_receive_s": target.state.receive_time,
                "state": source.state.values.tolist(),
                "action": [float(value) for value in action],
            }
        )
        self.source = None
        return dict(zip(ABSOLUTE_ACTION_NAMES, map(float, action), strict=True))

    def send_action(self, action: dict) -> dict:
        """原样返回实测动作标签，不向机械臂发送运动命令。"""
        return action

    def disconnect(self) -> None:
        """关闭本会话的日志、相机和通信。"""
        if CAPTURE_SESSIONS.get(self.config.bridge_id) is self:
            CAPTURE_SESSIONS.pop(self.config.bridge_id)
        self.connected, self.source, self.target = False, None, None
        with ExitStack() as cleanup:
            if self.log is not None:
                cleanup.callback(self.log.close)
                self.log = None
            cleanup.callback(self.reader.close)
            for camera in self.cameras.values():
                if camera.is_connected:
                    cleanup.callback(camera.disconnect)


class ElfinPassive(Teleoperator):
    """将本周期实测位姿作为上一周期观测的后继动作，不再等待或查询。"""

    config_class = ElfinPassiveConfig
    name = "elfin_passive"

    def __init__(self, config: ElfinPassiveConfig):
        super().__init__(config)
        self.config = config
        self.connected = False

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(ABSOLUTE_ACTION_NAMES, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.config.bridge_id not in CAPTURE_SESSIONS:
            raise RuntimeError("未找到相同 bridge_id 的只读 Robot")
        self.connected = True

    def calibrate(self) -> None:
        """没有遥操作硬件需要标定。"""

    def configure(self) -> None:
        """不配置或控制机械臂。"""

    def get_action(self) -> dict[str, float]:
        if not self.connected:
            raise RuntimeError("标签提供器尚未连接")
        return CAPTURE_SESSIONS[self.config.bridge_id].measured_action()

    def send_feedback(self, feedback: dict) -> None:
        """被动采集不向外部设备反馈控制量。"""

    def disconnect(self) -> None:
        self.connected = False
