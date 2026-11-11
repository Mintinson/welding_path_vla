"""通过华沿 CPS SDK 读取实测关节和当前工具的世界系位姿。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from welding_path_vla.core.config import SceneConfig
from welding_path_vla.core.geometry import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    rpy_degrees_to_quaternion,
    yaw_degrees_to_matrix,
)


@dataclass(frozen=True)
class CapturedState:
    """一次只读查询的状态及主机时序。

    Attributes:
        values: 13D 状态，依次为弧度关节角、米制 TCP 位置、wxyz 四元数。
        request_time: SDK 查询开始时的主机单调时钟秒数。
        receive_time: SDK 查询完成时的主机单调时钟秒数，并非控制器采样时间。
    """

    values: np.ndarray
    request_time: float
    receive_time: float


class CPSStateReader:
    """仅开放连接、读取和断开能力，不包含任何机器人运动调用。"""

    def __init__(
        self,
        client: Any,
        scene: SceneConfig,
        box_id: int = 0,
        rbt_id: int = 0,
        *,
        lock_calibration: bool = True,
    ):
        self.client = client
        self.box_id, self.rbt_id = box_id, rbt_id
        self.base_rotation = yaw_degrees_to_matrix(scene.robot_base_yaw_deg)
        self.base_position = np.asarray(scene.robot_base_position_m)
        self.lock_calibration = lock_calibration
        self.tcp: np.ndarray | None = None
        self.ucs: np.ndarray | None = None
        self.connected = False

    def check(self, code: int, operation: str) -> None:
        """保留 SDK 原始错误码，失败时停止采集。"""
        if code:
            raise RuntimeError(f"{operation} failed: CPS error {code}")

    def connect(self, host: str, port: int) -> None:
        """连接到已经由外部控制的机械臂，不改变控制模式或标定。"""
        self.tcp, self.ucs = None, None
        # SDK 连接失败也可能已经建立套接字或启动心跳，外层仍须调用 close 清理。
        self.connected = True
        self.check(self.client.HRIF_Connect(self.box_id, host, port), "HRIF_Connect")

    def read(self) -> CapturedState:
        """一次查询读取实测关节、Base 系 TCP 位姿及当前标定。

        四段返回值依次为关节角、Base 系 TCP 实测位姿、工具偏移和用户坐标系
        偏移。TCP 位姿已经包含有效工具标定，后两段仅记录和检查，不重复叠加。
        """
        started = perf_counter()
        result: list[str] = []
        self.check(self.client.HRIF_ReadActPos(self.box_id, self.rbt_id, result), "HRIF_ReadActPos")
        raw = np.asarray(result[:24], dtype=np.float64)
        if raw.shape != (24,) or not np.isfinite(raw).all():
            raise ValueError("HRIF_ReadActPos 必须返回 24 个有限数值")
        joints, base_pose, tcp, ucs = raw.reshape(4, 6)
        if not self.lock_calibration or self.tcp is None or self.ucs is None:
            self.tcp, self.ucs = tcp.copy(), ucs.copy()
        elif not (
            np.allclose(tcp, self.tcp, rtol=0, atol=1e-6)
            and np.allclose(ucs, self.ucs, rtol=0, atol=1e-6)
        ):
            raise RuntimeError("采集会话内 TCP/UCS 已改变, 请结束采集后重新启动")
        rotation = self.base_rotation @ quaternion_to_matrix(
            rpy_degrees_to_quaternion(*base_pose[3:])
        )
        position = self.base_rotation @ (base_pose[:3] / 1000) + self.base_position
        values = np.concatenate((np.deg2rad(joints), position, matrix_to_quaternion(rotation)))
        return CapturedState(values.astype(np.float32), started, perf_counter())

    def close(self) -> None:
        """仅关闭通信，不向外部控制中的机械臂发送停止命令。"""
        if self.connected:
            self.connected = False
            self.check(self.client.HRIF_DisConnect(self.box_id), "HRIF_DisConnect")
