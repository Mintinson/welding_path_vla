"""通过华沿 CPS SDK 下发世界系 TCP 目标。"""

from concurrent.futures import Future, TimeoutError
from queue import Empty, SimpleQueue
from threading import Event, Thread
from time import perf_counter, sleep

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from welding_path_vla.core.domain import Pose
from welding_path_vla.core.geometry import (
    absolute_ee_action_to_pose,
    quaternion_to_matrix,
    rotation_from_6d_rows,
    rotation_to_6d_rows,
)
from welding_path_vla.robot.cps_reader import CPSStateReader

SERVO_HZ = 100
SERVO_LOOKAHEAD_S = 0.05


class TCPInterpolation:
    """在同一时间轴上对位置线性插值，对姿态做球面插值。"""

    def __init__(self, actions: np.ndarray, action_hz: float) -> None:
        self.positions = actions[:, :3]
        self.times = np.arange(len(actions)) / action_hz
        self.orientation = Slerp(
            self.times, Rotation.from_matrix(rotation_from_6d_rows(actions[:, 3:]))
        )
        self.duration = float(self.times[-1])

    def sample(self, elapsed: float) -> np.ndarray:
        time = min(max(0.0, elapsed), self.duration)
        xyz = [np.interp(time, self.times, axis) for axis in self.positions.T]
        return np.r_[xyz, rotation_to_6d_rows(self.orientation(time).as_matrix())]


class CPSServo:
    """独立控制会话，复用采集端的基座外参与当前工具标定。"""

    def __init__(self, reader: CPSStateReader) -> None:
        self.reader = reader
        self.active = False
        self.last_euler: np.ndarray | None = None

    def start(self) -> None:
        """启动 ServoP，等待控制器初始化后再发首个目标。"""
        reader = self.reader
        reader.check(
            reader.client.HRIF_StartServo(
                reader.box_id, reader.rbt_id, 1 / SERVO_HZ, SERVO_LOOKAHEAD_S
            ),
            "HRIF_StartServo",
        )
        self.last_euler = None
        self.active = True
        sleep(0.025)

    def send_pose(self, target: Pose) -> None:
        """转换为示教器 Base 坐标，在 PushServoP 前检查目标 XYZ。"""
        if not self.active:
            raise RuntimeError("请先启动 ServoP 会话")
        reader = self.reader
        base_from_world = reader.base_rotation.T
        # 去掉坐标变换的浮点尾差，边界比较和下发使用同一组毫米数值。
        position_mm = (1000 * (base_from_world @ (target.position - reader.base_position))).round(9)
        rotation = base_from_world @ quaternion_to_matrix(target.quaternion_wxyz)
        euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        if self.last_euler is not None:
            # 选择离上一指令最近的等价角，避免在 ±180° 处跳变。
            euler = self.last_euler + (euler - self.last_euler + 180) % 360 - 180
        x, y, z = position_mm
        if not (
            np.isfinite(position_mm).all() and -400 <= x <= 200 and 200 <= y <= 450 and z >= -15
        ):
            self.stop()
            raise ValueError(f"ServoP 目标超出 Base TCP 边界 {position_mm.tolist()} mm")
        reader.check(
            reader.client.HRIF_PushServoP(
                reader.box_id,
                reader.rbt_id,
                np.concatenate((position_mm, euler)).tolist(),
                reader.tcp,
                [0.0] * 6,
            ),
            "HRIF_PushServoP",
        )
        self.last_euler = euler

    def stop(self) -> None:
        """仅结束本会话已启动的伺服。"""
        if self.active:
            self.active = False
            reader = self.reader
            reader.check(reader.client.HRIF_GrpStop(reader.box_id, reader.rbt_id), "HRIF_GrpStop")


class RolloutServo:
    """测试和官方 rollout 共用的单点目标插值与 ServoP 下发线程。"""

    def __init__(self, servo: CPSServo, action_hz: float) -> None:
        self.servo = servo
        self.action_hz = action_hz
        self.targets: SimpleQueue[tuple[np.ndarray, Future[None]]] = SimpleQueue()
        self.completion: Future[None] | None = None
        self.error: Exception | None = None
        self.ending = Event()
        self.thread = Thread(target=self.run, name="elfin-rollout-servo")

    def submit(self, target: np.ndarray) -> None:
        """首个动作到达后才启动伺服，按接收顺序播放目标。"""
        self.check_error()
        self.completion = Future()
        self.targets.put((target, self.completion))
        if self.thread.ident is None:
            self.thread.start()

    def check_error(self) -> None:
        if self.error is not None:
            raise self.error

    def wait(self) -> None:
        """等待最后一个已提交目标完成，并传播线程下发错误。"""
        completion = self.completion
        while completion is not None:
            self.check_error()
            try:
                completion.result(timeout=0.05)
                return
            except TimeoutError:
                continue

    def run(self) -> None:
        """从当前实测 TCP 开始插值，推理期间保持最后一个目标。"""
        try:
            values = self.servo.reader.read().values
            last = np.r_[values[6:9], rotation_to_6d_rows(quaternion_to_matrix(values[9:13]))]
            if self.ending.is_set():
                return
            self.servo.start()
            segment = None
            completion = None
            started = perf_counter()
            period = 1 / SERVO_HZ
            while not self.ending.is_set():
                tick = perf_counter()
                finished = None
                if segment is None or tick - started >= segment.duration:
                    if segment is not None:
                        last = segment.sample(segment.duration)
                        finished = completion
                        # 保留正常周期的余量，超时后从当前周期继续。
                        started = max(started + segment.duration, tick - period)
                    else:
                        started = tick
                    try:
                        target, completion = self.targets.get_nowait()
                    except Empty:
                        segment = None
                    else:
                        segment = TCPInterpolation(np.vstack((last, target)), self.action_hz)
                if segment is not None:
                    last = segment.sample(tick - started)
                self.servo.send_pose(absolute_ee_action_to_pose(last))
                if finished is not None:
                    finished.set_result(None)
                self.ending.wait(max(0.0, tick + period - perf_counter()))
        except Exception as error:
            self.error = error
        finally:
            try:
                self.servo.stop()
            except Exception as error:
                self.error = self.error or error
            if self.completion is not None and not self.completion.done():
                self.completion.set_exception(self.error or RuntimeError("ServoP 会话已结束"))

    def close(self) -> None:
        """停止下发线程，确保它退出后才断开 CPS。"""
        self.ending.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.check_error()
