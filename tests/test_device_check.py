"""用模拟时钟和设备验证轮询频率、独立开关及退出清理。"""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


class UpdatingReader:
    """每次查询返回不同状态，供测试确认没有复用旧观测。"""

    def __init__(self):
        self.connected = False
        self.reads = 0

    def connect(self, host, port):
        self.address = (host, port)
        self.connected = True

    def read(self):
        self.reads += 1
        return SimpleNamespace(values=np.full(13, self.reads, dtype=float))

    def close(self):
        self.connected = False


class UpdatingCamera:
    """每次读取改变红色通道，可同时检查画面变化和预览颜色。"""

    def __init__(self):
        self.is_connected = False
        self.reads = 0

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def async_read(self):
        self.reads += 1
        return np.full((2, 3, 3), [self.reads, 30, 10], dtype=np.uint8)


@pytest.fixture
def devices(monkeypatch):
    """替换物理设备与等待函数，检查仍走真实配置解析和输出逻辑。"""
    monkeypatch.syspath_prepend("scripts")
    command = import_module("check_real_devices")
    reader = UpdatingReader()
    cameras = {name: UpdatingCamera() for name in ("global", "wrist")}
    now = [0.0]
    reader_calls, camera_calls = [], []

    def make_reader(settings):
        reader_calls.append(settings)
        return reader

    def make_cameras(configs):
        camera_calls.append(configs)
        return cameras

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(command, "make_reader", make_reader)
    monkeypatch.setattr(command, "make_cameras_from_configs", make_cameras)
    monkeypatch.setattr(command, "perf_counter", lambda: now[0])
    monkeypatch.setattr(command, "sleep", advance)
    return SimpleNamespace(
        command=command,
        reader=reader,
        cameras=cameras,
        now=now,
        reader_calls=reader_calls,
        camera_calls=camera_calls,
    )


@pytest.mark.parametrize(
    "robot,cameras", [(True, True), (True, False), (False, True), (False, False)]
)
def test_independent_switches_and_live_polling(devices, capsys, robot, cameras):
    """默认五秒应读取十次；关闭的设备不能被创建或连接。"""
    command = devices.command
    command.check_devices(
        command.DeviceCheckArguments(
            robot=robot,
            cameras=cameras,
            preview=False,
            host="127.0.0.1",
        )
    )
    output = capsys.readouterr().out
    assert len(devices.reader_calls) == int(robot)
    assert len(devices.camera_calls) == int(cameras)
    assert devices.reader.reads == (10 if robot else 0)
    assert devices.now[0] == (5 if robot or cameras else 0)
    assert not devices.reader.connected
    for camera in devices.cameras.values():
        assert camera.reads == (10 if cameras else 0)
        assert not camera.is_connected
    if robot:
        assert devices.reader.address == ("127.0.0.1", 10003)
        assert "joints(rad)=[1.0" in output and "joints(rad)=[10.0" in output
    if cameras:
        assert "mean_RGB=[1.0, 30.0, 10.0]" in output
        assert "mean_RGB=[10.0, 30.0, 10.0]" in output
        settings = yaml.safe_load(command.DeviceCheckArguments().record_config.read_text())
        for name, config in devices.camera_calls[0].items():
            expected = settings["robot"]["cameras"][name]
            assert config.index_or_path == Path(expected["index_or_path"])
            assert config.backend == expected["backend"]
            assert config.fourcc == expected["fourcc"]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_read_failure_or_interrupt_closes_all_devices(devices, monkeypatch, failure):
    """查询失败和人工中断都必须释放已经打开的机器人与相机。"""

    def fail():
        raise failure("模拟读取失败")

    monkeypatch.setattr(devices.cameras["wrist"], "async_read", fail)
    with pytest.raises(failure):
        devices.command.check_devices(devices.command.DeviceCheckArguments(preview=False))
    assert not devices.reader.connected
    assert all(not camera.is_connected for camera in devices.cameras.values())


def test_partial_connection_failure_closes_devices(devices, monkeypatch):
    """第二路相机预热失败也要清理其连接及此前已连接的设备。"""

    def fail():
        devices.cameras["wrist"].is_connected = True
        raise RuntimeError("模拟相机预热失败")

    monkeypatch.setattr(devices.cameras["wrist"], "connect", fail)
    with pytest.raises(RuntimeError, match="预热失败"):
        devices.command.check_devices(devices.command.DeviceCheckArguments(preview=False))
    assert not devices.reader.connected
    assert all(not camera.is_connected for camera in devices.cameras.values())


def test_preview_color_and_window_cleanup(devices, monkeypatch):
    """默认 RGB 帧应转成 OpenCV 的 BGR，Esc 后关闭设备和窗口。"""
    displayed, closed = {}, []
    cv2 = devices.command.cv2
    monkeypatch.setattr(cv2, "imshow", lambda name, image: displayed.update({name: image}))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: closed.append(True))
    devices.command.check_devices(devices.command.DeviceCheckArguments(robot=False))
    assert set(displayed) == {"global", "wrist"}
    np.testing.assert_array_equal(displayed["global"][0, 0], [10, 30, 1])
    assert closed == [True]
    assert all(not camera.is_connected for camera in devices.cameras.values())
