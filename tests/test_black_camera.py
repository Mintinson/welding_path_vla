"""检查全黑占位相机可通过原生工厂创建、读取和关闭。"""

import draccus
import numpy as np
import pytest
from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot_robot_welding import BlackCamera


def test_native_black_camera_factory():
    """相机注册与工厂使用真实路径，不用 monkeypatch 替换物理设备。"""
    config = draccus.decode(CameraConfig, {"type": "black", "width": 32, "height": 24})
    cameras = make_cameras_from_configs({"global": config, "wrist": config})
    for camera in cameras.values():
        assert isinstance(camera, BlackCamera)
        with pytest.raises(RuntimeError, match="尚未连接"):
            camera.read_latest()
        camera.connect()
        frame = camera.read_latest()
        assert frame.shape == (24, 32, 3) and frame.dtype == np.uint8
        assert not np.any(frame)
        np.testing.assert_array_equal(camera.async_read(), frame)
        camera.disconnect()
        assert not camera.is_connected
