"""仅用于采集链路验收的全黑 RGB 占位相机。"""

from dataclasses import dataclass

import numpy as np
from lerobot.cameras import Camera, CameraConfig, ColorMode


@CameraConfig.register_subclass("black")
@dataclass
class BlackCameraConfig(CameraConfig):
    """显式选择 black 才启用；不能代表真实视觉观测。"""

    width: int | None = 640
    height: int | None = 480
    fps: int | None = 30
    color_mode: ColorMode = ColorMode.RGB


class BlackCamera(Camera):
    """按原生相机接口返回固定全黑帧，不访问任何物理相机。"""

    def __init__(self, config: BlackCameraConfig):
        super().__init__(config)
        if config.height is None or config.width is None:
            raise ValueError("虚拟相机需要明确的 width 和 height")
        self.connected = False
        self.frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)

    @property
    def is_connected(self) -> bool:
        return self.connected

    @staticmethod
    def find_cameras() -> list[dict]:
        """虚拟相机由配置创建，不枚举物理设备。"""
        return []

    def connect(self, warmup: bool = True) -> None:
        """标记可读，无需硬件预热。"""
        self.connected = True

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        """返回全黑 RGB 帧；占位图像不存在曝光时间或陈旧帧判断。"""
        if not self.connected:
            raise RuntimeError("虚拟相机尚未连接")
        return self.frame

    def read(self) -> np.ndarray:
        """同步读取同一占位图像。"""
        return self.read_latest()

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """立即返回占位图像；采样周期由录制循环控制。"""
        return self.read_latest()

    def disconnect(self) -> None:
        """结束虚拟相机会话。"""
        self.connected = False
