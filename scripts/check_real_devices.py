#!/usr/bin/env python3
"""短时检查机器人只读连接与 USB 相机画面，不创建采集数据。"""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, cast

import cv2
import numpy as np
import yaml
from common import cli
from draccus.parsers.decoding import decode
from lerobot.cameras import Camera, CameraConfig, ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.import_utils import register_third_party_plugins

from welding_path_vla.core.config import AppConfig
from welding_path_vla.robot.cps_reader import CPSStateReader


@dataclass
class DeviceCheckArguments:
    """设备检查选项。

    Attributes:
        record_config: 复用录制配置中的 IP、端口、安装外参与相机参数。
        robot: 是否连接并读取机器人，与 cameras 开关相互独立。
        cameras: 是否连接并读取配置中的全部相机。
        duration_s: 所选设备连接完成后的观测时长，单位为秒。
        interval_s: 两次轮询开始时间的目标间隔，单位为秒。
        preview: 是否显示相机画面；无图形界面时设为 false。
        host: 临时覆盖机器人 IP；省略时使用 record_config。
    """

    record_config: Path = Path("configs/real_record.yaml")
    robot: bool = True
    cameras: bool = True
    duration_s: float = 5.0
    interval_s: float = 0.5
    preview: bool = True
    host: str | None = None


def make_reader(settings: dict[str, Any]) -> CPSStateReader:
    """复用采集端 SDK 与坐标转换；仅启用机器人时才加载 SDK。"""
    from cps_client import CPSClient

    return CPSStateReader(
        CPSClient(),
        AppConfig.load(settings["scene_config"]).scene,
        settings.get("box_id", 0),
        settings.get("rbt_id", 0),
    )


def disconnect_camera(camera: Camera) -> None:
    """清理已打开的相机，包括连接期间配置或预热失败的情况。"""
    if camera.is_connected:
        camera.disconnect()


def check_devices(config: DeviceCheckArguments) -> None:
    """按目标周期打印最新状态，并在正常结束、异常或中断时释放设备。"""
    if config.duration_s <= 0 or config.interval_s <= 0:
        raise ValueError("duration_s 和 interval_s 必须大于 0")
    if not config.robot and not config.cameras:
        print("机器人和相机均已关闭, 无需连接。", flush=True)
        return
    settings = yaml.safe_load(config.record_config.read_text(encoding="utf-8"))["robot"]
    reader = None
    cameras = {}
    camera_configs: dict[str, CameraConfig] = {}
    with ExitStack() as resources:
        if config.robot:
            reader = make_reader(settings)
            resources.callback(reader.close)
            host = config.host or settings["host"]
            print(f"连接机器人 {host}:{settings['port']} ...", flush=True)
            reader.connect(host, settings["port"])
            print("机器人连接成功。", flush=True)
        if config.cameras:
            register_third_party_plugins()
            camera_configs = {
                name: decode(CameraConfig, values) for name, values in settings["cameras"].items()
            }
            cameras = make_cameras_from_configs(camera_configs)
            for name, camera in cameras.items():
                resources.callback(disconnect_camera, camera)
                print(f"连接相机 {name} ...", flush=True)
                camera.connect()
                print(f"相机 {name} 连接成功。", flush=True)
        if config.preview and cameras:
            resources.callback(cv2.destroyAllWindows)

        started = next_sample = perf_counter()
        deadline = started + config.duration_s
        while perf_counter() < deadline:
            prefix = f"[{perf_counter() - started:4.1f}s]"
            if reader is not None:
                values = reader.read().values.astype(float).round(4)
                print(
                    f"{prefix} robot joints(rad)={values[:6].tolist()} "
                    f"tcp_world(m)={values[6:9].tolist()} "
                    f"quaternion(wxyz)={values[9:13].tolist()}",
                    flush=True,
                )
            for name, camera in cameras.items():
                frame = camera.async_read()
                # 平均通道值仅用于观察画面变化，通道顺序沿用相机配置（默认 RGB）。
                color_mode = cast(OpenCVCameraConfig, camera_configs[name]).color_mode
                mean_channels = np.round(frame.mean(axis=(0, 1)), 1).tolist()
                print(
                    f"{prefix} camera={name} shape={frame.shape} "
                    f"mean_{color_mode.value.upper()}={mean_channels}",
                    flush=True,
                )
                if config.preview:
                    display = (
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if color_mode == ColorMode.RGB
                        else frame
                    )
                    cv2.imshow(name, display)
            if config.preview and cameras and cv2.waitKey(1) == 27:
                break
            # 按轮询开始时间计时，避免把查询耗时累加到 0.5 秒间隔中。
            next_sample += config.interval_s
            sleep(max(0.0, min(next_sample, deadline) - perf_counter()))
    print("检查结束, 已断开设备。", flush=True)


@cli
def main(config: DeviceCheckArguments) -> None:
    """运行设备检查；Ctrl+C 或预览窗口中的 Esc 可提前结束。"""
    try:
        check_devices(config)
    except KeyboardInterrupt:
        print("检查中断, 已断开设备。", flush=True)


if __name__ == "__main__":
    main()
