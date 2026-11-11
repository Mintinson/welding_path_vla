#!/usr/bin/env python3
"""从录制配置启动只读预览，复用官方 lerobot-teleoperate 循环。"""

from contextlib import ExitStack
from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from common import cli
from draccus.parsers.decoding import decode
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_teleoperate import (
    TeleoperateConfig,
    init_visualization,
    teleop_loop,
)
from lerobot.utils.visualization_utils import shutdown_visualization
from lerobot_robot_welding import (
    ElfinPassive,
    ElfinPassiveConfig,
    ElfinWelding,
    ElfinWeldingConfig,
)


@dataclass
class PreviewArguments:
    """只读预览选项。

    Attributes:
        record_config: 现有录制 YAML；复用设备、FPS 和可视化后端配置。
        duration_s: 预览秒数；省略时持续显示，直到按 Ctrl+C。
    """

    record_config: Path = Path("configs/real_record.yaml")
    duration_s: float | None = None


def preview_devices(arguments: PreviewArguments) -> None:
    """连接只读设备并运行官方循环，正常结束或异常时均清理资源。"""
    if arguments.duration_s is not None and arguments.duration_s <= 0:
        raise ValueError("duration_s 必须大于 0")
    settings = yaml.safe_load(arguments.record_config.read_text(encoding="utf-8"))
    options = {
        item.name: settings[item.name]
        for item in fields(TeleoperateConfig)
        if item.name in settings
    }
    options.update(
        fps=settings["dataset"]["fps"], display_data=True, teleop_time_s=arguments.duration_s
    )
    config = decode(TeleoperateConfig, options)
    if not isinstance(config.robot, ElfinWeldingConfig) or not isinstance(
        config.teleop, ElfinPassiveConfig
    ):
        raise ValueError("只读预览仅支持 elfin_welding + elfin_passive")
    robot, teleop = ElfinWelding(config.robot), ElfinPassive(config.teleop)
    with ExitStack() as resources:
        resources.callback(robot.disconnect)
        robot.connect()
        resources.callback(teleop.disconnect)
        teleop.connect()
        resources.callback(shutdown_visualization, config.display_mode)
        init_visualization(
            config.display_mode,
            session_name="real_preview",
            ip=config.display_ip,
            port=config.display_port,
        )
        with robot.observation_pairs():
            teleop_loop(
                teleop,
                robot,
                config.fps,
                *make_default_processors(),
                display_data=True,
                display_mode=config.display_mode,
                duration=config.teleop_time_s,
                display_compressed_images=config.display_compressed_images,
            )


@cli
def main(config: PreviewArguments) -> None:
    """运行预览；Ctrl+C 只退出显示和断开通信，不向机械臂发送停止命令。"""
    try:
        preview_devices(config)
    except KeyboardInterrupt:
        print("预览结束, 已断开设备。", flush=True)


if __name__ == "__main__":
    main()
