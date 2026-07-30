"""基于 LeRobot / PyAV 的统一视频录制接口。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from av.logging import ERROR, set_level
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.video_utils import StreamingVideoEncoder


@dataclass(slots=True)
class VideoRecorder:
    """把多相机 RGB 帧流式编码为兼容浏览器的 H.264 MP4。"""

    root: Path
    names: tuple[str, ...]
    encoder: StreamingVideoEncoder

    @classmethod
    def start(cls, root: Path, names: tuple[str, ...], fps: int) -> VideoRecorder:
        """启动一个多相机录制会话。

        Args:
            root: 最终视频所在目录。
            names: 相机名称，同时作为 MP4 文件名。
            fps: 图像采样与视频播放帧率。

        Returns:
            已启动的录制器。
        """

        set_level(ERROR)
        video_config = RGBEncoderConfig(vcodec="h264", crf=23, preset="veryfast")
        encoder = StreamingVideoEncoder(fps, rgb_encoder=video_config, queue_maxsize=0)
        encoder.start_episode(list(names), root)
        return cls(root, names, encoder)

    def append(self, images: dict[str, np.ndarray]) -> None:
        """写入同一采样时刻的所有相机帧。

        Args:
            images: 相机名称到 RGB ``uint8`` 图像的映射。
        """
        for name in self.names:
            self.encoder.feed_frame(name, images[name])

    def finish(self) -> tuple[str, ...]:
        """完成编码，并把 LeRobot 临时文件移动到 episode 根目录。

        Returns:
            按 ``names`` 顺序排列的最终视频路径。
        """
        encoded = self.encoder.finish_episode()
        videos: list[str] = []
        for name in self.names:
            source, _ = encoded[name]
            destination = self.root / f"{name}.mp4"
            shutil.move(source, destination)
            shutil.rmtree(source.parent, ignore_errors=True)
            videos.append(str(destination))
        self.encoder.close()
        return tuple(videos)

    def close(self) -> None:
        """关闭编码器；未完成的会话会由 LeRobot 清理。"""
        self.encoder.close()
