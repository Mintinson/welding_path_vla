"""基于 LeRobot / PyAV 的异步多线程视频录制接口。"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from av import logging as av_logging
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.video_utils import StreamingVideoEncoder


def video_codec_settings() -> dict[str, int | str]:
    """返回 rollout 使用的 H.264 编码参数。"""
    return {
        "vcodec": "h264",
        "pix_fmt": "yuv420p",
        "g": 2,
        "bf": 0,
        "crf": 23,
        "preset": "veryfast",
        "threads": 2,
    }


@dataclass(slots=True)
class VideoRecorder:
    """使用每相机独立线程流式编码 RGB 视频。"""

    root: Path
    names: tuple[str, ...]
    encoder: StreamingVideoEncoder
    restore_logging_callback: Callable[[], None] | None

    @classmethod
    def start(cls, root: Path, names: tuple[str, ...], fps: int) -> VideoRecorder:
        """启动多相机异步录制会话。

        Args:
            root: 最终视频所在目录。
            names: 相机名称，同时作为 MP4 文件名。
            fps: 图像采样与视频播放帧率。

        Returns:
            已启动的录制器。
        """
        root.mkdir(parents=True, exist_ok=True)
        restore_callback = av_logging.restore_default_callback

        def keep_python_callback() -> None:
            """避免编码线程恢复原生 stderr 回调。"""

        av_logging.set_level(av_logging.ERROR)
        av_logging.restore_default_callback = keep_python_callback
        settings = video_codec_settings()
        video_config = RGBEncoderConfig(
            vcodec=str(settings["vcodec"]),
            pix_fmt=str(settings["pix_fmt"]),
            g=int(settings["g"]),
            crf=int(settings["crf"]),
            preset=str(settings["preset"]),
            extra_options={"bf": int(settings["bf"])},
        )
        encoder = StreamingVideoEncoder(
            fps,
            rgb_encoder=video_config,
            queue_maxsize=0,
            encoder_threads=int(settings["threads"]),
        )
        try:
            encoder.start_episode(list(names), root)
        except BaseException:
            av_logging.restore_default_callback = restore_callback
            restore_callback()
            raise
        return cls(root, names, encoder, restore_callback)

    def append(self, images: dict[str, np.ndarray]) -> None:
        """把同一采样时刻的所有相机帧送入后台编码线程。"""
        for name in self.names:
            self.encoder.feed_frame(name, images[name])

    def finish(self) -> tuple[str, ...]:
        """等待后台编码完成，并把视频移动到 episode 目录。"""
        try:
            encoded = self.encoder.finish_episode()
            videos: list[str] = []
            for name in self.names:
                source, _ = encoded[name]
                destination = self.root / f"{name}.mp4"
                shutil.move(source, destination)
                shutil.rmtree(source.parent, ignore_errors=True)
                videos.append(str(destination))
            return tuple(videos)
        finally:
            self.encoder.close()
            self.restore_logging()

    def close(self) -> None:
        """关闭编码器并清理未完成的会话。"""
        try:
            self.encoder.close()
        finally:
            self.restore_logging()

    def restore_logging(self) -> None:
        """恢复创建录制器前的 PyAV 日志回调。"""
        if self.restore_logging_callback is None:
            return
        callback = self.restore_logging_callback
        self.restore_logging_callback = None
        av_logging.restore_default_callback = callback
        callback()


__all__ = ["VideoRecorder", "video_codec_settings"]
