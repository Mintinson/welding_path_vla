#!/usr/bin/env python3
"""同步回放一条 episode 的全局与腕部相机视频。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
from common import cli

from welding_path_vla.dataset.raw_schema import EpisodeReader


@dataclass
class ReplayArguments:
    """原始 episode 回放参数。"""

    episode: Path


@cli
def main(config: ReplayArguments) -> None:
    """按 episode 的策略频率播放双视角视频。"""
    episode = EpisodeReader(config.episode)
    global_video = cv2.VideoCapture(str(episode.path / "global.mp4"))
    wrist_video = cv2.VideoCapture(str(episode.path / "wrist.mp4"))
    delay = max(1, round(1000 / episode.metadata["resolved_config"]["timing"]["policy_hz"]))
    while global_video.isOpened() and wrist_video.isOpened():
        global_ok, global_frame = global_video.read()
        wrist_ok, wrist_frame = wrist_video.read()
        if not global_ok or not wrist_ok:
            break
        cv2.imshow("global", global_frame)
        cv2.imshow("wrist", wrist_frame)
        if cv2.waitKey(delay) == 27:
            break
    global_video.release()
    wrist_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
