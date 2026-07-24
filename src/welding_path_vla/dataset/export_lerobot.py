"""导出为 LeRobot 格式: 将原始录制数据转换为 lerobot 数据集格式。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from welding_path_vla.core.domain import EpisodeStatus
from welding_path_vla.dataset.actions import build_relative_action_chunk
from welding_path_vla.dataset.raw_schema import EpisodeReader


def video_frames(path: Path) -> list[np.ndarray]:
    """从 MP4 文件读取所有帧, 转为 RGB 顺序。

    Args:
        path: MP4 文件路径。

    Returns:
        RGB 图像数组列表, 每帧 [H, W, 3]。
    """
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def export_lerobot(dataset_root: str | Path, output: str | Path, repo_id: str) -> Path:
    """将原始焊接数据集导出为 LeRobot 格式。

    只导出 VALID_SUCCESS 和 VALID_RECOVERY 状态的 episode,
    按帧组织 observation (图像 + 状态) 和 action, 最终调用
    dataset.finalize() 完成写入。

    Args:
        dataset_root: 原始数据集根目录。
        output: 输出路径。
        repo_id: LeRobot 数据集标识符 (如 "huayan/weldpath_sim_v1")。

    Returns:
        输出目录路径。

    Raises:
        RuntimeError: 缺少 lerobot 依赖包。
        ValueError: 没有符合条件的有效 episode。
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("run this command in the Pixi data or dev environment") from error

    root = Path(dataset_root)
    destination = Path(output)
    episodes = [EpisodeReader(path) for path in sorted((root / "episodes").glob("episode_*"))]
    episodes = [
        episode
        for episode in episodes
        if episode.metadata["quality"]["status"]
        in {EpisodeStatus.VALID_SUCCESS.value, EpisodeStatus.VALID_RECOVERY.value}
    ]
    if not episodes:
        raise ValueError(f"no valid episodes found in {root}")
    camera = episodes[0].metadata["resolved_config"]["camera"]
    fps = episodes[0].metadata["resolved_config"]["timing"]["policy_hz"]
    features = {
        "observation.images.global": {
            "dtype": "video",
            "shape": (camera["height"], camera["width"], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (camera["height"], camera["width"], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (13,),
            "names": [f"state_{index}" for index in range(13)],
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["dx", "dy", "dz", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z"],
        },
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=destination, features=features)
    for episode in episodes:
        episode_path = episode.path
        global_frames = video_frames(episode_path / "global.mp4")
        wrist_frames = video_frames(episode_path / "wrist.mp4")
        trajectory = episode.trajectory
        action_source = episode.metadata["resolved_config"]["policy"]["action_source"]
        for index in range(episode.action_count):
            state = np.concatenate(
                [
                    trajectory["joint_position"][index],
                    trajectory["tcp_position"][index],
                    trajectory["tcp_quaternion_wxyz"][index],
                ]
            ).astype(np.float32)
            dataset.add_frame(
                {
                    "observation.images.global": global_frames[index],
                    "observation.images.wrist": wrist_frames[index],
                    "observation.state": state,
                    "action": build_relative_action_chunk(
                        episode, index, horizon=1, source=action_source
                    ).values[0],
                    "task": episode.metadata["instruction"],
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return destination
