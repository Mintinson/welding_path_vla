"""Framework-independent raw episode schema."""

import json
from pathlib import Path

import numpy as np

RAW_DATASET_FORMAT_V1 = "weldpath_raw_v1"
RAW_DATASET_FORMAT_V2 = "weldpath_raw_v2"
RAW_DATASET_FORMAT = RAW_DATASET_FORMAT_V2
RAW_VIDEO_STATS_FILE = "video_stats.json"
STATE_FILE = "trajectory.npz"
METADATA_FILE = "metadata.json"


class EpisodeReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata = json.loads((self.path / METADATA_FILE).read_text(encoding="utf-8"))
        self.trajectory = np.load(self.path / STATE_FILE, allow_pickle=False)

    @property
    def action_count(self) -> int:
        return int(self.trajectory["command_delta_pose_seam"].shape[0])

    @property
    def state_count(self) -> int:
        return int(self.trajectory["joint_position"].shape[0])


__all__ = [
    "METADATA_FILE", "RAW_DATASET_FORMAT", "RAW_DATASET_FORMAT_V1",
    "RAW_DATASET_FORMAT_V2", "RAW_VIDEO_STATS_FILE", "STATE_FILE", "EpisodeReader",
]
