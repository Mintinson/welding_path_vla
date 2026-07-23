"""Framework-independent raw episode schema."""

import json
from pathlib import Path

import numpy as np

RAW_DATASET_FORMAT = "weldpath_raw_v1"
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


__all__ = ["METADATA_FILE", "RAW_DATASET_FORMAT", "STATE_FILE", "EpisodeReader"]
