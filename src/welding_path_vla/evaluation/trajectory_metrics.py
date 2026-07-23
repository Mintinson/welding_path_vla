from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from welding_path_vla.config import AppConfig, QualityConfig
from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.domain import EpisodeStatus


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    status: EpisodeStatus
    valid: bool
    seam_progress: float
    cross_track_mean_m: float
    cross_track_p95_m: float
    cross_track_max_m: float
    orientation_p95_deg: float
    orientation_max_deg: float
    collision: bool
    ik_success: bool

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


def report_from_arrays(
    trajectory: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    quality: QualityConfig,
    recovery: bool,
) -> EpisodeReport:
    track = np.asarray(trajectory["phase"]) == "track"
    if recovery and "recovery_window" in trajectory:
        track &= ~np.asarray(trajectory["recovery_window"], dtype=bool)
    cross_track = np.asarray(trajectory["cross_track_error"])[track]
    orientation = np.asarray(trajectory["orientation_error_deg"])[track]
    progress = float(np.max(trajectory["seam_progress"], initial=0.0))
    mean = float(np.mean(cross_track)) if cross_track.size else float("inf")
    p95 = float(np.percentile(cross_track, 95)) if cross_track.size else float("inf")
    maximum = float(np.max(cross_track, initial=0.0)) if cross_track.size else float("inf")
    orientation_p95 = float(np.percentile(orientation, 95)) if orientation.size else float("inf")
    orientation_max = float(np.max(orientation, initial=0.0)) if orientation.size else float("inf")
    collision = bool(np.any(trajectory["collision"]))
    ik_success = bool(np.all(np.asarray(trajectory["ik_residual"]) <= 0.005))
    valid = (
        progress >= quality.minimum_progress
        and mean <= quality.cross_track_mean_m
        and p95 <= quality.cross_track_p95_m
        and maximum <= quality.cross_track_max_m
        and orientation_p95 <= quality.orientation_p95_deg
        and orientation_max <= quality.orientation_max_deg
        and not collision
        and ik_success
    )
    status = (
        EpisodeStatus.VALID_RECOVERY
        if valid and recovery
        else EpisodeStatus.VALID_SUCCESS
        if valid
        else EpisodeStatus.COLLISION_FAILURE
        if collision
        else EpisodeStatus.INVALID_PLANNING
        if not ik_success
        else EpisodeStatus.INVALID_SIMULATION
    )
    return EpisodeReport(
        status,
        valid,
        progress,
        mean,
        p95,
        maximum,
        orientation_p95,
        orientation_max,
        collision,
        ik_success,
    )


def validate_episode(path: str | Path, config: AppConfig | None = None) -> EpisodeReport:
    reader = EpisodeReader(path)
    quality = (
        config.quality if config else QualityConfig(**reader.metadata["resolved_config"]["quality"])
    )
    return report_from_arrays(reader.trajectory, quality, bool(reader.metadata.get("recovery")))
