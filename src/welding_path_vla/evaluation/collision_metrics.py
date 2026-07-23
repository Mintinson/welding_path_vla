from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CollisionReport:
    collision: bool
    collision_frames: int
    collision_rate: float
    pairs: tuple[str, ...]


def collision_report(trajectory: dict[str, np.ndarray] | np.lib.npyio.NpzFile) -> CollisionReport:
    flags = np.asarray(trajectory["collision"], dtype=bool)
    raw_pairs = trajectory.get("collision_pairs", ())
    pairs = tuple(sorted({pair for value in raw_pairs for pair in str(value).split("|") if pair}))
    return CollisionReport(
        collision=bool(flags.any()),
        collision_frames=int(flags.sum()),
        collision_rate=float(flags.mean()) if flags.size else 0.0,
        pairs=pairs,
    )


__all__ = ["CollisionReport", "collision_report"]
