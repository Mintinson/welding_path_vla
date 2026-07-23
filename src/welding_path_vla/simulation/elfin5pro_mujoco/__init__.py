"""Resolve the packaged Elfin5-Pro MuJoCo/URDF assets."""

from importlib.resources import files
from pathlib import Path


def model_path(asset: str = "elfin5/elfin5_welding.xml") -> Path:
    return Path(str(files("welding_path_vla").joinpath("assets", asset)))


__all__ = ["model_path"]
