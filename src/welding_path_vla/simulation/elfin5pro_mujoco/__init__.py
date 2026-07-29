"""解析打包后的 Elfin5-Pro MuJoCo / URDF 资产路径。"""

from importlib.resources import files
from pathlib import Path


def model_path(asset: str = "elfin5/elfin5pro_robot.xml") -> Path:
    """返回包内资产的文件系统路径。

    Args:
        asset: 相对 ``assets`` 目录的模型路径。

    Returns:
        可传给 MuJoCo 或 robosuite 的路径。
    """
    return Path(str(files("welding_path_vla").joinpath("assets", asset)))


__all__ = ["model_path"]
