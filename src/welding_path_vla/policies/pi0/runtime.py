"""π0 运行时。"""

from pathlib import Path

from welding_path_vla.policies.pi_family.runtime import PIRuntime
from welding_path_vla.policies.pi_family.spec import PI0


def load(checkpoint: str | Path, device: str) -> PIRuntime:
    """加载 π0 checkpoint 及其前后处理器。"""
    return PIRuntime.from_pretrained(checkpoint, device, PI0)


__all__ = ["load"]
