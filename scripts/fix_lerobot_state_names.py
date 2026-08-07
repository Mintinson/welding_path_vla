#!/usr/bin/env python3
"""一次性修正现有 LeRobot 数据集的状态分量名称。"""

import json
from dataclasses import dataclass
from pathlib import Path

from common import cli, output_json

from welding_path_vla.dataset.export_lerobot import OBSERVATION_STATE_NAMES


@dataclass
class MigrationArguments:
    """一次性状态名称迁移参数。

    Attributes:
        dataset: 已完成转换的本地 LeRobot 数据集目录。
    """

    dataset: Path = Path("datasets/weldpath_lerobot_relative_v1")


def migrate_state_names(dataset: Path) -> bool:
    """原子替换 `observation.state` 的 13 个分量名称。

    Args:
        dataset: 已完成转换的本地 LeRobot 数据集目录。

    Returns:
        本次是否修改了元数据。
    """
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info["features"]["observation.state"]
    old_names = [f"state_{index}" for index in range(len(OBSERVATION_STATE_NAMES))]
    if feature["names"] == list(OBSERVATION_STATE_NAMES):
        return False
    if feature["names"] != old_names:
        raise ValueError(f"unexpected observation.state names: {feature['names']}")

    feature["names"] = list(OBSERVATION_STATE_NAMES)
    temporary = info_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(info_path)
    return True


@cli
def main(config: MigrationArguments) -> None:
    """迁移本地元数据并输出修改结果。"""
    changed = migrate_state_names(config.dataset)
    output_json(
        {
            "dataset": str(config.dataset),
            "changed": changed,
            "state_names": list(OBSERVATION_STATE_NAMES),
        }
    )


if __name__ == "__main__":
    main()
