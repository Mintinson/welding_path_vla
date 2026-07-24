"""项目脚本共享的配置与 JSON 输出辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path

from welding_path_vla.core.config import AppConfig


def load_config(path: str, dataset: str | None = None) -> AppConfig:
    """加载统一配置，并按需覆盖采集目录。"""
    config = AppConfig.load(path)
    if dataset:
        config.collection.dataset_root = dataset
    return config


def output_json(value: object, output: str | None = None) -> None:
    """向终端或指定文件输出格式化 JSON。"""
    document = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    else:
        print(document)
