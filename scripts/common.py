"""项目脚本共享的配置与 JSON 输出辅助函数。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from draccus.argparsing import wrap


def cli[Config](command: Callable[[Config], None]) -> Callable[[], None]:
    """为 draccus 装饰器补充准确的无参入口类型。"""
    return cast(Callable[[], None], wrap()(command))


def output_json(value: object, output: str | Path | None = None) -> None:
    """向终端或指定文件输出格式化 JSON。"""
    document = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    else:
        print(document)
