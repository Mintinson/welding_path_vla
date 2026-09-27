"""项目脚本共享的配置与 JSON 输出辅助函数。"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import cast

from draccus.argparsing import wrap

from welding_path_vla.core.config_files import materialized_config


def cli[Config](command: Callable[[Config], None]) -> Callable[[], None]:
    """组合 YAML 模块，再由 Draccus 解析类型和命令行覆盖。"""
    source_config_path: str | None = None

    @wraps(command)
    def configured_command(config: Config) -> None:
        """在执行脚本前保存用户实际传入的入口 YAML 路径。"""
        if source_config_path is not None and hasattr(config, "source_config_path"):
            config.source_config_path = source_config_path
        command(config)

    wrapped = cast(Callable[[], None], wrap()(configured_command))

    def entrypoint() -> None:
        nonlocal source_config_path
        config_argument = next(
            (argument for argument in sys.argv[1:] if argument.startswith("--config_path=")),
            None,
        )
        if config_argument is None:
            wrapped()
            return

        source = config_argument.split("=", 1)[1]
        source_config_path = source
        with materialized_config(source) as config_path:
            arguments = sys.argv
            sys.argv = [
                f"--config_path={config_path}" if item == config_argument else item
                for item in arguments
            ]
            try:
                wrapped()
            finally:
                sys.argv = arguments
                source_config_path = None

    return entrypoint


def output_json(value: object, output: str | Path | None = None) -> None:
    """向终端或指定文件输出格式化 JSON。"""
    document = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    else:
        print(document)
