"""支持多文件组合的 YAML 配置入口。"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


def merge_config(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并配置，后出现的文件覆盖先出现的文件。

    Args:
        base: 已组合的基础配置。
        overlay: 当前覆盖配置。

    Returns:
        不修改输入对象的新配置字典。
    """
    merged = dict(base)
    for name, value in overlay.items():
        previous = merged.get(name)
        merged[name] = (
            merge_config(previous, value)
            if isinstance(previous, Mapping) and isinstance(value, Mapping)
            else value
        )
    return merged


def compose_config(path: str | Path, parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    """解析一个 YAML 及其 ``includes`` 列表。

    ``includes`` 中的相对路径以当前 YAML 所在目录为基准，列表越靠后的文件优先级
    越高，入口文件自身优先级最高。

    Args:
        path: 当前 YAML 文件。
        parents: 递归解析链，用于报告循环引用。

    Returns:
        已完成递归合并、且不再包含 ``includes`` 的配置。
    """
    source = Path(path).resolve()
    if source in parents:
        chain = " -> ".join(str(item) for item in (*parents, source))
        raise ValueError(f"cyclic config includes: {chain}")

    document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    includes = document.pop("includes", [])
    merged: dict[str, Any] = {}
    for included in includes:
        included_path = (source.parent / included).resolve()
        merged = merge_config(merged, compose_config(included_path, (*parents, source)))
    return merge_config(merged, document)


@contextmanager
def materialized_config(path: str | Path) -> Iterator[Path]:
    """把组合后的配置临时交给 Draccus 进行类型解析和命令行覆盖。

    Args:
        path: 用户指定的模块化 YAML 入口。

    Yields:
        仅在上下文内有效的扁平 YAML 路径。
    """
    with tempfile.TemporaryDirectory(prefix="welding_vla_config_") as directory:
        target = Path(directory) / "config.yaml"
        target.write_text(
            yaml.safe_dump(compose_config(path), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        yield target


__all__ = ["compose_config", "materialized_config", "merge_config"]
