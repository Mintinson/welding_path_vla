#!/usr/bin/env python3
"""一次性把已有 raw 和 LeRobot 数据集的中文任务指令迁移为英文。"""

import json
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from common import cli, output_json

INSTRUCTIONS = {
    "沿 L 形工件的直线角焊缝完成焊接轨迹。": (
        "Weld along the straight fillet seam of the L-shaped workpiece."
    ),
    "沿圆管与底板连接处的下圆弧完成角焊轨迹。": (
        "Weld along the lower circular fillet seam where the pipe meets the base plate."
    ),
    "沿圆管上沿的圆弧完成焊接轨迹。": (
        "Weld along the circular seam around the top rim of the pipe."
    ),
    "沿平板上的周期曲线焊缝完成焊接轨迹。": (
        "Weld along the periodic curved seam on the flat plate."
    ),
}
PIPE_BASE_ZH = "沿圆管与底板连接处的下圆弧完成角焊轨迹。"
PIPE_DETAIL = re.compile(
    rf"{re.escape(PIPE_BASE_ZH[:-1])}; "
    r"从工件局部圆周角 (?P<start>-?\d+(?:\.\d+)?)° 起, "
    r"沿(?P<direction>顺时针|逆时针)执行 (?P<sweep>\d+(?:\.\d+)?)°, "
    r"工作角 (?P<work>\d+(?:\.\d+)?)°, 行走角 (?P<travel>\d+(?:\.\d+)?)°。"
)


@dataclass
class MigrationArguments:
    """任务语言迁移参数。

    Attributes:
        raw_glob: 待修改的 raw 数据集路径表达式；设为空可跳过。
        lerobot_datasets: 待修改的本地 LeRobot 数据集目录。
    """

    raw_glob: str | None = "datasets/*_raw_v2"
    lerobot_datasets: list[Path] = field(default_factory=list)


def translate_instruction(instruction: str) -> str:
    """把已知中文焊接指令翻译为含义一致的英文。"""
    if instruction in INSTRUCTIONS:
        return INSTRUCTIONS[instruction]
    if instruction in INSTRUCTIONS.values() or instruction.startswith(
        INSTRUCTIONS[PIPE_BASE_ZH][:-1] + ";"
    ):
        return instruction
    match = PIPE_DETAIL.fullmatch(instruction)
    if match is None:
        raise ValueError(f"unknown task instruction: {instruction}")
    direction = "clockwise" if match["direction"] == "顺时针" else "counterclockwise"
    return (
        f"{INSTRUCTIONS[PIPE_BASE_ZH][:-1]}; "
        f"start at a local circumferential angle of {match['start']}°, "
        f"travel {match['sweep']}° {direction}, with a {match['work']}° work angle "
        f"and a {match['travel']}° travel angle."
    )


def write_json(path: Path, document: dict) -> None:
    """原子写入 JSON 文档。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def migrate_raw_dataset(root: Path) -> int:
    """修改一个 raw 数据集内的顶层和解析后任务指令。"""
    changed = 0
    for path in sorted((root / "episodes").glob("episode_*/metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        instruction = translate_instruction(metadata["instruction"])
        resolved_task = metadata.get("resolved_config", {}).get("task")
        resolved_instruction = (
            translate_instruction(resolved_task["instruction"]) if resolved_task else instruction
        )
        if metadata["instruction"] == instruction and (
            resolved_task is None or resolved_task["instruction"] == resolved_instruction
        ):
            continue
        metadata["instruction"] = instruction
        if resolved_task is not None:
            resolved_task["instruction"] = resolved_instruction
        write_json(path, metadata)
        changed += 1
    return changed


def rewrite_episode_tasks(path: Path) -> bool:
    """原子翻译一个 LeRobot episode metadata Parquet 文件。"""
    source = pq.ParquetFile(path)
    changed = False
    temporary = path.with_suffix(".tmp")
    with pq.ParquetWriter(temporary, source.schema_arrow, compression="snappy") as writer:
        for index in range(source.num_row_groups):
            table = source.read_row_group(index)
            tasks = table.column("tasks")
            original = tasks.to_pylist()
            translated = [
                [translate_instruction(task) for task in row] for row in original
            ]
            changed |= translated != original
            column = pa.array(translated, type=tasks.type)
            updated = table.set_column(table.schema.get_field_index("tasks"), "tasks", column)
            writer.write_table(updated)
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def migrate_lerobot_dataset(root: Path) -> tuple[int, int]:
    """翻译 LeRobot 的任务表与 episode metadata，不改 task_index。"""
    tasks_path = root / "meta/tasks.parquet"
    tasks = pq.read_table(tasks_path)
    original = tasks.column("task").to_pylist()
    translated = [translate_instruction(task) for task in original]
    task_changes = sum(old != new for old, new in zip(original, translated, strict=True))
    if task_changes:
        column = pa.array(translated, type=tasks.column("task").type)
        updated = tasks.set_column(tasks.schema.get_field_index("task"), "task", column)
        temporary = tasks_path.with_suffix(".tmp")
        pq.write_table(updated, temporary, compression="snappy")
        temporary.replace(tasks_path)
    episode_changes = sum(
        rewrite_episode_tasks(path) for path in sorted((root / "meta/episodes").rglob("*.parquet"))
    )
    return task_changes, episode_changes


@cli
def main(config: MigrationArguments) -> None:
    """执行一次性迁移并输出修改统计。"""
    raw_roots = [Path(path) for path in sorted(glob(config.raw_glob))] if config.raw_glob else []
    raw_changes = {str(root): migrate_raw_dataset(root) for root in raw_roots}
    lerobot_changes = {
        str(root): migrate_lerobot_dataset(root) for root in config.lerobot_datasets
    }
    output_json({"raw_episodes": raw_changes, "lerobot_metadata": lerobot_changes})


if __name__ == "__main__":
    main()
