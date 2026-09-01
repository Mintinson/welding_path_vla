"""给已有 LeRobot 数据集原地补充焊接任务参数。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

from datasets import Features
from welding_path_vla.dataset.export_lerobot import MANIFEST_PATH, source_id
from welding_path_vla.dataset.task_parameters import (
    TASK_DIRECTION,
    TASK_FEATURES,
    TASK_PARAMETERS,
    task_feature_values,
)

STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


@dataclass(frozen=True, slots=True)
class EpisodeTaskParameters:
    """一个目标 episode 对应的数值任务参数。"""

    direction: np.ndarray
    parameters: np.ndarray
    length: int


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """任务参数迁移或校验结果。"""

    dataset: str
    episodes: int
    frames: int
    data_files_changed: int
    metadata_files_changed: int
    verified: bool

    def as_dict(self) -> dict[str, object]:
        """转换为 JSON 可序列化字典。"""
        return asdict(self)


def read_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: dict[str, Any]) -> None:
    """原子写入 JSON，避免中断留下半个元数据文件。"""
    temporary = path.with_suffix(".task-parameters.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def raw_roots_by_source(roots: list[Path]) -> dict[str, Path]:
    """按导出清单使用的稳定 source id 索引 raw 数据集。"""
    indexed: dict[str, Path] = {}
    for root in roots:
        identifier = source_id(root)
        if identifier in indexed:
            raise ValueError(f"duplicate raw source id: {identifier}")
        indexed[identifier] = root
    return indexed


def target_episode_rows(dataset: Path) -> list[dict[str, Any]]:
    """按 episode_index 读取目标数据集的长度和语言任务。"""
    paths = sorted((dataset / "meta/episodes").rglob("*.parquet"))
    rows = [
        row
        for path in paths
        for row in pq.read_table(path, columns=["episode_index", "tasks", "length"]).to_pylist()
    ]
    return sorted(rows, key=lambda row: row["episode_index"])


def episode_mapping(dataset: Path, raw_roots: list[Path]) -> dict[int, EpisodeTaskParameters]:
    """用导出清单恢复目标 episode 到 raw episode 的一一对应关系。

    映射完成后还会比对 episode 长度和语言任务。若旧清单顺序不足以唯一还原，
    会在修改任何文件前终止，避免把参数写到错误轨迹。
    """
    manifest = read_json(dataset / MANIFEST_PATH)
    indexed = raw_roots_by_source(raw_roots)
    raw_paths = []
    for identifier, names in manifest["sources"].items():
        if identifier not in indexed:
            raise ValueError(f"raw dataset not found for manifest source: {identifier}")
        raw_paths.extend(indexed[identifier] / "episodes" / name for name in names)

    target_rows = target_episode_rows(dataset)
    if len(raw_paths) != len(target_rows):
        raise ValueError(
            f"manifest has {len(raw_paths)} episodes, target metadata has {len(target_rows)}"
        )

    mapping = {}
    for row, raw_path in zip(target_rows, raw_paths, strict=True):
        metadata = read_json(raw_path / "metadata.json")
        with np.load(raw_path / "trajectory.npz") as trajectory:
            length = len(trajectory["command_delta_pose_seam"])
        if length != row["length"] or metadata["instruction"] not in row["tasks"]:
            raise ValueError(
                "manifest order cannot safely identify target episode "
                f"{row['episode_index']}: raw={raw_path.name}"
            )
        values = task_feature_values(metadata)
        mapping[row["episode_index"]] = EpisodeTaskParameters(
            values[TASK_DIRECTION],
            values[TASK_PARAMETERS],
            length,
        )
    return mapping


def task_arrays(
    episode_indices: list[int],
    mapping: dict[int, EpisodeTaskParameters],
) -> tuple[pa.Array, pa.Array]:
    """为一批数据帧创建两个固定长度 Arrow 数组。"""
    directions = [int(mapping[index].direction[0]) for index in episode_indices]
    parameters = [mapping[index].parameters.tolist() for index in episode_indices]
    parameter_type = pa.list_(pa.field("element", pa.float32()), 4)
    return pa.array(directions, type=pa.int64()), pa.array(parameters, type=parameter_type)


def episode_stats(value: EpisodeTaskParameters) -> dict[str, dict[str, np.ndarray]]:
    """按 LeRobot 官方算法计算两个常量特征的 episode 统计。"""
    data: dict[str, list[str] | np.ndarray] = {
        TASK_DIRECTION: np.repeat(value.direction[None, :], value.length, axis=0),
        TASK_PARAMETERS: np.repeat(value.parameters[None, :], value.length, axis=0),
    }
    return compute_episode_stats(data, TASK_FEATURES)


def append_data_columns(
    table: pa.Table,
    mapping: dict[int, EpisodeTaskParameters],
) -> pa.Table:
    """给一个数据 row group 追加任务参数列。"""
    indices = table.column("episode_index").to_pylist()
    direction, parameters = task_arrays(indices, mapping)
    return table.append_column(TASK_DIRECTION, direction).append_column(TASK_PARAMETERS, parameters)


def append_metadata_columns(
    table: pa.Table,
    statistics: dict[int, dict[str, dict[str, np.ndarray]]],
) -> pa.Table:
    """给 episode metadata 追加与 LeRobot writer 相同的统计列。"""
    indices = table.column("episode_index").to_pylist()
    for feature in (TASK_DIRECTION, TASK_PARAMETERS):
        for stat in STAT_NAMES:
            values = [statistics[index][feature][stat].tolist() for index in indices]
            scalar = pa.int64() if stat == "count" else pa.float64()
            table = table.append_column(
                f"stats/{feature}/{stat}",
                pa.array(values, type=pa.list_(pa.field("element", scalar))),
            )
    return table


def rewrite_parquet(path: Path, transform: Any) -> None:
    """逐 row group 原子重写 Parquet，峰值内存与单个 row group 大小一致。"""
    source = pq.ParquetFile(path)
    first = transform(source.read_row_group(0))
    schema = Features.from_arrow_schema(first.schema.remove_metadata()).arrow_schema
    compression = source.metadata.row_group(0).column(0).compression.lower()
    temporary = path.with_suffix(".task-parameters.tmp")
    with pq.ParquetWriter(temporary, schema, compression=compression) as writer:
        writer.write_table(first.cast(schema))
        for index in range(1, source.num_row_groups):
            writer.write_table(transform(source.read_row_group(index)).cast(schema))
    temporary.chmod(path.stat().st_mode)
    temporary.replace(path)


def feature_state(names: list[str], expected: set[str], path: Path) -> bool:
    """判断一组字段是完整存在还是完整缺失，拒绝半迁移文件。"""
    present = expected.intersection(names)
    if present and present != expected:
        raise ValueError(f"partially migrated parquet file: {path}")
    return present == expected


def verify_data_table(
    table: pa.Table,
    mapping: dict[int, EpisodeTaskParameters],
    path: Path,
) -> None:
    """校验数据帧上的参数值与 raw 元数据逐项一致。"""
    indices = table.column("episode_index").to_pylist()
    expected_direction, expected_parameters = task_arrays(indices, mapping)
    actual_direction = np.asarray(table.column(TASK_DIRECTION).to_pylist(), dtype=np.int64)
    expected_direction_array = np.asarray(expected_direction.to_pylist(), dtype=np.int64)
    if not np.array_equal(actual_direction, expected_direction_array):
        raise ValueError(f"incorrect {TASK_DIRECTION} values: {path}")
    actual = np.asarray(table.column(TASK_PARAMETERS).to_pylist(), dtype=np.float32)
    expected = np.asarray(expected_parameters.to_pylist(), dtype=np.float32)
    if not np.array_equal(actual, expected):
        raise ValueError(f"incorrect {TASK_PARAMETERS} values: {path}")


def migrate_data_files(
    dataset: Path,
    mapping: dict[int, EpisodeTaskParameters],
    verify_only: bool,
) -> int:
    """迁移或校验所有逐帧数据 Parquet。"""
    changed = 0
    expected = {TASK_DIRECTION, TASK_PARAMETERS}
    for path in sorted((dataset / "data").rglob("*.parquet")):
        source = pq.ParquetFile(path)
        migrated = feature_state(source.schema_arrow.names, expected, path)
        if not migrated:
            if verify_only:
                raise ValueError(f"task parameters are missing: {path}")
            rewrite_parquet(path, lambda table: append_data_columns(table, mapping))
            changed += 1
            source = pq.ParquetFile(path)
        for index in range(source.num_row_groups):
            verify_data_table(source.read_row_group(index), mapping, path)
    return changed


def verify_metadata_table(
    table: pa.Table,
    statistics: dict[int, dict[str, dict[str, np.ndarray]]],
    path: Path,
) -> None:
    """校验 episode metadata 中新增统计量的形状和值。"""
    indices = table.column("episode_index").to_pylist()
    for feature in (TASK_DIRECTION, TASK_PARAMETERS):
        for stat in STAT_NAMES:
            name = f"stats/{feature}/{stat}"
            actual = np.asarray(table.column(name).to_pylist())
            expected = np.asarray([statistics[index][feature][stat] for index in indices])
            if not np.allclose(actual, expected):
                raise ValueError(f"incorrect episode statistic {name}: {path}")


def migrate_metadata_files(
    dataset: Path,
    statistics: dict[int, dict[str, dict[str, np.ndarray]]],
    verify_only: bool,
) -> int:
    """迁移或校验所有 episode metadata Parquet。"""
    changed = 0
    expected = {
        f"stats/{feature}/{stat}"
        for feature in (TASK_DIRECTION, TASK_PARAMETERS)
        for stat in STAT_NAMES
    }
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        source = pq.ParquetFile(path)
        migrated = feature_state(source.schema_arrow.names, expected, path)
        if not migrated:
            if verify_only:
                raise ValueError(f"task parameter statistics are missing: {path}")
            rewrite_parquet(path, lambda table: append_metadata_columns(table, statistics))
            changed += 1
            source = pq.ParquetFile(path)
        for index in range(source.num_row_groups):
            verify_metadata_table(source.read_row_group(index), statistics, path)
    return changed


def verify_global_stats(
    actual: dict[str, Any],
    expected: dict[str, dict[str, np.ndarray]],
) -> None:
    """校验全局统计量，确保训练归一化元数据完整。"""
    for feature in (TASK_DIRECTION, TASK_PARAMETERS):
        if feature not in actual:
            raise ValueError(f"global statistics are missing: {feature}")
        for stat in STAT_NAMES:
            if not np.allclose(actual[feature][stat], expected[feature][stat]):
                raise ValueError(f"incorrect global statistic: {feature}/{stat}")


def migrate_dataset_metadata(
    dataset: Path,
    statistics: dict[int, dict[str, dict[str, np.ndarray]]],
    verify_only: bool,
) -> None:
    """最后更新 info.json 和 stats.json；不触碰语言 task/task_index。"""
    info_path = dataset / "meta/info.json"
    stats_path = dataset / "meta/stats.json"
    info = read_json(info_path)
    expected_features = {
        key: {**value, "shape": list(value["shape"])} for key, value in TASK_FEATURES.items()
    }
    existing = {key: info["features"].get(key) for key in TASK_FEATURES}
    global_stats = aggregate_stats(list(statistics.values()))
    stats = read_json(stats_path)
    if verify_only:
        if existing != expected_features:
            raise ValueError("info.json is missing or has incompatible task parameter features")
        verify_global_stats(stats, global_stats)
        return
    for key, value in existing.items():
        if value is not None and value != expected_features[key]:
            raise ValueError(f"incompatible feature in info.json: {key}")
    info["features"].update(expected_features)
    stats.update(
        {
            feature: {name: value.tolist() for name, value in feature_stats.items()}
            for feature, feature_stats in global_stats.items()
        }
    )
    write_json(stats_path, stats)
    write_json(info_path, info)


def migrate_task_parameters(
    dataset: str | Path,
    raw_roots: Sequence[str | Path],
    *,
    verify_only: bool = False,
) -> MigrationReport:
    """原地迁移已有 LeRobot 数据集，或只校验此前迁移结果。

    数据文件按 row group 流式处理并逐文件原子替换；如果进程中断，重新运行同一
    命令会跳过已完成文件并继续。视频、动作、状态、语言任务和索引均不会改变。
    """
    root = Path(dataset)
    mapping = episode_mapping(root, [Path(path) for path in raw_roots])
    statistics = {index: episode_stats(value) for index, value in mapping.items()}
    data_changes = migrate_data_files(root, mapping, verify_only)
    metadata_changes = migrate_metadata_files(root, statistics, verify_only)
    migrate_dataset_metadata(root, statistics, verify_only)
    info = read_json(root / "meta/info.json")
    return MigrationReport(
        str(root),
        len(mapping),
        int(info["total_frames"]),
        data_changes,
        metadata_changes,
        True,
    )
