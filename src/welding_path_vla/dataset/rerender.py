"""使用已记录仿真状态原地重渲染 raw 和 LeRobot 数据集。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from welding_path_vla.core.config import AppConfig, LeRobotExportConfig
from welding_path_vla.dataset.export_lerobot import (
    GLOBAL_IMAGE,
    MANIFEST_PATH,
    export_lerobot_many,
    source_id,
)
from welding_path_vla.dataset.raw_schema import EpisodeReader
from welding_path_vla.dataset.video import VideoRecorder

LOW_DIMENSIONAL_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


@dataclass(frozen=True, slots=True)
class RerenderReport:
    """一次原地重渲染的结果。"""

    dataset: str
    dataset_type: str
    raw_datasets: list[str]
    episodes: int
    frames: int
    backup: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """返回可直接写为 JSON 的报告。"""
        return asdict(self)


def video_frame_count(path: Path) -> int:
    """读取视频帧数，并拒绝无法打开的文件。"""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open rendered video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def decode_episode_config(values: dict[str, Any]) -> AppConfig:
    """解码历史 episode 配置，并迁移已废弃的策略字段。

    Args:
        values: ``metadata.json`` 中保存的 ``resolved_config``。

    Returns:
        可用于重建仿真的当前配置对象。
    """
    if "include_current" not in values.get("policy", {}):
        return AppConfig.from_dict(values)
    migrated = dict(values)
    migrated["policy"] = dict(values["policy"])
    migrated["policy"].pop("include_current")
    return AppConfig.from_dict(migrated)


def render_episode_videos(path: str | Path, show_progress: bool = True) -> int:
    """按旧 episode 的工件位姿和逐帧关节状态重新录制双相机视频。

    Args:
        path: 包含 ``trajectory.npz`` 和 ``metadata.json`` 的 raw episode。
        show_progress: 是否显示帧级渲染进度。

    Returns:
        重渲染的视频帧数。
    """
    episode = EpisodeReader(path)
    config = decode_episode_config(episode.metadata["resolved_config"])
    required = ("workpiece_position", "workpiece_quaternion_wxyz")
    missing = [name for name in required if name not in episode.metadata]
    if missing:
        raise ValueError(f"episode lacks simulation reconstruction metadata: {missing}")
    joint_positions = episode.trajectory["joint_position"].copy()
    tcp_positions = episode.trajectory["tcp_position"].copy()
    episode.trajectory.close()
    if len(joint_positions) != len(tcp_positions):
        raise ValueError(f"joint/TCP state counts disagree: {path}")

    episode_path = Path(path)
    temporary = Path(tempfile.mkdtemp(prefix=".rerender-", dir=episode_path))
    simulation = None
    recorder = None
    try:
        from welding_path_vla.simulation import WeldingEnv

        simulation = WeldingEnv(config, seed=episode.metadata.get("seed"), ignore_done=True)
        simulation.mj_model.body_pos[simulation.workpiece_id] = episode.metadata[
            "workpiece_position"
        ]
        simulation.mj_model.body_quat[simulation.workpiece_id] = episode.metadata[
            "workpiece_quaternion_wxyz"
        ]
        simulation.robosuite_sim.forward()
        simulation.reset_weld_visuals()
        camera_names = (config.camera.global_name, config.camera.wrist_name)
        recorder = VideoRecorder.start(temporary, camera_names, config.timing.policy_hz)
        states = zip(joint_positions, tcp_positions, strict=True)
        for joints, tcp in tqdm(
            states,
            total=len(joint_positions),
            desc=f"  {episode_path.name}",
            unit="frame",
            leave=False,
            disable=not show_progress,
        ):
            simulation.set_joint_position(joints)
            simulation.update_weld_visuals(tcp)
            observation = simulation.observe()
            recorder.append(simulation.images_from_observation(observation))
        rendered = recorder.finish()
        recorder = None
        for video in map(Path, rendered):
            if video_frame_count(video) != len(joint_positions):
                raise RuntimeError(f"rendered video has an unexpected frame count: {video}")
        for name in camera_names:
            os.replace(temporary / f"{name}.mp4", episode_path / f"{name}.mp4")
        return len(joint_positions)
    finally:
        if recorder is not None:
            recorder.close()
        if simulation is not None:
            simulation.close()
        shutil.rmtree(temporary, ignore_errors=True)


def rerender_raw_dataset(
    root: str | Path,
    episode_names: list[str] | None = None,
    show_progress: bool = True,
) -> RerenderReport:
    """原地替换 raw 数据集的视频，其他 episode 工件保持不变。"""
    dataset = Path(root)
    paths = (
        [dataset / "episodes" / name for name in episode_names]
        if episode_names is not None
        else sorted((dataset / "episodes").glob("episode_*"))
    )
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"raw episodes not found: {missing}")
    frames = 0
    for path in tqdm(
        paths,
        desc=f"Rerendering {dataset.name}",
        unit="episode",
        disable=not show_progress,
    ):
        frames += render_episode_videos(path, show_progress)
    return RerenderReport(str(dataset), "raw", [str(dataset)], len(paths), frames)


def resolve_raw_sources(
    manifest: dict[str, Any],
    raw_dataset_glob: str,
) -> list[tuple[Path, list[str]]]:
    """按 LeRobot manifest 顺序解析本地 raw 数据源。"""
    from glob import glob

    candidates = [Path(path) for path in sorted(glob(raw_dataset_glob))]
    by_id = {source_id(path): path for path in candidates}
    resolved = []
    for identifier, names in manifest["sources"].items():
        direct = Path(identifier)
        root = direct if direct.is_dir() else by_id.get(identifier)
        if root is None:
            raise FileNotFoundError(
                f"cannot resolve raw source {identifier!r}; searched {raw_dataset_glob!r}"
            )
        resolved.append((root, list(names)))
    return resolved


def stage_raw_sources(
    sources: list[tuple[Path, list[str]]],
    root: Path,
) -> list[Path]:
    """创建只包含 LeRobot 已导出 episode 的轻量符号链接数据集。"""
    staged = []
    for index, (source, names) in enumerate(sources):
        target = root / f"source-{index:03d}"
        episodes = target / "episodes"
        episodes.mkdir(parents=True)
        summary = source / "dataset.json"
        if summary.exists():
            shutil.copy2(summary, target / summary.name)
        for name in names:
            episode = source / "episodes" / name
            if not episode.is_dir():
                raise FileNotFoundError(f"raw episode not found: {episode}")
            (episodes / name).symlink_to(episode.resolve(), target_is_directory=True)
        staged.append(target)
    return staged


def export_options(info: dict[str, Any]) -> LeRobotExportConfig:
    """从现有 LeRobot schema 恢复视觉存储和编码参数。"""
    feature = info["features"][GLOBAL_IMAGE]
    video_info = feature.get("info", {})
    codec = {"av1": "libsvtav1"}.get(video_info.get("video.codec"), video_info.get("video.codec"))
    return LeRobotExportConfig(
        save_images=feature["dtype"] == "image",
        video_codec=codec or "libsvtav1",
        video_quality=video_info.get("video.crf", 30),
        video_preset=video_info.get("video.preset", "10"),
    )


def parquet_table(root: Path, relative: str, columns: tuple[str, ...] | None = None) -> Any:
    """按文件名顺序读取一组 Parquet shard。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / relative).rglob("*.parquet"))
    selected = list(columns) if columns is not None else None
    tables = [pq.read_table(path, columns=selected) for path in paths]
    if not tables:
        raise ValueError(f"no parquet files found under {root / relative}")
    return pa.concat_tables(tables).combine_chunks()


def validate_lerobot_preserved(source: Path, rendered: Path) -> None:
    """确认重建前后的低维轨迹、动作、任务和索引逐值相等。"""
    import pyarrow.parquet as pq

    original = parquet_table(source, "data", LOW_DIMENSIONAL_COLUMNS)
    replacement = parquet_table(rendered, "data", LOW_DIMENSIONAL_COLUMNS)
    for name in LOW_DIMENSIONAL_COLUMNS:
        if not original[name].equals(replacement[name]):
            raise ValueError(f"rerender changed LeRobot column: {name}")
    task_columns = ["task_index", "task"]
    original_tasks = pq.read_table(source / "meta/tasks.parquet", columns=task_columns)
    replacement_tasks = pq.read_table(rendered / "meta/tasks.parquet", columns=task_columns)
    if not original_tasks.equals(replacement_tasks):
        raise ValueError("rerender changed LeRobot task mapping")


def replace_dataset_directory(source: Path, rendered: Path, keep_backup: bool) -> Path | None:
    """在同一文件系统原子切换数据集目录，并在失败时恢复旧目录。"""
    backup = source.with_name(f"{source.name}_before_rerender")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    os.replace(source, backup)
    try:
        os.replace(rendered, source)
    except Exception:
        os.replace(backup, source)
        raise
    if keep_backup:
        return backup
    shutil.rmtree(backup)
    return None


def rerender_lerobot_dataset(
    root: str | Path,
    raw_dataset_glob: str = "datasets/*_raw_v2",
    keep_backup: bool = False,
    show_progress: bool = True,
) -> RerenderReport:
    """通过 raw 唯一事实源重建 LeRobot 视觉数据，并原地替换数据集。"""
    dataset = Path(root)
    manifest = json.loads((dataset / MANIFEST_PATH).read_text(encoding="utf-8"))
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    source_count = sum(len(names) for names in manifest["sources"].values())
    if source_count != info["total_episodes"]:
        raise ValueError("LeRobot dataset is incomplete or currently being exported")
    sources = resolve_raw_sources(manifest, raw_dataset_glob)
    raw_reports = [rerender_raw_dataset(source, names, show_progress) for source, names in sources]
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset.name}.rerender-", dir=dataset.parent))
    rendered = temporary / "dataset"
    backup = None
    try:
        staged = stage_raw_sources(sources, temporary / "raw")
        representation = manifest["action_representation"]
        export_lerobot_many(
            staged,
            rendered,
            "local/rerendered-weldpath",
            export_options(info),
            representation["horizon"],
            representation["stride"],
        )
        validate_lerobot_preserved(dataset, rendered)
        replacement_manifest = json.loads((rendered / MANIFEST_PATH).read_text(encoding="utf-8"))
        if replacement_manifest["sources"] != manifest["sources"]:
            raise ValueError("rerender changed LeRobot raw source mapping")
        backup = replace_dataset_directory(dataset, rendered, keep_backup)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return RerenderReport(
        str(dataset),
        "lerobot",
        [str(source) for source, _ in sources],
        sum(report.episodes for report in raw_reports),
        sum(report.frames for report in raw_reports),
        str(backup) if backup else None,
    )


def rerender_dataset(
    root: str | Path,
    raw_dataset_glob: str = "datasets/*_raw_v2",
    keep_backup: bool = False,
    show_progress: bool = True,
) -> RerenderReport:
    """自动识别 raw 或 LeRobot 格式并执行原地重渲染。"""
    dataset = Path(root)
    if (dataset / "episodes").is_dir():
        return rerender_raw_dataset(dataset, show_progress=show_progress)
    if (dataset / "meta/info.json").is_file() and (dataset / MANIFEST_PATH).is_file():
        return rerender_lerobot_dataset(
            dataset,
            raw_dataset_glob,
            keep_backup,
            show_progress,
        )
    raise ValueError(f"not a supported raw or LeRobot dataset: {dataset}")


__all__ = [
    "RerenderReport",
    "render_episode_videos",
    "rerender_dataset",
    "rerender_lerobot_dataset",
    "rerender_raw_dataset",
    "resolve_raw_sources",
    "validate_lerobot_preserved",
]
