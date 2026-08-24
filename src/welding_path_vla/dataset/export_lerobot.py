"""把原始焊接 episode 增量转换为 LeRobot 数据集。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from ctypes import CDLL
from dataclasses import asdict, dataclass, replace
from gc import collect
from pathlib import Path
from sys import platform
from time import sleep
from typing import Any

import cv2
import httpx
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from welding_path_vla.core.config import LeRobotExportConfig
from welding_path_vla.core.domain import EpisodeStatus
from welding_path_vla.core.geometry import relative_ee_actions_from_absolute
from welding_path_vla.dataset.actions import (
    ABSOLUTE_ACTION_NAMES,
    build_absolute_actions,
)
from welding_path_vla.dataset.raw_schema import METADATA_FILE, EpisodeReader

GLOBAL_IMAGE = "observation.images.global"
WRIST_IMAGE = "observation.images.wrist"
OBSERVATION_STATE_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_qw",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
)
MANIFEST_PATH = Path("meta/welding_path_vla_export.json")
VALID_STATUSES = {EpisodeStatus.VALID_SUCCESS.value, EpisodeStatus.VALID_RECOVERY.value}
ACTION_REPRESENTATION = "relative_action"
ACTION_STORAGE = "absolute_ee_world"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExportReport:
    """一次转换请求的选择与写入统计。"""

    output: str
    selected_episodes: int
    exported_episodes: int
    skipped_episodes: int
    first_episode: int | None
    last_episode: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchExportReport:
    """一次单源或多源转换的汇总统计。

    Attributes:
        output: 目标 LeRobot 数据集路径。
        sources: 本次请求的原始数据集路径。
        selected_episodes: 通过编号和质量筛选的 episode 数。
        exported_episodes: 本次新写入的 episode 数。
        skipped_episodes: 因增量清单已存在而跳过的 episode 数。
        hub_url: 上传成功后的 Hugging Face Dataset 地址。
    """

    output: str
    sources: list[str]
    selected_episodes: int
    exported_episodes: int
    skipped_episodes: int
    hub_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        """转换为可直接输出的 JSON 字典。"""
        return asdict(self)


def episode_number(path: Path) -> int:
    """从 `episode_000123` 目录名提取编号。"""
    return int(path.name.rsplit("_", 1)[1])


def valid_episode_paths(
    dataset_root: str | Path,
    start_episode: int | None = None,
    end_episode: int | None = None,
) -> list[Path]:
    """按源 episode 编号闭区间选择有效数据。"""
    root = Path(dataset_root)
    paths = sorted((root / "episodes").glob("episode_*"), key=episode_number)
    selected = [
        path
        for path in paths
        if (start_episode is None or episode_number(path) >= start_episode)
        and (end_episode is None or episode_number(path) <= end_episode)
    ]
    return [
        path
        for path in selected
        if json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))["quality"]["status"]
        in VALID_STATUSES
    ]


def source_id(root: Path) -> str:
    """生成不依赖绝对路径的原始数据集标识。"""
    summary_path = root / "dataset.json"
    if not summary_path.exists():
        return str(root.resolve())
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    values = [str(summary.get(key, "")) for key in ("dataset", "format", "seed")]
    return "|".join(values) if any(values) else str(root.resolve())


def read_dataset_info(destination: Path) -> dict[str, Any]:
    """读取已有 LeRobot 数据集元信息。"""
    path = destination / "meta/info.json"
    if not path.exists():
        raise ValueError(f"existing output is not a LeRobot dataset: {destination}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(
    destination: Path,
    source: str,
    existing_count: int,
    action_horizon: int,
    action_stride: int,
) -> dict[str, Any]:
    """读取增量清单，并拒绝混用旧动作定义的数据集。"""
    path = destination / MANIFEST_PATH
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    elif existing_count:
        raise ValueError("现有数据集缺少 relative_action 清单, 请重新导出")
    else:
        manifest = {
            "format_version": 2,
            "action_representation": {
                "type": ACTION_REPRESENTATION,
                "storage": ACTION_STORAGE,
                "frame": "prediction_tcp",
                "rotation": "rotation_6d_rows",
                "horizon": action_horizon,
                "stride": action_stride,
            },
            "sources": {source: []},
        }
    representation = manifest.get("action_representation", {})
    expected = {
        "type": ACTION_REPRESENTATION,
        "storage": ACTION_STORAGE,
        "frame": "prediction_tcp",
        "rotation": "rotation_6d_rows",
        "horizon": action_horizon,
        "stride": action_stride,
    }
    if representation != expected:
        raise ValueError(f"目标数据集动作定义不兼容: expected={expected}, got={representation}")
    recorded = sum(len(set(names)) for names in manifest["sources"].values())
    if recorded != existing_count:
        raise ValueError(
            "incremental manifest and LeRobot metadata disagree; inspect the interrupted export"
        )
    return manifest


def write_manifest(destination: Path, manifest: dict[str, Any]) -> None:
    """原子写入源 episode 到目标 episode 的完成清单。"""
    path = destination / MANIFEST_PATH
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def features(camera: dict[str, Any], save_images: bool) -> dict[str, dict[str, object]]:
    """创建视频或图片模式的 LeRobot feature schema。"""
    visual_dtype = "image" if save_images else "video"
    visual = {
        "dtype": visual_dtype,
        "shape": (camera["height"], camera["width"], 3),
        "names": ["height", "width", "channels"],
    }
    return {
        GLOBAL_IMAGE: dict(visual),
        WRIST_IMAGE: dict(visual),
        "observation.state": {
            "dtype": "float32",
            "shape": (13,),
            "names": list(OBSERVATION_STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(ABSOLUTE_ACTION_NAMES),
        },
    }


def validate_existing_dataset(
    info: dict[str, Any],
    camera: dict[str, Any],
    fps: int,
    save_images: bool,
) -> None:
    """确保新增 episode 与目标数据集 schema 相容。"""
    expected_dtype = "image" if save_images else "video"
    expected_shape = [camera["height"], camera["width"], 3]
    for key in (GLOBAL_IMAGE, WRIST_IMAGE):
        feature = info["features"][key]
        if feature["dtype"] != expected_dtype or feature["shape"] != expected_shape:
            raise ValueError(f"existing dataset feature is incompatible: {key}")
    if info["fps"] != fps:
        raise ValueError(f"existing dataset fps={info['fps']} does not match source fps={fps}")
    if info["features"]["action"].get("names") != list(ABSOLUTE_ACTION_NAMES):
        raise ValueError("现有数据集不是 absolute EE storage, 不能增量写入 relative_action 数据")


def rgb_encoder(options: LeRobotExportConfig, info: dict[str, Any] | None = None) -> Any:
    """创建新编码器，增量写入时沿用已有视频编码参数。"""
    from lerobot.configs import RGBEncoderConfig

    if info is not None:
        return RGBEncoderConfig.from_video_info(info["features"][GLOBAL_IMAGE].get("info"))
    preset = None if options.video_codec == "auto" else options.video_preset
    return RGBEncoderConfig(
        vcodec=options.video_codec,
        crf=options.video_quality,
        preset=preset,
    )


def open_lerobot_dataset(
    repo_id: str,
    destination: Path,
    fps: int,
    schema: dict[str, dict[str, object]],
    options: LeRobotExportConfig,
    existing_info: dict[str, Any] | None,
) -> Any:
    """创建或恢复 LeRobot writer，并应用官方并行编码参数。"""

    encoder = None if options.save_images else rgb_encoder(options, existing_info)
    writer_options = {
        "image_writer_processes": options.image_writer_processes,
        "image_writer_threads": options.image_writer_threads,
        "rgb_encoder": encoder,
        "encoder_threads": options.encoder_threads,
        "streaming_encoding": options.streaming_encoding and not options.save_images,
        "encoder_queue_maxsize": options.encoder_queue_maxsize,
    }
    if existing_info is not None:
        return LeRobotDataset.resume(repo_id=repo_id, root=destination, **writer_options)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=destination,
        features=schema,
        use_videos=not options.save_images,
        **writer_options,
    )


def video_frame_pairs(path: Path, count: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """逐帧解码双相机视频，避免把整个 episode 放入内存。"""
    captures = [cv2.VideoCapture(str(path / f"{name}.mp4")) for name in ("global", "wrist")]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError(f"cannot open episode videos: {path}")
    try:
        for index in range(count):
            frames = [capture.read() for capture in captures]
            if not all(ok for ok, _ in frames):
                raise ValueError(f"video ended before action {index}: {path.name}")
            global_frame = cv2.cvtColor(frames[0][1], cv2.COLOR_BGR2RGB)
            wrist_frame = cv2.cvtColor(frames[1][1], cv2.COLOR_BGR2RGB)
            yield global_frame, wrist_frame
    finally:
        for capture in captures:
            capture.release()


def release_encoder_memory() -> None:
    """回收编码线程释放但 glibc 仍保留的原生堆内存。

    SVT-AV1 每条 episode 都会创建新的编码线程。线程退出后，其大块工作内存会
    留在 glibc arena 中，长时间转换时 RSS 会持续增长；Linux 下用
    ``malloc_trim`` 把已经空闲的页归还给系统。
    """
    collect()
    if platform.startswith("linux"):
        CDLL(None).malloc_trim(0)


def add_episode(
    dataset: Any,
    episode: EpisodeReader,
    options: LeRobotExportConfig,
    show_progress: bool = False,
) -> None:
    """批量计算数值特征，并流式送入 LeRobot writer。"""
    trajectory = episode.trajectory
    count = episode.action_count
    states = np.concatenate(
        [
            trajectory["joint_position"][:count],
            trajectory["tcp_position"][:count],
            trajectory["tcp_quaternion_wxyz"][:count],
        ],
        axis=1,
    ).astype(np.float32)
    action_source = episode.metadata["resolved_config"]["policy"]["action_source"]
    actions = build_absolute_actions(episode, source=action_source)
    frames = tqdm(
        video_frame_pairs(episode.path, count),
        total=count,
        desc=f"  {episode.path.name}",
        unit="frame",
        leave=False,
        disable=not show_progress,
    )
    for index, (global_frame, wrist_frame) in enumerate(frames):
        dataset.add_frame(
            {
                GLOBAL_IMAGE: global_frame,
                WRIST_IMAGE: wrist_frame,
                "observation.state": states[index],
                "action": actions[index],
                "task": episode.metadata["instruction"],
            }
        )
    try:
        dataset.save_episode(parallel_encoding=options.parallel_video_encoding)
    finally:
        release_encoder_memory()


def remove_video_image_tree(destination: Path) -> None:
    """清除视频模式中断时遗留的临时图片。"""
    root = destination / "images"
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    root.rmdir()


def update_episode_stats(
    statistics: Any,
    actions: list[list[float]],
    states: list[list[float]],
    action_horizon: int,
    action_stride: int,
) -> None:
    """把一个 episode 中的有效 relative action chunks 加入统计量。"""
    action_array = np.asarray(actions, dtype=np.float32)
    state_array = np.asarray(states, dtype=np.float32)
    sample_count = len(action_array) - (action_horizon - 1) * action_stride
    if sample_count < 1:
        return
    offsets = np.arange(action_horizon) * action_stride
    for start in range(0, sample_count, 1024):
        starts = np.arange(start, min(start + 1024, sample_count))
        chunks = action_array[starts[:, None] + offsets]
        relative = relative_ee_actions_from_absolute(
            chunks,
            state_array[starts, 6:9],
            state_array[starts, 9:13],
        )
        statistics.update(relative)


def recompute_relative_action_stats(
    destination: Path,
    action_horizon: int,
    action_stride: int,
) -> None:
    """按训练时的共享锚点定义重算 action normalization statistics。"""
    import pyarrow.dataset as arrow_dataset
    from lerobot.datasets.compute_stats import RunningQuantileStats

    files = sorted((destination / "data").rglob("*.parquet"))
    parquet = arrow_dataset.dataset([str(path) for path in files], format="parquet")
    scanner = parquet.scanner(columns=["action", "observation.state", "episode_index"])
    statistics = RunningQuantileStats()
    current_episode: int | None = None
    actions: list[list[float]] = []
    states: list[list[float]] = []
    for batch in scanner.to_batches():
        for action, state, episode in zip(
            batch.column("action").to_pylist(),
            batch.column("observation.state").to_pylist(),
            batch.column("episode_index").to_pylist(),
            strict=True,
        ):
            if current_episode is not None and episode != current_episode:
                update_episode_stats(statistics, actions, states, action_horizon, action_stride)
                actions, states = [], []
            current_episode = episode
            actions.append(action)
            states.append(state)
    if actions:
        update_episode_stats(statistics, actions, states, action_horizon, action_stride)
    action_stats = {
        name: np.asarray(value).tolist() for name, value in statistics.get_statistics().items()
    }
    path = destination / "meta/stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["action"] = action_stats
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def export_lerobot(
    dataset_root: str | Path,
    output: str | Path,
    repo_id: str,
    options: LeRobotExportConfig | None = None,
    action_horizon: int = 30,
    action_stride: int = 1,
    *,
    recompute_stats: bool = True,
    show_progress: bool = True,
) -> ExportReport:
    """导出绝对目标，并写入 relative action 训练所需的统计与契约。"""
    options = options or LeRobotExportConfig()
    root = Path(dataset_root)
    destination = Path(output)
    if destination.exists() and not options.incremental:
        raise FileExistsError(f"output already exists; enable incremental mode: {destination}")

    selected = valid_episode_paths(root, options.start_episode, options.end_episode)
    if not selected:
        raise ValueError(f"no valid episodes found in selected range: {root}")
    first = EpisodeReader(selected[0])
    camera = first.metadata["resolved_config"]["camera"]
    fps = int(first.metadata["resolved_config"]["timing"]["policy_hz"])
    source = source_id(root)
    existing_info = read_dataset_info(destination) if destination.exists() else None
    existing_count = int(existing_info["total_episodes"]) if existing_info else 0
    manifest = load_manifest(
        destination,
        source,
        existing_count,
        action_horizon,
        action_stride,
    )
    exported_names = set(manifest["sources"].get(source, []))
    pending = [path for path in selected if path.name not in exported_names]
    if not pending:
        write_manifest(destination, manifest)
        if not options.save_images:
            remove_video_image_tree(destination)
        return ExportReport(
            str(destination),
            len(selected),
            0,
            len(selected),
            episode_number(selected[0]),
            episode_number(selected[-1]),
        )

    schema = features(camera, options.save_images)
    if existing_info:
        validate_existing_dataset(existing_info, camera, fps, options.save_images)
    dataset = open_lerobot_dataset(repo_id, destination, fps, schema, options, existing_info)
    saved: list[str] = []
    try:
        for path in tqdm(
            pending,
            desc=f"Exporting {root.name}",
            unit="episode",
            disable=not show_progress,
        ):
            add_episode(dataset, EpisodeReader(path), options, show_progress)
            saved.append(path.name)
    finally:
        dataset.finalize()
        if saved:
            manifest["sources"].setdefault(source, []).extend(saved)
            if recompute_stats:
                recompute_relative_action_stats(destination, action_horizon, action_stride)
            write_manifest(destination, manifest)
        if not options.save_images:
            remove_video_image_tree(destination)

    return ExportReport(
        str(destination),
        len(selected),
        len(saved),
        len(selected) - len(saved),
        episode_number(selected[0]),
        episode_number(selected[-1]),
    )


def export_sequentially(
    roots: list[Path],
    destination: Path,
    repo_id: str,
    options: LeRobotExportConfig,
    action_horizon: int,
    action_stride: int,
) -> BatchExportReport:
    """使用单 writer 安全创建或增量写入多个源。

    Args:
        roots: 按写入顺序排列的原始数据集。
        destination: 新建或已有的 LeRobot 目标。
        repo_id: 目标数据集仓库标识。
        options: 选择、增量和编码配置。
        action_horizon: 训练动作块长度。
        action_stride: 动作块相邻目标的帧间隔。

    Returns:
        所有源的汇总转换统计。
    """
    reports = []
    try:
        for index, root in enumerate(roots):
            step_options = replace(
                options,
                incremental=options.incremental or index > 0,
            )
            reports.append(
                export_lerobot(
                    root,
                    destination,
                    repo_id,
                    step_options,
                    action_horizon,
                    action_stride,
                    recompute_stats=False,
                )
            )
    finally:
        if (destination / "meta/stats.json").exists():
            recompute_relative_action_stats(destination, action_horizon, action_stride)
    return BatchExportReport(
        str(destination),
        [str(root) for root in roots],
        sum(report.selected_episodes for report in reports),
        sum(report.exported_episodes for report in reports),
        sum(report.skipped_episodes for report in reports),
    )


def export_lerobot_many(
    dataset_roots: Sequence[str | Path],
    output: str | Path,
    repo_id: str,
    options: LeRobotExportConfig | None = None,
    action_horizon: int = 30,
    action_stride: int = 1,
) -> BatchExportReport:
    """用单 writer 转换一个或多个原始数据集。

    episode 按顺序写入，避免多个数据集 writer 和视频编码器同时占用内存；
    双相机视频仍由 LeRobot 在每个 episode 内并行编码。

    Args:
        dataset_roots: 一个或多个原始数据集根目录。
        output: 目标 LeRobot 数据集路径。
        repo_id: 目标数据集仓库标识。
        options: 选择、增量和编码配置。
        action_horizon: 训练动作块长度。
        action_stride: 动作块相邻目标的帧间隔。

    Returns:
        多个原始数据源的汇总转换统计。
    """
    options = options or LeRobotExportConfig()
    roots = list(dict.fromkeys(Path(root) for root in dataset_roots))
    if not roots:
        raise ValueError("at least one raw dataset is required")
    destination = Path(output)
    report = export_sequentially(
        roots,
        destination,
        repo_id,
        options,
        action_horizon,
        action_stride,
    )
    return push_export_to_hub(report, destination, repo_id, options)


def push_export_to_hub(
    report: BatchExportReport,
    destination: Path,
    repo_id: str,
    options: LeRobotExportConfig,
) -> BatchExportReport:
    """将已 finalize 的 LeRobot 数据集上传到 Dataset Hub。

    Args:
        report: 本次转换统计。
        destination: 完整的本地 LeRobot 数据集。
        repo_id: Hugging Face Dataset 仓库标识。
        options: Hub 开关和可见性配置。

    Returns:
        未启用上传时返回原报告，否则附加 Dataset Hub 地址。
    """
    if not options.push_to_hub:
        return report
    hub_url = upload_lerobot_dataset(
        destination,
        repo_id,
        private=options.hub_private,
        attempts=options.hub_upload_attempts,
        retry_wait_s=options.hub_retry_wait_s,
    )
    return replace(report, hub_url=hub_url)


def upload_lerobot_dataset(
    root: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    attempts: int = 5,
    retry_wait_s: float = 30.0,
) -> str:
    """上传已经完成的本地 LeRobot 数据集，并重试临时网络错误。

    Hugging Face 的文件上传会复用本地缓存；命令中断后可再次调用本函数，
    不会重新转换原始 episode。

    Args:
        root: 已完成 `finalize()` 的本地 LeRobot 数据集目录。
        repo_id: Hugging Face Dataset 仓库标识。
        private: 新建仓库时是否设为私有。
        attempts: 包含首次请求在内的总尝试次数。
        retry_wait_s: 两次尝试之间的等待秒数。

    Returns:
        上传目标的 Dataset Hub 地址。
    """
    if attempts < 1:
        raise ValueError("upload attempts must be positive")
    dataset = LeRobotDataset(repo_id=repo_id, root=Path(root))
    for attempt in range(1, attempts + 1):
        try:
            dataset.push_to_hub(
                tags=["lerobot", "robotics", "welding"],
                private=private,
            )
            break
        except httpx.TransportError:
            if attempt == attempts:
                raise
            logger.warning(
                "Hub connection failed (%d/%d); retrying in %.0f seconds",
                attempt,
                attempts,
                retry_wait_s,
            )
            sleep(retry_wait_s)
    return f"https://huggingface.co/datasets/{repo_id}"


__all__ = [
    "OBSERVATION_STATE_NAMES",
    "BatchExportReport",
    "ExportReport",
    "export_lerobot",
    "export_lerobot_many",
    "push_export_to_hub",
    "upload_lerobot_dataset",
    "valid_episode_paths",
]
