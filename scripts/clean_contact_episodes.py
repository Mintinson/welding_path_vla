#!/usr/bin/env python3
"""审计 v5 全部保存状态/动作目标，用 LeRobot 官方 API 创建排除接触的 v6。"""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from tqdm import tqdm

from welding_path_vla.dataset.contact_audit import (
    audit_episode,
    filtered_manifest,
    stored_episodes,
)
from welding_path_vla.dataset.phase_index import (
    load_phase_index,
    phase_index_path,
    write_phase_index,
)

MANIFEST = Path("meta/welding_path_vla_export.json")


def write_json(path: Path, value: object) -> None:
    """写出可阅读的审计结果或数据集附属清单。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_jobs(source: Path, raw_root: Path, metadata, manifest: dict):
    """核对清单、任务、帧顺序后生成完整逐 episode 审计输入。"""
    entries = []
    for source_id, names in manifest["sources"].items():
        task = source_id.split("|", 1)[0]
        for name in names:
            index = len(entries)
            episode = metadata.episodes[index]
            entries.append(
                {
                    "old_episode_index": index,
                    "task": task,
                    "raw_episode": name,
                    "raw_path": str((raw_root / task / "episodes" / name).resolve()),
                    "instruction": episode["tasks"][0],
                    "frames": int(episode["length"]),
                    "from_index": int(episode["dataset_from_index"]),
                    "to_index": int(episode["dataset_to_index"]),
                }
            )
    if len(entries) != metadata.total_episodes:
        raise ValueError("source manifest and episode count differ")
    for entry, (index, states, actions) in zip(entries, stored_episodes(source), strict=True):
        if index != entry["old_episode_index"]:
            raise ValueError("source manifest and parquet episode order differ")
        yield entry, states, actions


def summarize(results: list[dict], source: Path, destination: Path) -> dict:
    """分别汇总实际记录接触、危险标签及两者交集，不把标签接触当成实际接触。"""
    groups = {}
    for task in ["all", *sorted({result["task"] for result in results})]:
        selected = [result for result in results if task == "all" or result["task"] == task]
        actual = {
            result["old_episode_index"]
            for result in selected
            if result["executed"]["contact_frames"]
        }
        labels = {
            result["old_episode_index"] for result in selected if result["labels"]["contact_frames"]
        }
        groups[task] = {
            "episodes": len(selected),
            "frames": sum(result["frames"] for result in selected),
            "executed_contact_episodes": len(actual),
            "dangerous_label_episodes": len(labels),
            "both": len(actual & labels),
            "label_only": len(labels - actual),
            "removed": len(actual | labels),
            "kept": len(selected) - len(actual | labels),
            "kept_frames": sum(result["frames"] for result in selected if not result["remove"]),
        }
        for kind in ("executed", "labels"):
            phase_counts = Counter()
            for result in selected:
                phase_counts.update(result[kind]["phase_contacts"])
            groups[task][kind] = {
                "checked_frames": sum(result[kind]["checked_frames"] for result in selected),
                "contact_frames": sum(result[kind]["contact_frames"] for result in selected),
                "max_depth_m": max(result[kind]["max_depth_m"] for result in selected),
                "phase_contacts": dict(phase_counts),
                "max_fk_position_error_m": max(
                    result[kind]["max_fk_position_error_m"] for result in selected
                ),
            }
    return {
        "source": str(source),
        "destination": str(destination),
        "groups": groups,
        "criterion": "unfiltered robot/workpiece MuJoCo contacts with dist <= 0",
        "scope": "all saved joint states including terminal state, and all safe_command labels",
        "limitation": "not a replay of unsaved physics substeps; no continuous collision guarantee",
    }


def write_report(report: Path, summary: dict, results: list[dict]) -> None:
    """保存删除依据、旧新索引映射与按任务统计。"""
    write_json(report / "summary.json", summary)
    with (report / "episode_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "old_episode_index",
                "new_episode_index",
                "task",
                "raw_episode",
                "executed_contact_frames",
                "dangerous_label_frames",
                "max_depth_m",
            ]
        )
        new_index = 0
        for result in results:
            writer.writerow(
                [
                    result["old_episode_index"],
                    "" if result["remove"] else new_index,
                    result["task"],
                    result["raw_episode"],
                    result["executed"]["contact_frames"],
                    result["labels"]["contact_frames"],
                    max(result[kind]["max_depth_m"] for kind in ("executed", "labels")),
                ]
            )
            new_index += not result["remove"]
    rows = [
        "|任务|原 episodes|记录接触|危险标签|仅标签|删除并集|保留|保留帧|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, group in summary["groups"].items():
        rows.append(
            f"|{task}|{group['episodes']}|{group['executed_contact_episodes']}|"
            f"{group['dangerous_label_episodes']}|{group['label_only']}|{group['removed']}|"
            f"{group['kept']}|{group['kept_frames']}|"
        )
    text = "# v5 接触审计与 v6 清理\n\n" + "\n".join(rows)
    total = summary["groups"]["all"]
    old_flagged = sum(result["recorded_collision_frames"] > 0 for result in results)
    actual, labels = total["executed"], total["labels"]
    text += (
        f"\n\n删除 {total['removed'] / total['episodes']:.2%} 的 episode, "
        f"删除 {total['frames'] - total['kept_frames']:,} 帧。"
        f"两类问题交集 {total['both']} 条, 仅执行状态接触 "
        f"{total['executed_contact_episodes'] - total['both']} 条, "
        f"仅标签危险 {total['label_only']} 条。\n\n"
        f"原 collision 标志为真的 episode 只有 {old_flagged} 条, "
        "不能用它来执行本次严格筛选。"
        f"重算执行状态命中 {actual['contact_frames']} 帧, "
        f"最大穿入 {actual['max_depth_m'] * 1000:.6f} mm; "
        f"标签命中 {labels['contact_frames']} 帧, "
        f"最大穿入 {labels['max_depth_m'] * 1000:.6f} mm。\n\n"
        "|类型|approach 接触帧|track 接触帧|retreat 接触帧|\n"
        "|---|---:|---:|---:|\n"
    )
    for name, statistics in (("记录执行状态", actual), ("动作目标标签", labels)):
        phases = statistics["phase_contacts"]
        text += f"|{name}|{phases.get('approach', 0)}|{phases.get('track', 0)}|"
        text += f"{phases.get('retreat', 0)}|\n"
    text += (
        "\n\n## 判据与范围\n\n"
        "逐帧核对 v5 的 state/action 与 raw 导出结果完全相同；使用历史物理配置、"
        "工件位姿和完整机器人碰撞几何，重算 MuJoCo 原始接触。任意机器人/焊枪与工件"
        "的 dist <= 0 接触均剔除整条 episode，不沿用旧 collision 深度/力过滤。"
        "执行状态包含 raw 末帧；标签使用 joint_position_command，并核验 FK 实现标签位置。"
        "接触帧数以关节配置为单位，不是接触点数。动作目标接触不代表该时刻实际执行接触。\n\n"
        "未保存的物理子步没有历史轨迹，不能据此保证连续运动完全零接触；原碰撞几何"
        "也不是高精度 CAD/真机几何。清理数据不能替代部署时的严格碰撞检测。\n\n"
        "## 文件与可追溯性\n\n"
        "episodes.jsonl 包含每条 episode 的命中帧索引、几何对、最大穿入和首次接触；"
        "episode_mapping.csv 给出 v5→v6 映射及删除原因。summary.json 按任务、阶段统计。"
        "通过官方 delete_episodes 创建新目录，v5 不修改；同步过滤源清单和 phase sidecar，"
        "重新计算共享锚点 relative-action 统计。没有上传 Hugging Face。\n\n"
        "官方文档：[delete episodes](https://huggingface.co/docs/lerobot/main/en/"
        "using_dataset_tools#delete-episodes)。\n"
    )
    text += (
        "\n## 视频、验证与训练使用\n\n"
        "默认使用官方重编码路径; 若启用 --lossless-video: "
        "仅在调用期间替换官方视频裁剪回调为 packet 原样重封装, "
        "不修改安装库。每个裁剪区间必须从关键帧开始、无帧重排, "
        "并逐个输出文件核对全部保留 packet 的 SHA256。"
        "不满足契约会报错, 不静默跳过。"
        "元数据读取采用缓存, 避免每个 episode 重复读取整个 parquet, "
        "返回的记录和官方函数一致。\n\n"
        "完成标志为 v6_validation.json。验证包括 v6 全部 state/action 与对应 raw "
        "完全相同、连续索引与总帧数、全部视频长度/帧率和首末帧可解码, "
        "以及 v5 文件列表/大小/修改时间与关键 JSON 摘要未变化。"
        "真实 AV1 样例的删除边界图像另已逐像素核对相同。\n\n"
        f"执行状态 FK 与 raw 缓存 TCP 位置的最大差异为 "
        f"{actual['max_fk_position_error_m'] * 1000:.6f} mm, "
        "因此检测以记录的关节姿态重建完整机器人为准, 不能将缓存 TCP 点"
        "直接当成全部碰撞几何。标签 FK 位置误差为零, "
        "采集代码中 safe_command 位姿由同一 joint_command 的 FK 生成。\n\n"
        "训练配置未自动切换。要使用新数据集, 设置:\n\n"
        "```yaml\ntraining:\n"
        "  dataset_root: datasets/lerobot/weldpath_lerobot_relative_v6\n"
        "  dataset_repo_id: mintinson/weldpath_relative_v6\n```\n\n"
        "v6 重算了 normalization stats, 不应把旧 checkpoint 的 processor 统计"
        "当成新统计直接混用。比较 v5/v6 时应固定训练预算、种子和部署条件, "
        "优先分别从头训练。删除的是整条 episode, 不会产生片段拼接的新动作跳变。\n"
    )
    validation_path = report / "v6_validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        text += (
            "\n## 完成结果\n\n"
            f"v6 已生成并验证: {validation['episodes']:,} episodes, "
            f"{validation['frames']:,} frames, "
            f"{validation['video_files_checked']} 个视频文件。"
            f"视频编辑模式: `{validation['video_editing']}`。"
            "数值逐帧一致, v5 保持不变。详细验证见 v6_validation.json。\n"
        )
    (report / "report.md").write_text(text, encoding="utf-8")


def create_cleaned_dataset(
    source: Path,
    destination: Path,
    report: Path,
    results: list[dict],
    manifest: dict,
    repo_id: str,
    lossless_video: bool = False,
) -> None:
    """复用官方删除 API，并修复项目 sidecar 与 relative-action 统计契约。"""
    import pandas as pd
    from lerobot.datasets import LeRobotDataset, delete_episodes
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

    from welding_path_vla.dataset.export_lerobot import recompute_relative_action_stats
    from welding_path_vla.dataset.video_filter import remux_video_ranges

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    removed = {result["old_episode_index"] for result in results if result["remove"]}
    snapshot = source_snapshot(source)
    dataset: Any = LeRobotDataset("local/weldpath_relative_v5", root=source)
    # 官方工具会对每个视频分块反复查全部 episode；内存列表保持同样的记录与顺序。
    dataset.meta.episodes = list(dataset.meta.episodes)
    episode_tables = {}

    def episode_statistics(source_dataset, index: int) -> dict:
        """每个元数据 parquet 只读取一次，返回与官方加载函数相同的原始统计。"""
        episode = source_dataset.meta.episodes[index]
        path = source_dataset.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=episode["meta/episodes/chunk_index"],
            file_index=episode["meta/episodes/file_index"],
        )
        if path not in episode_tables:
            episode_tables[path] = pd.read_parquet(path).set_index("episode_index", drop=False)
        return episode_tables[path].loc[index].to_dict()

    # 仅在本次调用替换视频剪辑回调；不修改安装库、数据结构或官方重排/统计逻辑。
    video_context = (
        patch(
            "lerobot.datasets.dataset_tools._keep_episodes_from_video_with_av", remux_video_ranges
        )
        if lossless_video
        else nullcontext()
    )
    with (
        video_context,
        patch("lerobot.datasets.dataset_tools._load_episode_with_stats", episode_statistics),
    ):
        cleaned = delete_episodes(dataset, sorted(removed), output_dir=destination, repo_id=repo_id)
    write_json(destination / MANIFEST, filtered_manifest(manifest, removed))
    phase = load_phase_index(phase_index_path(source))
    kept = [result for result in results if not result["remove"]]
    filtered_phase = np.concatenate(
        [phase[result["from_index"] : result["to_index"]] for result in kept]
    )
    if len(filtered_phase) != cleaned.meta.total_frames or len(kept) != cleaned.meta.total_episodes:
        raise ValueError("cleaned metadata and retained audit frames differ")
    write_phase_index(phase_index_path(destination), filtered_phase)
    representation = manifest["action_representation"]
    logging.info("Recomputing relative-action statistics")
    recompute_relative_action_stats(
        destination, representation["horizon"], representation["stride"]
    )
    # 重新核对 v6 所有数值帧，避免仅计数相同而重排/截断错误。
    for new_index, (result, (index, states, actions)) in enumerate(
        zip(kept, stored_episodes(destination), strict=True)
    ):
        if index != new_index or not validate_episode_frames(result, states, actions):
            raise ValueError(f"cleaned frame mismatch: {result['raw_path']}")
    video_count = validate_video_files(destination, cleaned.meta)
    if snapshot != source_snapshot(source):
        raise ValueError("source dataset changed during cleanup")
    provenance = {
        "audit_report": str(report),
        "source": str(source),
        "removed_episode_indices": sorted(removed),
        "repo_id": repo_id,
        "episodes": cleaned.meta.total_episodes,
        "frames": cleaned.meta.total_frames,
        "numerical_frames_verified": True,
        "source_unchanged": True,
        "video_files_checked": video_count,
        "video_editing": "payload-verified lossless remux"
        if lossless_video
        else "official reencoding",
    }
    write_json(destination / "meta/contact_audit.json", provenance)
    write_json(report / "v6_validation.json", provenance)


def validate_video_files(destination: Path, metadata) -> int:
    """核对全部视频长度、帧率，并解码首末帧验证索引覆盖范围。"""
    import av

    expected = {}
    for episode in metadata.episodes:
        for key in metadata.video_keys:
            path = destination / metadata.video_path.format(
                video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"],
            )
            end = round(episode[f"videos/{key}/to_timestamp"] * metadata.fps)
            expected[path] = max(expected.get(path, 0), end)
    for path, frames in tqdm(expected.items(), desc="Validate videos"):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            time_base = stream.time_base
            if time_base is None:
                raise ValueError(f"video has no time base: {path}")
            if stream.frames != frames or stream.average_rate != metadata.fps:
                raise ValueError(f"video length/rate mismatch: {path}")
            first = next(container.decode(stream))
            if round(float(first.pts * time_base) * metadata.fps) != 0:
                raise ValueError(f"video does not start at frame 0: {path}")
            container.seek(round((frames - 1) / metadata.fps / time_base), stream=stream)
            if not any(
                round(float(frame.pts * time_base) * metadata.fps) == frames - 1
                for frame in container.decode(stream)
            ):
                raise ValueError(f"video last frame cannot be decoded: {path}")
    return len(expected)


def source_snapshot(source: Path) -> dict:
    """记录源文件大小/修改时间及关键元数据摘要，核实清理没有修改 v5。"""
    files = {
        str(path.relative_to(source)): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in source.rglob("*")
        if path.is_file()
    }
    metadata = {
        str(path.relative_to(source)): sha256(path.read_bytes()).hexdigest()
        for path in (source / "meta").glob("*.json")
    }
    return {"files": files, "metadata_sha256": metadata}


def validate_episode_frames(entry: dict, states: np.ndarray, actions: np.ndarray) -> bool:
    """复用导出公式核对清理后的数值帧，不重复计算碰撞。"""
    from welding_path_vla.dataset.actions import build_absolute_actions
    from welding_path_vla.dataset.raw_schema import EpisodeReader

    episode = EpisodeReader(entry["raw_path"])
    try:
        expected = np.concatenate(
            [
                episode.trajectory[name][: episode.action_count]
                for name in ("joint_position", "tcp_position", "tcp_quaternion_wxyz")
            ],
            axis=1,
        )
        return np.array_equal(states, expected.astype(np.float32)) and np.array_equal(
            actions, build_absolute_actions(episode)
        )
    finally:
        episode.trajectory.close()


def main() -> None:
    """支持完整审计、仅审计和复用已完成审计，始终保持 v5 不变。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-id", default="mintinson/weldpath_relative_v6")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-audit", action="store_true")
    parser.add_argument("--lossless-video", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    source, destination, report = (
        args.source.resolve(),
        args.destination.resolve(),
        args.report.resolve(),
    )
    if destination.exists() or source == destination:
        raise FileExistsError(f"destination must be new: {destination}")
    report.mkdir(parents=True, exist_ok=True)
    metadata = LeRobotDatasetMetadata("local/weldpath_relative_v5", root=source)
    manifest = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    results_path = report / "episodes.jsonl"
    if args.reuse_audit:
        previous = json.loads((report / "summary.json").read_text())
        if previous["source"] != str(source) or (
            "source_snapshot" in previous and previous["source_snapshot"] != source_snapshot(source)
        ):
            raise ValueError("reused audit belongs to a different or changed source dataset")
        results = [json.loads(line) for line in results_path.read_text().splitlines()]
    else:
        if results_path.exists():
            raise FileExistsError(f"audit already exists; use --reuse-audit: {results_path}")
        jobs = audit_jobs(source, args.raw_root, metadata, manifest)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=context) as executor:
            results = []
            with results_path.open("w", encoding="utf-8") as handle:
                for result in tqdm(
                    executor.map(audit_episode, jobs, chunksize=8),
                    total=metadata.total_episodes,
                    desc="Audit episodes",
                ):
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    results.append(result)
    if [result["old_episode_index"] for result in results] != list(range(metadata.total_episodes)):
        raise ValueError("audit does not cover every source episode in order")
    summary = summarize(results, source, destination)
    summary["source_snapshot"] = source_snapshot(source)
    write_report(report, summary, results)
    logging.info("Audit summary: %s", summary["groups"]["all"])
    if not args.audit_only:
        create_cleaned_dataset(
            source, destination, report, results, manifest, args.repo_id, args.lossless_video
        )
        write_report(report, summary, results)
        logging.info("Validated cleaned dataset: %s", destination)


if __name__ == "__main__":
    main()
