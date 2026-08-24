import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation import collector as collection


def test_collection_targets_valid_episodes_and_keeps_failures(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig()
    config.collection.dataset_root = str(tmp_path)
    config.collection.max_attempt_multiplier = 3
    config.collection.headless = False
    outcomes = iter([False, True, True])

    def fake_collect(config: AppConfig, episode_index: int, seed: int) -> Path:
        path = Path(config.collection.dataset_root) / "episodes" / f"episode_{episode_index:06d}"
        path.mkdir(parents=True)
        valid = next(outcomes)
        status = "valid_success" if valid else "invalid_simulation"
        metadata = {"quality": {"valid": valid, "status": status}, "seed": seed}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    monkeypatch.setattr(collection, "collect_episode", fake_collect)
    paths = collection.collect_dataset(config, episodes=2)
    summary = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))

    assert len(paths) == 3
    assert summary["valid_episodes"] == 2
    assert summary["attempted_episodes"] == 3
    assert summary["status"] == {"invalid_simulation": 1, "valid_success": 2}
    assert len(list((tmp_path / "episodes").iterdir())) == 3


def test_parallel_collection_stops_at_exact_valid_target(tmp_path: Path, monkeypatch) -> None:
    """并发调度应补充失败任务，且不会多采集有效 episode。"""
    config = AppConfig()
    config.collection.dataset_root = str(tmp_path)
    config.collection.workers = 2
    outcomes = iter([False, True, True])

    def fake_collect(config: AppConfig, episode_index: int, seed: int) -> Path:
        path = Path(config.collection.dataset_root) / "episodes" / f"episode_{episode_index:06d}"
        path.mkdir(parents=True)
        valid = next(outcomes)
        status = "valid_success" if valid else "invalid_simulation"
        metadata = {"quality": {"valid": valid, "status": status}, "seed": seed}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def thread_executor(max_workers: int, mp_context: object) -> ThreadPoolExecutor:
        """用线程替身稳定测试父进程的并发调度逻辑。"""
        del mp_context
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(collection, "collect_episode", fake_collect)
    monkeypatch.setattr(collection, "ProcessPoolExecutor", thread_executor)
    paths = collection.collect_dataset(config, episodes=2)
    summary = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))

    assert len(paths) == 3
    assert summary["last_request_collected_valid_episodes"] == 2
    assert summary["last_request_attempts"] == 3
    assert summary["collection_workers"] == 2


def test_collection_retries_expected_sampling_errors(tmp_path: Path, monkeypatch) -> None:
    """单个随机场景不可行时应继续下一 seed，而不是中止长时间采集。"""
    config = AppConfig()
    config.collection.dataset_root = str(tmp_path)
    calls = 0

    def fake_collect(config: AppConfig, episode_index: int, seed: int) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("unreachable randomized task")
        path = Path(config.collection.dataset_root) / "episodes" / f"episode_{episode_index:06d}"
        path.mkdir(parents=True)
        metadata = {"quality": {"valid": True, "status": "valid_success"}, "seed": seed}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    monkeypatch.setattr(collection, "collect_episode", fake_collect)
    paths = collection.collect_dataset(config, episodes=1)
    summary = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))

    assert len(paths) == 1
    assert summary["last_request_attempts"] == 2
    assert summary["last_request_collection_errors"] == 1
    assert summary["status"] == {"valid_success": 1, "collection_error": 1}


def test_interrupted_collection_writes_summary_and_cleans_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Ctrl+C 应保存现有汇总，并清除未完成的 episode。"""
    config = AppConfig()
    config.collection.dataset_root = str(tmp_path)
    config.collection.headless = False
    calls = 0

    def fake_collect(config: AppConfig, episode_index: int, seed: int) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            incomplete = tmp_path / ".incomplete" / f"episode_{episode_index:06d}"
            incomplete.mkdir(parents=True)
            (incomplete / "partial.mp4").write_bytes(b"partial")
            raise KeyboardInterrupt
        path = tmp_path / "episodes" / f"episode_{episode_index:06d}"
        path.mkdir(parents=True)
        metadata = {"quality": {"valid": True, "status": "valid_success"}, "seed": seed}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    monkeypatch.setattr(collection, "collect_episode", fake_collect)
    with pytest.raises(KeyboardInterrupt):
        collection.collect_dataset(config, episodes=2)

    summary = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))
    assert summary["last_request_collected_valid_episodes"] == 1
    assert summary["last_request_attempts"] == 1
    assert summary["last_request_interrupted"] is True
    assert summary["next_episode_index"] == 1
    assert summary["status"] == {"valid_success": 1}
    assert not (tmp_path / ".incomplete").exists()
