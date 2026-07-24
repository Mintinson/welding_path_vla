import json
from pathlib import Path

from welding_path_vla.core.config import AppConfig
from welding_path_vla.simulation import collector as collection


def test_collection_targets_valid_episodes_and_keeps_failures(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig()
    config.collection.dataset_root = str(tmp_path)
    config.collection.max_attempt_multiplier = 3
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
