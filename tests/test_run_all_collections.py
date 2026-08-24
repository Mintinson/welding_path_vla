import importlib.util
import json
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_all_collections", Path("scripts/run_all_collections.py")
)
assert spec and spec.loader
run_all_collections = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_all_collections
spec.loader.exec_module(run_all_collections)


def test_batch_collection_uses_composed_task_configs() -> None:
    """批处理必须使用包含 base.yaml 的公开配置入口。"""
    assert [task.config for task in run_all_collections.TASKS] == [
        "configs/default.yaml",
        "configs/curve_plate.yaml",
        "configs/trihedral_vertical.yaml",
        "configs/trihedral_horizontal.yaml",
        "configs/pipe_bottom.yaml",
        "configs/pipe_top.yaml",
    ]


def test_interrupt_collection_waits_for_summary_and_releases_process_group(tmp_path: Path) -> None:
    """Ctrl+C 应允许子进程保存总结，并确保整个采集进程组退出。"""
    summary = tmp_path / "dataset.json"
    ready = tmp_path / "ready"
    child = tmp_path / "child.py"
    child.write_text(
        "\n".join(
            [
                "import json, signal, subprocess, sys, time",
                "summary, ready = sys.argv[1:]",
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "def stop(signum, frame):",
                "    open(summary, 'w').write(json.dumps({'last_request_interrupted': True}))",
                "    raise KeyboardInterrupt",
                "signal.signal(signal.SIGINT, stop)",
                "open(ready, 'w').write('ready')",
                "time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )
    process = run_all_collections.start_collection(
        [sys.executable, str(child), str(summary), str(ready)]
    )
    deadline = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    run_all_collections.interrupt_collection(process, timeout_s=3)

    assert process.poll() is not None
    assert json.loads(summary.read_text(encoding="utf-8"))["last_request_interrupted"] is True
    assert run_all_collections.process_group_exists(process.pid) is False
