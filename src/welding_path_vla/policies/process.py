"""策略训练子进程的日志执行辅助。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_logged_command(command: list[str], output_dir: Path) -> Path:
    """执行命令，并把 stdout/stderr 同时写入同级日志。"""
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("cannot capture policy training output")
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
            stream.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return log_path


__all__ = ["run_logged_command"]
