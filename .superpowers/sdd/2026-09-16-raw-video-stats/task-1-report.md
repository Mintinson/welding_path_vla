# Task 1 实施报告

## 实现总结

- 新增 `ActionAlignedVideoStats`，采用一帧延迟累计，统计动作帧并排除 terminal frame。
- 新增原始 RGB `uint8` 输入验证、逐通道 mean/min/max 统计，保留 0--255 数值。
- 在 `raw_schema.py` 集中定义 v1/v2 格式常量、默认 v2 格式和 `video_stats.json` sidecar 名称。
- 新增 sidecar 原子写入（临时文件 + `os.replace`）和读取完整性校验：格式、算法、action count、相机集合、字段、形状和有限值。

## 文件变更

- `src/welding_path_vla/dataset/video_stats.py`
- `src/welding_path_vla/dataset/raw_schema.py`
- `tests/test_video_stats.py`

## TDD 记录

RED 命令：

```text
/root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py::test_action_aligned_stats_exclude_terminal_frame -q
```

相关输出：`ImportError: cannot import name 'ActionAlignedVideoStats'`（缺失待实现接口）。

GREEN 命令：

```text
PYTHONPATH=/tmp/welding_path_vla_raw_video_stats/src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py -q
```

相关输出：`4 passed in 0.20s`。

## 测试与审查

- `python -m compileall -q`：通过。
- `git diff --check`：通过。
- focused pytest：通过，覆盖终止帧、RGB 输入、round-trip、错误算法、缺失相机和非有限值。
- 自审确认改动仅限任务 brief 指定的三个源码/测试文件；已提交 commit `414a9d8`。

## Concern

仓库环境未提供独立 `ruff` 可执行文件，因此无法运行 ruff；其余可用验证均通过。editable 安装指向主工作区，测试需显式设置当前 worktree 的 `PYTHONPATH`。
