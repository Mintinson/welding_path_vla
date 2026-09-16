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

## Fix Round

### 覆盖测试

- 覆盖 action/frame（含 terminal）数量不匹配、RGB 错误 dtype/shape。
- 覆盖 sidecar 错误 shape、非 `float64` dtype、非有限值、不完整字段、错误 action count/camera set/algorithm。
- 覆盖原子写入后临时文件不残留且最终文件存在。
- 读写端均强制 exact fields：`mean`、`min`、`max`，并校验 `(3,)`、`float64`、finite。

### 命令与输出

```text
PYTHONPATH=/tmp/welding_path_vla_raw_video_stats/src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py -q
.........                                                                [100%]
9 passed in 0.89s

PYTHONPATH=/tmp/welding_path_vla_raw_video_stats/src /root/micromamba/envs/welding_vla/bin/python -m compileall -q src/welding_path_vla/dataset/video_stats.py src/welding_path_vla/dataset/raw_schema.py
git diff --check
```

### 跨任务依赖

Task 1 将 `RAW_DATASET_FORMAT` 默认设为 v2，但 recorder 尚未写入 v2 sidecar；因此 Task 2 集成前新 episode 可能处于 metadata v2、sidecar 缺失的中间状态。Task 1 不伪造 recorder 集成或回退默认格式；这是 Task 2 的硬依赖，Task 2 必须在 episode 完成路径原子发布 sidecar 后再发布 v2 metadata。

本轮修复提交：`fix: harden raw video statistics contract`。

## Fix Round 2

本轮仅补充测试，未修改生产代码。新增覆盖：

- RGB 三维帧使用 `float32` 时拒绝，而非仅覆盖二维形状错误。
- action/state 数量不匹配时 accumulator 拒绝完成；另保留视频帧/action count mismatch 覆盖。
- 通过记录 `os.replace` 的源/目标路径，验证 sidecar 更新确实使用临时文件原子替换，而非直接写最终文件。

精确命令及输出：

```text
PYTHONPATH=/tmp/welding_path_vla_raw_video_stats/src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py -q
............                                                             [100%]
12 passed in 0.45s
```

本轮提交：`test: cover raw video statistics fix round 2`。
