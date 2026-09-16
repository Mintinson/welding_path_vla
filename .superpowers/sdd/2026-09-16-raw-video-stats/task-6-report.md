# Task 6：原始视频统计兼容性文档与验证报告

## 结果

- README 新增“原始视频统计兼容性”，说明 v1/v2 行为、H.264 passthrough 和旧数据回填。
- 文档提交：`a9ed0af docs: explain raw video statistics compatibility`。
- 本次只修改 `README.md`；报告位于任务工作目录，未纳入文档提交。
- 合成 v1/v2 passthrough 烟测均通过；完整指定回归受两个基线问题影响而未全绿。

## 文档覆盖

- v2 在 H.264 编码前以 RGB `uint8` 帧收集 `video_stats.json` 统计。
- N 个动作对应 N+1 状态/视频帧，统计只纳入前 N 个动作对齐帧，排除 terminal frame。
- v2 先校验 sidecar，再将统计直接交给 LeRobot，不为统计解码；无效 sidecar 在任何 LeRobot 元数据写入前失败。
- v1 或没有 sidecar 的旧 episode 保持 decode fallback：passthrough 重封装后解码两个相机的前 N 帧。
- passthrough 对同 FPS、无 B 帧、packet 足够的 H.264 逐 packet 原样重封装，不重新编码或改变压缩数据。
- `scripts/backfill_raw_video_stats.py` 默认 dry-run，`--apply` 才写入；有效已有 sidecar 跳过，冲突/无效 sidecar 拒绝覆盖，且不修改 raw MP4、NPZ 或 metadata。

## 变更文件

- `README.md`

## 验证命令与精确输出

### 指定 focused regression suite

```bash
/root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py tests/test_episode.py tests/test_export_lerobot.py tests/test_smolvla_pipeline.py -q
```

```text
3 failed, 65 passed, 3 warnings in 100.48s (0:01:40)
```

失败精确名称与根因如下（其余 65 项已通过）：

```text
FAILED tests/test_episode.py::test_episode_records_action_aligned_video_stats
ImportError: Cannot initialize a EGL device display. This likely means that your EGL driver does not support the PLATFORM_DEVICE extension, which is required for creating a headless rendering context.

FAILED tests/test_episode.py::test_recorder_writes_action_aligned_video_stats
ValueError: statistics quantiles must be ordered

FAILED tests/test_episode.py::test_episode_has_n_plus_one_states
OpenGL.raw.EGL._errors.EGLError: EGLError(err = EGL_BAD_DISPLAY, baseOperation = eglChooseConfig, ...)
```

测试还给出以下既有警告：

```text
PytestReturnNotNoneWarning: tests/test_episode.py::test_config returned <class 'welding_path_vla.core.config.AppConfig'>
PytestUnraisableExceptionWarning: MjRenderContextOffscreen.__del__ raised AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
```

### 两类合成 passthrough smoke export

```bash
/root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_export_lerobot.py -q -k 'stored_video_stats_skip_decode_and_preserve_h264_packets or legacy_video_stats_decode_each_camera'
```

```text
..                                                                       [100%]
2 passed, 14 deselected in 7.35s
```

`test_stored_video_stats_skip_decode_and_preserve_h264_packets` 构造 v2 H.264 双相机 episode，
验证 global/wrist 同时出现在 `meta/stats.json` 和逐 episode parquet 统计；用抛错 mock 证明
v2 统计路径不解码，并比较导出前后两个相机前 3 个 packet 的 SHA-256。`test_legacy_video_stats_decode_each_camera`
构造无 sidecar 的 v1 episode，验证 global 和 wrist 都以 3 个动作帧进入 decode fallback。

### 格式、lint 与 diff

```bash
pixi run -e dev ruff format --check src/welding_path_vla/dataset tests scripts
```

```text
Would reformat: scripts/backfill_raw_video_stats.py
Would reformat: scripts/fix_task_language.py
Would reformat: scripts/generate_evaluation_scenes.py
Would reformat: scripts/inspect_lerobot_dataset.py
Would reformat: src/welding_path_vla/dataset/export_lerobot.py
Would reformat: src/welding_path_vla/dataset/raw_schema.py
Would reformat: src/welding_path_vla/dataset/repair_video_stats.py
Would reformat: src/welding_path_vla/dataset/video_stats.py
Would reformat: tests/test_deployment.py
Would reformat: tests/test_evaluation_scenes.py
Would reformat: tests/test_export_lerobot.py
Would reformat: tests/test_rollout_metrics.py
Would reformat: tests/test_simulation.py
Would reformat: tests/test_smolvla_pipeline.py
Would reformat: tests/test_trihedral_visibility.py
Would reformat: tests/test_video_stats.py
16 files would be reformatted, 47 files already formatted
```

```bash
pixi run -e dev ruff check src/welding_path_vla/dataset tests scripts
```

```text
Found 19 errors.
[*] 7 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).
```

精确诊断类别/位置：`E402` 和 `I001` 位于
`scripts/backfill_raw_video_stats.py`；`RUF001` 位于
`scripts/backfill_raw_video_stats.py:26`、`scripts/generate_evaluation_scenes.py:87`；`I001` 位于
`scripts/export_lerobot.py`、`scripts/rerender_dataset.py`、`scripts/train_xvla.py`、
`src/welding_path_vla/dataset/migrate_task_parameters.py`、`tests/test_export_lerobot.py`、
`tests/test_video_stats.py`；`E501` 位于
`src/welding_path_vla/dataset/repair_video_stats.py:110`、
`src/welding_path_vla/dataset/video_stats.py:157,181`、
`tests/test_export_lerobot.py:190,202`、`tests/test_smolvla_pipeline.py:88`、
`tests/test_video_stats.py:160`；`RUF005` 位于 `tests/test_rollout_metrics.py:47`。

```bash
git diff --check
```

```text
<no output; exit 0>
```

Pixi 成功建立 dev 环境；没有触发 NFS-prefix 锁，因此不需要临时本地 prefix 的 Ruff 回退。

## 自审

- 阅读 `git diff --word-diff=plain -- README.md`，确认新增内容逐项对应任务 brief，且没有修改代码或测试。
- 对照 `raw_schema.py`、`video_stats.py`、`export_lerobot.py` 和回填 CLI：文件名、格式名、动作计数、校验顺序和 H.264 限制均与实现一致。
- `git diff a0a8f3b -- src/welding_path_vla/dataset/video_stats.py tests/test_episode.py` 无输出：回归的 recorder 分位数失败和 EGL 失败不是本任务引入的代码差异。
- 未格式化或 lint 修复既有文件，避免违反“不要修改无关文件”的范围限制。

## 关注点与环境限制

- EGL：此运行环境无法初始化 robosuite headless EGL device display，导致两个 simulation episode 测试失败；stderr 同时显示 `libEGL warning: egl: failed to create dri2 screen`。这不是 README 或原始视频统计快路径的失败。
- 统计：`test_recorder_writes_action_aligned_video_stats` 在基线 `ActionAlignedVideoStats.finish()` 产生的统计进入严格 `validate_statistics()` 时报告 `statistics quantiles must be ordered`。Task 5 的 `compute_video_stats()` 已 canonicalize decode 结果，但 recorder 的 in-memory 路径未在本任务范围内修改。
- Ruff：format/lint 所报 16/19 个既有问题覆盖任务 1--5 的实现及无关脚本/测试；本任务 README 不在 Ruff 目标中。修复它们会超出 Task 6 指定的 `README.md` 修改范围。
- 本轮不涉及 torchcodec 或配置加载失败；可执行的两个合成 smoke 均完成。

## Task 6 fix round 1

提交：`83cd65c fix: canonicalize recorder video statistics`。

### 修复内容

- 将 decode 路径原有的直方图分位数夹紧/单调化提取为
  `canonicalize_statistics()`；它保持 `min`、`max`、`mean`、`std`、`count` 和原始
  0--255 RGB 量纲，仅将 `min`、各 quantile、`max` 夹紧并单调化。
- `compute_video_stats()` 和 `ActionAlignedVideoStats.finish()` 共同调用该函数。因此 recorder
  的 pre-encode accumulator 和 v1 decode fallback 产生相同的 canonical sidecar 统计。
- 扩展 `test_action_aligned_stats_exclude_terminal_frame`：在验证前两个动作帧的 mean=15、
  max=20 后，实际写入并读回双相机 sidecar。该断言同时覆盖 terminal frame=250 继续被排除、
  raw 0--255 值不变，以及 recorder accumulator 输出可通过严格 sidecar 校验。
- 使用 Ruff 格式化并 lint Tasks 1--5 触及的 10 个 dataset/script/test 文件。对回填 CLI 的
  `sys.path` 后导入保留文件级 `E402`/`I001` 豁免，维持“优先导入当前 worktree src”的行为；
  其余修改均为 Ruff 格式或 import 排序。

### TDD 证据

先修改现有 accumulator 测试，使其把统计写入并读回 sidecar：

```bash
/root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py -q -k action_aligned_stats_exclude_terminal_frame
```

```text
FAILED tests/test_video_stats.py::test_action_aligned_stats_exclude_terminal_frame
ValueError: statistics quantiles must be ordered
1 failed, 26 deselected in 7.05s
```

实现共享规范化后，执行 recorder 和 accumulator 回归：

```bash
/root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py tests/test_episode.py -q -k 'action_aligned_stats_exclude_terminal_frame or recorder_writes_action_aligned_video_stats'
```

```text
..                                                                       [100%]
2 passed, 34 deselected in 8.96s
```

### 指定 focused regression suite

```bash
/root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py tests/test_episode.py tests/test_export_lerobot.py tests/test_smolvla_pipeline.py -q
```

```text
2 failed, 66 passed, 3 warnings in 99.58s (0:01:39)
```

两个失败均为此前已确认的 headless EGL 环境限制：

```text
FAILED tests/test_episode.py::test_episode_records_action_aligned_video_stats
ImportError: Cannot initialize a EGL device display.

FAILED tests/test_episode.py::test_episode_has_n_plus_one_states
OpenGL.raw.EGL._errors.EGLError: EGLError(err = EGL_BAD_DISPLAY, baseOperation = eglChooseConfig, ...)
```

stderr 仍为 `libEGL warning: egl: failed to create dri2 screen`。此前的
`test_recorder_writes_action_aligned_video_stats` 分位数失败已消失；本次没有 torchcodec 或配置加载失败。

### Ruff 与 diff

先对 Tasks 1--5 改动集执行：

```bash
pixi run -e dev ruff format --check src/welding_path_vla/dataset/video_stats.py src/welding_path_vla/dataset/raw_schema.py src/welding_path_vla/dataset/recorder.py src/welding_path_vla/dataset/export_lerobot.py src/welding_path_vla/dataset/repair_video_stats.py src/welding_path_vla/dataset/backfill_raw_video_stats.py scripts/backfill_raw_video_stats.py tests/test_episode.py tests/test_export_lerobot.py tests/test_video_stats.py
pixi run -e dev ruff check src/welding_path_vla/dataset/video_stats.py src/welding_path_vla/dataset/raw_schema.py src/welding_path_vla/dataset/recorder.py src/welding_path_vla/dataset/export_lerobot.py src/welding_path_vla/dataset/repair_video_stats.py src/welding_path_vla/dataset/backfill_raw_video_stats.py scripts/backfill_raw_video_stats.py tests/test_episode.py tests/test_export_lerobot.py tests/test_video_stats.py
```

```text
10 files already formatted
All checks passed!
```

按要求再执行完整范围检查：

```bash
pixi run -e dev ruff format --check src/welding_path_vla/dataset tests scripts
pixi run -e dev ruff check src/welding_path_vla/dataset tests scripts
git diff --check
```

```text
Would reformat: scripts/fix_task_language.py
Would reformat: scripts/generate_evaluation_scenes.py
Would reformat: scripts/inspect_lerobot_dataset.py
Would reformat: tests/test_deployment.py
Would reformat: tests/test_evaluation_scenes.py
Would reformat: tests/test_rollout_metrics.py
Would reformat: tests/test_simulation.py
Would reformat: tests/test_smolvla_pipeline.py
Would reformat: tests/test_trihedral_visibility.py
9 files would be reformatted, 54 files already formatted

Found 7 errors.
[*] 4 fixable with the `--fix` option (1 hidden fix can be enabled with the `--unsafe-fixes` option).

format_status=1 lint_status=1 diff_status=0
```

完整 lint 的 7 个遗留位置为 `scripts/export_lerobot.py` (`I001`)、
`scripts/generate_evaluation_scenes.py:87` (`RUF001`)、`scripts/rerender_dataset.py`
(`I001`)、`scripts/train_xvla.py` (`I001`)、
`src/welding_path_vla/dataset/migrate_task_parameters.py` (`I001`)、
`tests/test_rollout_metrics.py:47` (`RUF005`) 和
`tests/test_smolvla_pipeline.py:88` (`E501`)。完整 format 的 9 个文件与完整 lint 的这些
文件都不属于 Tasks 1--5 的改动集，因此没有改写它们。`git diff --check` 无输出且退出 0。

### fix round 1 自审与关注点

- `canonicalize_statistics()` 对 decode 和 recorder 共用，避免再次产生两个统计契约；调用在
  action-frame 一帧延迟逻辑之后，故不改变 N/N+1 对齐或 terminal-frame exclusion。
- 逐项查看 `git diff`：功能性变更仅在共享统计函数和其 focused regression；其余六个文件为
  指定 Task 1--5 范围内的 Ruff 格式/import 机械修改。
- 目标 10 文件 Ruff 已全绿；全目录的 9/7 项与本轮未改文件相同，保留为 pre-existing
  formatting/lint concerns。EGL 仍阻塞两项 simulation integration test，需具备 EGL
  PLATFORM_DEVICE 的 headless GPU 环境才能完成。

## Final fix wave

### 修复内容

- `passthrough_statistics()` 对带有效 `video_stats.json` 的 v1 episode 与 v2 一样复用统计；只有缺失 sidecar 的 v1 才解码回退。
- sidecar 现在必含并严格验证 `color_space=RGB`、`frame_selection=first_action_count_frames_excluding_terminal_state` 与逐相机正整数 HWC `image_shapes`；录制器保存采集帧形状，回填从视频流读取形状。
- README 明确 sidecar 保存 0--255 的未归一化 raw RGB 值，归一化由 LeRobot 完成。

### TDD 与最终验证

先新增 v1 回填 sidecar 不调用 `compute_video_stats` 的导出测试，以及 required metadata 的 round-trip/拒绝测试；实现前运行得到 5 failures（v1 调用 decode、缺少 `color_space`、且错误 metadata 未拒绝）。实现后执行：

```bash
git diff --check
PYTHONPATH=src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_export_lerobot.py::test_backfilled_v1_video_stats_skip_decode tests/test_video_stats.py::test_raw_video_stats_round_trip_includes_required_image_metadata tests/test_video_stats.py::test_raw_video_stats_rejects_invalid_required_image_metadata -q
pixi run ruff check src/welding_path_vla/dataset/video_stats.py src/welding_path_vla/dataset/export_lerobot.py src/welding_path_vla/dataset/recorder.py src/welding_path_vla/dataset/backfill_raw_video_stats.py tests/test_video_stats.py tests/test_export_lerobot.py tests/test_episode.py
```

输出：

```text
......                                                                   [100%]
6 passed in 7.56s
All checks passed!
```

`git diff --check` 无输出且退出 0。

### 自审与关注点

- v2 仍在创建 LeRobot 元数据前校验 sidecar；v1 缺失 sidecar 的既有 decode fallback 未改变。
- action-aligned accumulator 继续只累计前 N 帧，terminal `N+1` 帧仍被排除；sidecar raw 统计量纲和 H.264 remux 路径未改。
- 本轮按用户中断要求未运行广泛套件；已知 host EGL 限制的仿真测试未重跑。
