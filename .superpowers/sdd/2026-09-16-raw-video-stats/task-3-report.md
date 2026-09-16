# Task 3 报告：passthrough export statistics fast path

## 实现结果

`PassthroughEpisodeEncoder` 现接收 `(videos, frame_count, statistics=None)`。
v2 episode 在创建 LeRobot writer 前读取并校验 `video_stats.json`，将 raw
`global`、`wrist` 统计映射到 `observation.images.global`、
`observation.images.wrist`。这些值保持 0--255 原始量纲，交给 LeRobot
writer 统一归一化。缺失 sidecar 的 v1 episode 仅对两个重封装后的相机视频
调用 `compute_video_stats` 解码回退。H.264 packet remux 路径未改动。

`compute_video_stats` 使用动作对齐的前 N 帧，复用 RGB 校验/展平逻辑，并返回
LeRobot streaming writer 所需的 raw `count`、`mean`、`min`、`max`、`std`。

## TDD 证据

### RED

先添加 v2 不解码、v1 双相机解码、无效 v2 提前失败以及回退统计测试，运行：

```text
PYTHONPATH=src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_export_lerobot.py -k "stored_video_stats or legacy_video_stats" -q
```

精确结果：

```text
FAILED tests/test_export_lerobot.py::test_stored_video_stats_skip_decode_and_preserve_h264_packets
AttributeError: module 'welding_path_vla.dataset.export_lerobot' has no attribute 'compute_video_stats'
FAILED tests/test_export_lerobot.py::test_legacy_video_stats_decode_each_camera
AttributeError: module 'welding_path_vla.dataset.export_lerobot' has no attribute 'compute_video_stats'
2 failed, 13 deselected in 10.04s
```

回退统计测试在实现前也无法收集：

```text
ImportError: cannot import name 'compute_video_stats' from 'welding_path_vla.dataset.video_stats'
1 error in 0.36s
```

这证明测试会捕获缺少回退接口和缺少 passthrough supplied-statistics 快路径的缺陷。

### GREEN

提交前重新运行的精确命令和输出：

```text
PYTHONPATH=src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_export_lerobot.py -k "stored_video_stats or legacy_video_stats or invalid_v2_video_stats" -q
...                                                                      [100%]
3 passed, 12 deselected in 10.67s

PYTHONPATH=src /root/micromamba/envs/welding_vla/bin/python -m pytest tests/test_video_stats.py -q
.............                                                            [100%]
13 passed in 0.38s

git diff --check
```

`git diff --check` 无输出并以 0 退出。

## 覆盖范围

- v2 sidecar：将 `compute_video_stats` 替换为抛错函数后仍成功导出。
- v1 无 sidecar：spy 确认 global/wrist 两路各解码一次。
- 无效 v2 sidecar：在 `meta/` 创建前抛出 `ValueError`。
- 端到端：`meta/stats.json` 和 episode Parquet 的每个相机 mean/min/max 都等于
  sidecar 值除以 255；前 N 个 H.264 packet 哈希保持相同。
- 回退：只取动作帧，排除 terminal frame。

## 修改文件

- `src/welding_path_vla/dataset/export_lerobot.py`
- `src/welding_path_vla/dataset/video_stats.py`
- `tests/test_export_lerobot.py`
- `tests/test_video_stats.py`

## 自审

- v2 校验循环位于 `open_lerobot_dataset()` 之前，满足元数据创建顺序。
- 导出器未归一化 sidecar；归一化只由 LeRobot writer 的既有逻辑完成。
- 未修改 `remux_h264_frames()`，packet passthrough 的回归测试验证 packet 内容。
- 所有新测试均使用真实导出器、writer 和 MP4；仅 v1 解码调用使用 spy 来观察
  两路相机边界行为。
- 未运行完整仓库测试套件，遵循任务限制。

## 关注项

当前 v2 sidecar 契约只持久化 mean/min/max，而本机 LeRobot writer 聚合时还需要
std/count。count 使用动作帧数（各 episode 分辨率固定，因此权重比例一致），std
只能置零；mean/min/max 保持 sidecar 的精确值并由测试覆盖。若图像标准差参与后续
训练归一化，应在未来版本的 raw sidecar 契约中显式记录 std 与像素样本数，不能在
本任务的无解码快路径中恢复它们。
