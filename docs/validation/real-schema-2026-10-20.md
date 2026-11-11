# 2026-10-20 真机格式与服务器训练子集对照

参考数据：`datasets/weldpath_lerobot_relative_small`，30 个 episode、27,445 帧、6 个任务。
对照对象：`datasets/real/vm_black_smoke_20261019`，以及修改后经原生录制生成的完整标签测试 fixture。

结论：原有真机核心数值、图像和动作存储格式已经兼容，但旧黑帧联调集缺少
`task.direction` 与 `task.parameters`，不能直接用于启用了这些字段的服务器训练配置。
根据用户明确的要求，正式采集与整理仅强制方向；工艺参数及自定义标签均可省略。
参考数据集校验也遵循这一规则，已提供的可选字段仍须满足格式定义。
阶段标签按用户要求暂不要求，真机训练关闭 phase sampling。

编码对齐修正：首次对照只确认了训练兼容性，未将 H.264/AV1 的实际编码差别纳入一致性检查，
该结论不完整。正式录制现已改为 AV1，并按参考集检查完整视频 info，不再把 H.264 判为格式一致。

## 字段与契约

| 项目 | 参考数据 | 真机结果与处理 |
|---|---|---|
| LeRobot 版本及布局 | v3.0，Parquet + MP4 + meta/episodes | 一致 |
| 全局/腕部图像 | RGB，480×640×3 | 一致 |
| `observation.state` | float32[13]，关节角 + 米制 TCP + wxyz 四元数 | 类型、维度和 names 一致 |
| `action` | float32[9]，XYZ + rotation-6D 两行 | 类型、维度和 names 一致 |
| `task.direction` | int64，forward=0 / reverse=1 | 正式采集必须填写，原生写入类型相同 |
| `task.parameters` | float32[4]，速度/工作角/行走角/工具滚转角 | 可省略；提供时完整填写四项，名称、顺序和类型相同 |
| frame/episode/task 索引及 timestamp | 原生 LeRobot 标量字段 | Parquet 类型一致 |
| 动作清单 | relative_action / absolute_ee_world / prediction_tcp / rotation_6d_rows / horizon=30 / stride=1 | 全部一致 |
| action normalization stats | 长度 9 的统计与分位数 | 复用同一 relative action 统计函数，统计定义与形状一致；数值按各自数据计算 |
| 阶段 | `meta/phase_index.npz`，approach/track/retreat | 用户确认暂不要求，不补虚构标签，phase sampling 关闭 |
| 视频编码 | AV1、yuv420p、GOP 2、CRF 30、preset 12、fast_decode 0 | 正式采集已全部对齐，实际视频流检查为 AV1 |

目录名称包含 relative，不代表 Parquet 中直接保存相对动作。参考数据也保存世界系绝对位姿，
训练时才围绕当前 TCP 转成相对动作；真机沿用相同设计。
监督来源仍有差别：真机动作来自后继实测状态，不应伪装成仿真控制命令。

`robot_type=null` 与 `robot_type=elfin_welding` 是真实来源差别，必须保留。
参考数据的 `sources`/`subset.json` 记录仿真导出和子集抽取，真机使用采集契约及 SDK 会话日志；
这些不是模型输入字段，不复制不真实的仿真来源。自定义 `task.extra.*` 仍作为可选扩展保存。

## 修改后的正式流程

1. `configs/real_record.yaml` 默认 `require_task_labels: true`，仅要求用户明确填写真实方向。
   `parameters` 和 `extra` 可省略；方向未填写时在创建数据和连接设备前报错。
2. `prepare-real-data` 默认要求方向，检查字段 dtype、shape、names、方向编码；
   工艺参数仅在提供时检查其四维格式和有限性。
3. 传入 `--reference_dataset=datasets/weldpath_lerobot_relative_small` 可额外核对参考 schema、
   FPS、LeRobot 版本、视频 info（含 codec、像素格式及编码参数）及完整动作契约，包括 horizon/stride。
   不要求补齐参考集中的可选任务标签，也不要求 phase sidecar。
4. 整理清单补上 `video_storage.mode=encoded`；不改写状态、动作、Parquet 或视频。
5. `configs/real_training.yaml` 默认只启用方向提示，phase sampling 保持关闭。
   数据包含四项参数时，可另行启用相应提示字段。
6. 仅无方向标签的联调数据需要显式使用 `--require_task_labels=false`，训练提示设为 `[]`。

```bash
pixi run --no-install --frozen -e real prepare-real-data \
  --dataset=datasets/real/elfin_demo \
  --reference_dataset=datasets/weldpath_lerobot_relative_small
pixi run --no-install --frozen -e real policy-data-check --config_path=configs/real_training.yaml
```

## 验证证据与边界

完整标签回归测试调用原生 `record()`，模拟只读 CPS 与双 RGB 相机，生成并恢复录制
2 个 episode、64 帧数据。其所有参考必需 Parquet 字段类型及 `meta/info.json` 的
dtype/shape/names 与真实参考子集一致，动作清单也一致。随后通过完整参数训练数据检查，
并实际读回样本、构造方向与四项工艺参数提示。

编码修正后的录制测试还检查两路实际 MP4 流均为 AV1，并成功解码为 480×640×3 RGB；
保存的视频 info 与参考数据逐项一致。旧 H.264 VM 测试数据和旧 fixture 保留为历史测试结果，
不声称它们已经变成 AV1。

相关 schema、录制、恢复、设备与脚本入口测试共 **55 项通过**（包含仅方向的正式录制、整理、
训练数据检查，以及参考集可选字段缺失与编码不一致拒绝测试）；变更文件 Ruff、
BasedPyright、`git diff --check` 及 Pixi 锁文件检查通过，未安装新依赖。

这是用于验证格式的模拟设备 fixture，不是新采集的 VM 真实运动数据；测试标签仅用于测试。
保留结果位于 `outputs/validation/real_schema_20261020/compatible_recording_fixture`。
首次对照留下的 H.264 fixture 已移到同目录 `legacy_h264_recording_fixture`，保留历史结果。
逐字段对照结果见同目录 `comparison.json`，整理及训练入口检查见 `prepared.json`、
`policy_data_check.json`。

旧 VM 数据在严格模式下被正确拒绝，错误为缺失 `task.direction`。
已检查其全部文件 SHA-256 在校验前后保持不变；未复制参考集标签、未填零或默认参数。
缺失的方向必须由用户明确提供后重新录制，或通过另行审核的人工标注流程补充；
缺失的工艺参数及自定义标签无需补齐。

此前 VM 采样实际约 27.83 Hz、metadata 声明 30 Hz 的时序问题仍未修复。
schema 一致不表示物理采样时序合格；本次未修改采样调度，也未重新进行硬件或模型训练验收。

上述为格式对照阶段的历史结论；后续已修正连续采样调度并完成新一轮实机读取验证，
详见[连续观测配对修正](consecutive-capture-2026-10-20.md)。
