# 数据采集与训练工作流

本文描述当前推荐链路：仿真原始 episode → 质量验证 → LeRobot Dataset v3 → 策略训练 →
离线评估或闭环部署。本文侧重实际操作；几何专家、逐步时间对齐、raw 字段和 LeRobot 文件结构见
[仿真数据采集原理与格式](data-collection.md)，命令入口和配置覆盖规则见
[脚本入口与配置规则](script-entrypoints.md)。

## 1. 数据分层

项目把原始事实、训练表示和模型输出分开：

```text
robosuite / 真机
  → 原始 episode（绝对状态、绝对命令、视频、任务与质量信息）
  → LeRobot Dataset（视频 feature、13D state、9D absolute EE target）
  → relative-action processor（以当前 TCP 为共享锚点）
  → policy（预测短时 relative action chunk）
  → postprocessor（恢复世界系 absolute EE target）
  → 安全门、IK 与控制器
```

原始 episode 是唯一事实源，不绑定某个策略或动作块长度。一个目录包含：

```text
episode_000000/
├── metadata.json      # seed、任务参数、完整配置、质量状态和坐标约定
├── trajectory.npz     # 状态、参考轨迹、命令、执行结果和诊断信号
├── global.mp4         # 640×480、30 FPS、H.264
└── wrist.mp4          # 640×480、30 FPS、H.264
```

当前时间尺度为 600 Hz MuJoCo 物理、120 Hz IK/位置控制和 30 Hz 策略观测。原始视频、
关节状态与 TCP 状态按策略频率同步；需要研究底层控制动态时，应另建高频控制流，不重复写图像。
各数组的形状、单位及 `N` 条动作与 `N+1` 个状态的对应关系见
[原始 episode 保存了什么](data-collection.md#4-原始-episode-保存了什么)。

## 2. 任务随机化与质量门

初始化顺序为：采样成组任务参数 → 随机化工件 → 求解 staging → 随机化关节和 TCP →
预检完整参考轨迹 → 执行并验证。不可达、跳解、限位或预检碰撞会触发重采样，不进入正式录制。

任务方向、姿态、速度、圆弧参数和曲线参数默认每
`randomization.task_group_size=10` 个全局 episode 编号变化一次，并按字段精度取整。工件位姿、
初始关节、TCP 偏移和恢复扰动仍逐条变化。英文 `task.instruction` 在同一任务内保持固定；具体
速度、方向和几何变化写入 `task_parameters`，避免语言标签随样本产生无意义的细碎变化。

当前任务及数据目录见[工件与焊缝任务](simulation/workpieces.md)。调整随机性时遵循两条规则：

1. 先保证完整轨迹可达、无碰撞，再扩大范围；
2. 一次只扩大一类因素，持续监测拒绝率、IK 残差和关节覆盖。

执行后的 episode 只有满足进度、横向误差、姿态、IK 和碰撞条件才标记为
`valid_success` 或 `valid_recovery`。`quality.failure_reasons` 和 `collision_pairs` 保留具体原因，
不能用视频观感替代结构化碰撞判定。

## 3. 仿真采集

先以单 worker 检查场景和一小批数据：

```bash
pixi install -e sim
pixi run -e sim sim-view --config_path=configs/default.yaml
pixi run -e sim sim-collect \
  --config_path=configs/default.yaml \
  --collection.episodes=5 \
  --collection.workers=1
pixi run -e sim data-validate \
  --config_path=configs/default.yaml \
  --collection.dataset_root=datasets/weldpath_raw_v2
```

回放一条数据：

```bash
pixi run -e sim sim-replay \
  --episode=datasets/weldpath_raw_v2/episodes/episode_000000
```

确认机器人外观、双相机、焊枪姿态、进度和碰撞信号后再并行采集：

```bash
pixi run -e sim sim-collect \
  --config_path=configs/curve_plate.yaml \
  --collection.episodes=500 \
  --collection.workers=4
```

每个 worker 独立持有 MuJoCo、EGL 和视频编码器。增加 worker 会同时提高 GPU 渲染、CPU 编码
和内存压力；从 2 开始测量吞吐，出现显存不足或驱动不稳定时退回 1。交互式采集自动使用单进程。

无头采集默认使用 `camera.offscreen_backend: egl`。若目标机器没有可用 EGL，可在确认 X11
会话存在后临时使用 `MUJOCO_GL=glfw`；OSMesa 需要系统软件渲染库。服务器部署前应先用
`sim-view` 或单 episode 验证图形后端。

## 4. 动作表示

状态向量固定为：

```text
[joint_1 … joint_6, tcp_x, tcp_y, tcp_z, tcp_qw, tcp_qx, tcp_qy, tcp_qz]
```

动作固定为 9D 末端目标：三维位置加旋转矩阵前两行的 rotation-6D。原始数据和 LeRobot
Parquet 保存世界系 absolute EE target；训练 preprocessor 在归一化之前把一个 future chunk
统一变换到预测时刻的 TCP 局部坐标系。块内所有目标共享同一个锚点，不是逐步 delta。

完整的帧映射、Parquet schema、视频 shard 和 relative-action 公式见
[LeRobot Dataset 保存了什么](data-collection.md#6-lerobot-dataset-保存了什么)。

策略输出 relative action；postprocessor 使用同一次观测缓存的 TCP 锚点恢复世界系目标，
再交给安全门和 IK。动作统计也在 relative 空间计算，因此不能用旧 absolute/delta 数据集的
`meta/stats.json` 训练当前 checkpoint。

需要在研究代码中直接构造动作块时使用：

```python
from welding_path_vla.dataset.actions import build_relative_actions
from welding_path_vla.dataset.raw_schema import EpisodeReader

episode = EpisodeReader("datasets/weldpath_raw_v2/episodes/episode_000000")
chunk = build_relative_actions(
    episode,
    frame_index=100,
    horizon=30,
    stride=1,
    source="safe_command",
)
actions = chunk.values       # (30, 9)
valid_mask = chunk.valid_mask
```

尾部填充必须由 `valid_mask` 屏蔽。默认标签源 `safe_command` 表示经过 IK 和速度约束后真正
发送给控制器的目标；`reference` 只用于几何专家消融，`executed` 适合没有显式命令的示教数据。

## 5. 导出 LeRobot Dataset

新建一个视频数据集：

```bash
pixi install -e data
pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset_glob='datasets/*_raw_v2' \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=USER/weldpath_relative_v1
```

默认只保存视频 feature，不保留逐帧图片。LeRobot 使用 AV1、`yuv420p`、CRF 30、preset 12；
双相机由两个编码线程并行处理，但多个 raw 数据源顺序追加，避免多个 writer 竞争 metadata。
默认流式编码直接完成“raw MP4 解码帧 → AV1 MP4”，不会执行临时 PNG 的写入和读取；每路相机
最多排队 30 帧，每路编码器使用 4 个线程。终端只显示数据集、episode 和帧进度，隐藏 Hugging
Face `Map` 进度及 SVT 初始化信息。可用以下参数调节：

- `lerobot_export.parallel_video_encoding`：关闭流式模式时，并行编码双相机；
- `lerobot_export.encoder_queue_maxsize`：每路编码器最大排队帧数，默认 30；
- `lerobot_export.encoder_threads`：每个编码器线程数；
- `lerobot_export.streaming_encoding=false`：仅在编码兼容问题时退回较慢的临时 PNG 流程；
- `lerobot_export.save_images=true`：改为图片 feature，不生成视频 feature。
- `lerobot_export.video_codec=h264`：愿意降低画质换取速度时改用 H.264 编码。

不要用外层 episode 多进程同时写一个 LeRobot 目标。内存紧张时先减小编码队列或编码线程数；
`collection.workers` 只影响仿真采集，不影响数据导出。

### 5.1 为旧数据补录焊缝状态

旧 raw 仿真数据已经保存完整配置、工件位姿和逐帧关节/TCP 状态，因此无需重新运行专家或物理控制，
可以只恢复每帧场景并重新渲染双相机视频：

```bash
pixi run -e sim rerender-dataset \
  --dataset=datasets/weldpath_trihedral_vertical_raw_v2
```

脚本先在 episode 内的临时目录录制并校验两路视频帧数，成功后才替换 `global.mp4` 和
`wrist.mp4`；`trajectory.npz`、`metadata.json`、初始姿态、状态和动作均不写入。

LeRobot 数据必须能找到 manifest 中对应的 raw 唯一事实源：

```bash
pixi run -e sim rerender-dataset \
  --dataset=datasets/weldpath_lerobot_relative_v1 \
  --raw_dataset_glob='datasets/*_raw_v2'
```

该模式先同步重渲染涉及的 raw episode，再在同一磁盘的临时目录重建 LeRobot 视觉数据。替换前会
逐列验证 `observation.state`、`action`、时间戳、frame/episode/index 和任务映射完全相等；验证失败
时保留原 LeRobot 数据集。默认成功后删除旧目录，也可加 `--keep_backup=true` 保留
`<dataset>_before_rerender`。运行前必须结束该数据集上的采集、导出或训练写入进程。

按原始 episode 编号选择闭区间：

```bash
pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset=datasets/weldpath_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --lerobot_export.start_episode=100 \
  --lerobot_export.end_episode=199
```

原始数据增加后增量追加：

```bash
pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset_glob='datasets/*_raw_v2' \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --lerobot_export.incremental=true
```

`meta/welding_path_vla_export.json` 记录源目录、已导出 episode 和动作契约；重复源 episode 会跳过。
同一目标不得混用图片/视频 schema、动作 horizon 或 stride。

## 6. Hugging Face Dataset Hub

令牌保存在 Hugging Face 凭据库，不写入 YAML：

```bash
pixi run -e data hf auth whoami
pixi run -e data hf auth login
```

转换完成后自动上传：

```bash
HF_XET_HIGH_PERFORMANCE=1 pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset_glob='datasets/*_raw_v2' \
  --output=/path/to/weldpath_lerobot_relative_v1 \
  --repo_id=mintinson/weldpath_relative_v1 \
  --lerobot_export.push_to_hub=true
```

上传发生在数据集 `finalize()` 之后，不是边转换边发布；否则 Parquet footer、视频索引和 metadata
尚不完整。网络中断但本地转换已完成时，只重新上传现有目录：

```bash
HF_XET_HIGH_PERFORMANCE=1 pixi run -e data upload-lerobot \
  --dataset=/run/media/mintinson/DataDiskD/welding_path_vla/lerobot/weldpath_relative_v1 \
  --repo_id=mintinson/weldpath_relative_v1
```

重复执行上传会复用 Hub/Xet 已有文件，不重新转换 raw episode。不要使用 `sudo`：它会改变 PATH、
文件所有者和 Hugging Face 凭据。目录权限有问题时只修正专用数据目录的所有者。

## 7. 训练、恢复与评估

所有 LeRobot 策略共用相同入口。先检查数据和最终训练计划：

```bash
pixi install -e train
pixi run -e train policy-data-check --config_path=configs/smolvla.yaml
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --dry_run=true
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --training.resume=false
```

训练日志写入 `<training.output_dir>/train.log`；W&B 默认以 offline 模式写入同目录的 `wandb/`。
联网后使用 `wandb sync` 上传。训练中断后保持原 `training.output_dir`：

```bash
pixi run -e train train-policy \
  --config_path=configs/smolvla.yaml \
  --training.resume=true
```

`training.steps` 是结束时的总 optimizer step，不是本次追加量。只加载权重、但不恢复 optimizer、
scheduler 和全局 step 时，使用 `policy.checkpoint` 并指定新的输出目录。

双 A100 使用对应的 `_a100.yaml` 和 DDP 入口：

```bash
pixi run -e train train-policy-2gpu \
  --config_path=configs/trajectory_vla_a100.yaml
```

batch、step 和数据遍历量的计算见[训练规模指南](training-scale-guide.md)。模型特有设置见
[ACT](act-pipeline.md)、[SmolVLA](smolvla-pipeline.md)、[π0 / π0.5](pi-pipeline.md)和
[Trajectory-VLA](trajectory-vla.md)。

离线评估只衡量留出数据拟合，不等价于闭环成功率：

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/smolvla.yaml \
  --policy.checkpoint=CHECKPOINT
```

SmolVLA、π0、π0.5 和 Trajectory-VLA 的闭环任务使用
`configs/deploy/{policy}_{task}.yaml`；ACT 与 Traj-VLA-Qwen 当前使用策略组合入口和命令行
checkpoint。所有 rollout 都应结合[统一评估规范](evaluation.md)解释 `summary.json`。

## 8. 数据规模快照与扩展顺序

2026-08-10 的本地 LeRobot 数据集快照为 4,250 个 episode、4,096,853 帧、15 个 task index；
按当前 10% episode 留出规则，训练划分为 3,674,023 帧。该数字只用于复现实验配置，数据增加后
应重新运行 `policy-data-check`，不要继续照抄旧 step。

扩大多样性时建议依次加入：

1. 工件和焊缝几何、执行方向与焊缝长度；
2. 机器人初态与工件位姿；
3. 速度、工作角、行走角和滚转角；
4. 可恢复的轨迹扰动；
5. 光照、材质、背景和小范围相机误差；
6. 经实测约束的 TCP、关节零偏、延迟和摩擦；
7. 使用同一 schema 的高质量真机数据。

训练、验证和测试必须按 episode / scene seed 划分，不能按帧随机拆分。每次扩大采集前至少检查：

- 有效 episode 数和失败原因分布；
- 碰撞对、IK 残差、关节限位与初态覆盖；
- 全局相机包含机器人和工件，腕部相机没有结构性遮挡；
- `reference`、`safe_command` 和 `executed` 的差异符合控制链；
- 动作 source、frame、rotation、horizon 和 stride 已写入实验配置。
