# 数据采集、训练与多样性扩展报告

本文命令通过 Pixi 任务调用仓库 `scripts/` 下的单一职责入口；具体映射见
[脚本入口与模块边界](script-entrypoints.md)。

## 1. 当前数据链路

项目采用“不可变原始 episode → 动作适配器 → LeRobot/自定义训练器”三层结构。仿真采集不把数据绑定到某一种 VLA 动作定义，而是同时保存：

- 30 Hz 双相机图像、时间戳、关节位置和关节速度；
- world frame 下的实际 TCP 绝对位姿；
- 几何专家生成的参考 TCP 绝对位姿；
- 经过 IK 和关节速度限制后的安全命令 TCP 绝对位姿；
- seam/base/world 三个坐标系下的单步命令；
- phase、焊缝进度、跟踪误差、IK 残差、碰撞对和 episode 结束标记；
- 工件位姿、任务速度、工作角、行走角、工具滚转角以及全部解析后配置。

`weldpath_raw_v1` 的单个 episode 结构为：

```text
episode_000000/
├── metadata.json
├── trajectory.npz
├── global.mp4
└── wrist.mp4
```

当前数据配置使用 30 Hz 策略观测。robosuite 每次 `step()` 对应一个策略周期，并在内部
维持 MuJoCo 600 Hz 仿真和 120 Hz IK/位置控制；双相机、关节与 TCP 由同一个 observation
dictionary 更新。当前尚未单独落盘 `control.npz`；需要研究底层控制动态时，再增加独立的
120 Hz 控制流，不应把重复图像写到 120 Hz。`weldpath_raw_v1` 是 schema 标识，不表示
采样频率；新的 30 Hz 数据默认写入 `datasets/weldpath_raw_v2`，避免与旧 20 Hz 数据混用。

## 2. 初始位姿和场景如何随机化

每个 episode 按如下顺序初始化：

1. 在 `xy_m / z_m / yaw_deg` 范围内随机化工件；
2. 求解工件上方的 staging 位姿，并拒绝不可达或碰撞场景；
3. 以 staging 关节构型为中心，按 `joint_degs` 对六个关节独立均匀采样；
4. 应用关节限位余量并拒绝碰撞构型；
5. 在 `initial_tcp_m` 范围内额外采样 TCP 平移，失败时确定性重试；
6. 按概率插入轨迹中途恢复扰动；
7. 执行专家轨迹，保存成功、恢复成功和失败 episode，训练导出时只选有效数据。

默认配置：

```yaml
randomization:
  xy_m: 0.1
  z_m: 0.0
  yaw_deg: 30.0
  joint_degs: [60, 20, 20, 30, 60, 60]
  max_sampling_attempts: 10
  initial_tcp_m: 0.1
  recovery_probability: 0.25
  recovery_position_m: 0.005
  recovery_rotation_deg: 3.0
```

`joint_degs` 是六轴最大独立偏移，不是固定偏移。每条数据的实际偏移、采样次数和 TCP 偏移均写入 `metadata.json`，因此可以检查数据覆盖范围并复现实验。提高范围后无效 episode 会增加，这是合理现象；`--collection.episodes=N` 表示目标有效 episode 数，失败数据保留用于诊断但不会被 LeRobot 导出器用于行为克隆。

## 3. 如何采集仿真数据

安装并先目视检查场景：

```bash
pixi install -e sim
pixi run -e sim sim-view --config_path=configs/default.yaml
```

无头采集默认通过 `camera.offscreen_backend: egl` 使用 EGL 离屏上下文。交互式
`sim-view` 在 Wayland 会话中显式使用已有的 XWayland / X11 后端，避免 GLFW 窗口位置、
libdecor 和 `mjr_makeContext` OpenGL 0x502 提示。若机器没有 EGL，可临时执行
`MUJOCO_GL=glfw pixi run -e sim sim-collect ...`；OSMesa 只有在系统安装对应软件渲染库后
才能使用。

切换到圆管下圆弧或上圆弧只需更换配置：

```bash
pixi run -e sim sim-view --config_path=configs/pipe_bottom.yaml
pixi run -e sim sim-collect \
  --config_path=configs/pipe_top.yaml \
  --collection.episodes=5
```

两个配置使用独立的数据集目录，不会与默认 L 形直线任务混写。
专家轨迹通过 `task.approach_speed_mps`、`task.speed_mps` 和
`task.retreat_speed_mps` 分别控制接近、焊接和退出速度。圆管配置默认使用
`60 / 4 / 40 mm/s`，使空中接近更快、沿焊缝跟踪更慢。

先采集 5 条小样本：

```bash
pixi run -e sim sim-collect \
  --config_path=configs/default.yaml \
  --collection.episodes=5

pixi run -e sim data-validate \
  --collection.dataset_root=datasets/weldpath_raw_v2
```

回放全局和腕部视频：

```bash
pixi run -e sim sim-replay \
  --episode=datasets/weldpath_raw_v2/episodes/episode_000000
```

确认相机、焊枪朝向、焊缝进度和碰撞结果后再扩大规模：

```bash
pixi run -e sim sim-collect \
  --config_path=configs/default.yaml \
  --collection.episodes=500
```

建议为每组消融实验复制一份 YAML，并使用不同的 `collection.dataset_root`，不要在同一个目录混入相机参数、动作语义或机器人模型版本不同的数据。

原始 episode 与策略 rollout 共用 `VideoRecorder`。它直接复用 LeRobot 的
`StreamingVideoEncoder` 和 PyAV，生成 `H.264 / yuv420p / avc1` MP4，不再维护
OpenCV backend、FFmpeg 子进程或码率参数；输出可由 VS Code 和浏览器直接播放。

## 4. 动作标签如何构造

仿真监督标签优先使用 `safe_command`，因为它代表经过 IK 和速度限制后真正发送给控制器的目标。`reference` 用于几何轨迹消融，`executed` 用于拖拽示教或没有显式命令的真机数据。

`build_relative_actions` 根据 LeRobot 的定义实现相对当前末端坐标系的累计 future chunk：

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

actions = chunk.values       # (30, 9): xyz + rotation_6d_rows
valid_mask = chunk.valid_mask
```

每个 future target 都以当前真实 TCP 为共同参考，而不是相对前一个 future target。episode 尾部重复最后目标以维持固定形状，同时必须用 `valid_mask` 屏蔽填充损失。

## 5. 导出 LeRobot 数据

安装数据环境并导出有效 episode：

```bash
pixi install -e data
pixi run -e data export-lerobot \
  --config_path=configs/default.yaml \
  --dataset=datasets/weldpath_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=YOUR_NAME/weldpath_relative_v1
```

默认配置只生成 LeRobot 视频 feature，不保留逐帧图片。转换器逐帧读取原始 MP4，批量计算数值 action，并将两个相机直接送入独立编码线程；这样省去了整段视频驻留内存和临时 PNG I/O。最终数据使用 LeRobot 官方的
`libsvtav1 / CRF 30 / preset 12` 默认配置，在焊缝细节、文件体积和训练解码之间取得平衡。
需要快速人工预览时查看原始 H.264 即可，不必为了播放器兼容性改变训练数据编码。需要图片
feature 时显式添加 `--lerobot_export.save_images=true`，同一目标数据集不能混用视频和图片
schema。

源 episode 编号筛选采用闭区间，且仍会自动排除无效 episode：

```bash
# 仅转换 episode_000100 到 episode_000199
pixi run -e data export-lerobot \
  --dataset=datasets/weldpath_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --lerobot_export.start_episode=100 \
  --lerobot_export.end_episode=199
```

只给 `--lerobot_export.start_episode` 表示从该编号到末尾，只给
`--lerobot_export.end_episode` 表示从开头到该编号。原始数据增加后，可恢复已有目标并接着写：

```bash
pixi run -e data export-lerobot \
  --dataset=datasets/weldpath_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --lerobot_export.incremental=true \
  --lerobot_export.start_episode=200
```

转换器在目标的 `meta/welding_path_vla_export.json` 中记录已完成的源 episode 和动作契约；重叠区间会自动跳过，不会重复写入。旧数据缺少该契约时会要求重新导出。恢复视频目标时沿用其原编码格式，避免在同一视频 chunk 中混入不同 codec。

如果不使用流式编码，可用
`--lerobot_export.streaming_encoding=false --lerobot_export.image_writer_processes=2`
`--lerobot_export.image_writer_threads=4` 启用 LeRobot 的临时图片与多进程相机编码路径。
通常默认流式模式 I/O 更少，更适合本项目的双相机长 episode。

LeRobot parquet 中保存 9D 世界系 absolute EE targets。加载 future chunk 后，项目 processor 在归一化前把整段目标统一转换到预测时刻 TCP 坐标系；`meta/stats.json` 中的 action 统计量也在相同 relative action 空间计算。这样所有 policy 共享完全一致的动作语义。

## 6. 如何训练 SmolVLA 基线

先在 `configs/default.yaml` 中填写与导出一致的配置：

```yaml
policy:
  family: smolvla
  device: cuda
  action_horizon: 30
  action_steps: 8
  action_representation: relative_action
  action_source: safe_command

training:
  dataset_repo_id: YOUR_NAME/weldpath_relative_v1
  dataset_root: datasets/weldpath_lerobot_relative_v1
  output_dir: outputs/train/smolvla_v2
  batch_size: 16
  steps: 100000
```

安装训练环境，先打印最终命令：

```bash
pixi install -e train
pixi run -e train train-policy \
  --config_path=configs/default.yaml \
  --dry_run=true
```

确认数据路径、GPU 和输出目录后开始训练：

```bash
pixi run -e train train-policy --config_path=configs/default.yaml
```

训练器会从 YAML 生成 LeRobot 0.6 的 `lerobot-train` 参数，包括 policy 类型、设备、chunk 长度、batch size、steps 和输出目录。首次实验建议先设 `steps: 1000` 验证 loss、显存、checkpoint 和视频键，再恢复正式训练步数。服务器端同步源码、`pixi.lock` 和数据集，执行同一命令即可复现环境；不要同步 `.pixi/`。

## 7. 如何系统扩大数据多样性

按下列顺序扩展，且每次只改变一组因素并记录配置：

1. **任务几何**：工件 XY/yaw、焊缝长度、焊接方向、直线/折线/曲线焊缝；这是最重要的任务分布。
2. **机器人初态**：逐步扩大 `joint_degs` 和 `initial_tcp_m`，观察有效率、碰撞率和关节覆盖直方图。
3. **工艺条件**：随机化速度、工作角、行走角和工具滚转角，并同步改写自然语言指令，避免图像相同而指令标签互相矛盾。
4. **恢复数据**：改变恢复扰动的位置、方向和幅度，保持普通成功与恢复成功的比例可控。
5. **视觉域**：增加灯光、曝光、材质、背景、相机外参小扰动和遮挡物；几何标定误差与纯外观随机化应分开配置。
6. **模型域**：在实测误差范围内随机化关节零偏、TCP、控制延迟和摩擦，不要用不物理的大范围噪声掩盖标定问题。
7. **真机数据**：用相同绝对状态/命令 schema 记录人工示教和策略执行，再用少量高质量真机数据微调。

建议分阶段建设数据集：50 条用于链路检查，500 条用于基线和动作表示消融，数千条以上用于视觉域随机化和多任务训练。训练/验证/测试应按 episode 和场景 seed 划分，不能按帧随机切分，否则同一条轨迹会同时出现在训练集和验证集。

## 8. 采集质量检查清单

扩大采集前至少确认：

- `dataset.json` 达到目标有效 episode 数；
- 有效 episode 的碰撞率为零、IK 全程成功、焊缝进度达到阈值；
- 六个关节的 `initial_joint_offset_deg` 均有覆盖，而不是大量卡在关节限位；
- `scene_sampling_attempts` 和初态采样次数没有长期接近上限；
- 全局相机能同时看到机械臂、工件和焊缝，腕部相机不被焊枪完全遮挡；
- `safe_command`、`reference`、`executed` 三条轨迹的差异符合控制器行为；
- 训练动作的 source、frame、rotation、horizon 和 stride 固定写入实验配置。
