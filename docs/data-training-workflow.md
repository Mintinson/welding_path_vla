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

1. 每 10 个 episode 编号围绕标称值采样一组工作角、行走角、工具滚转角和姿态跟随比例；
2. 同组共享接近、焊接和退出速度，以及正反方向；圆管任务还共享几何起点和扫掠角；
3. 在 `xy_m / z_m / yaw_deg` 范围内随机化工件；
4. 求解工件上方的 staging 位姿，并拒绝不可达或碰撞场景；
5. 以 staging 关节构型为中心，按 `joint_degs` 对六个关节独立均匀采样；
6. 在 `initial_tcp_m` 范围内额外采样 TCP 平移；
7. 对完整参考轨迹逐帧执行连续 IK，拒绝跳解、限位、不可达和碰撞路径；
8. 预检失败时重新采样工件和初态，不创建无意义的视频文件；
9. 按概率插入轨迹中途恢复扰动；
10. 执行专家轨迹并保存质量结果，训练导出时只选择有效数据。

默认配置：

```yaml
randomization:
  xy_m: 0.05
  z_m: 0.0
  yaw_deg: 15.0
  joint_degs: [30, 10, 10, 15, 25, 25]
  max_sampling_attempts: 10
  initial_tcp_m: 0.03
  recovery_probability: 0.25
  recovery_position_m: 0.003
  recovery_rotation_deg: 2.0
  work_angle_range_deg: 3.0
  travel_angle_range_deg: 3.0
  tool_roll_range_deg: 5.0
  orientation_follow_range: 0.05
  arc_start_range_deg: 0.0
  arc_sweep_range_deg: 0.0
  approach_speed_range_mps: 0.005
  speed_range_mps: 0.002
  retreat_speed_range_mps: 0.005
  reverse_probability: 0.5
  task_group_size: 10
```

`joint_degs` 是六轴最大独立偏移，不是固定偏移。每条数据的实际偏移、完整运动重采样次数、轨迹预检最大 IK 残差和 TCP 偏移均写入 `metadata.json`。`quality.failure_reasons` 会明确列出进度、横向误差、姿态、碰撞或 IK 中未通过的条件；碰撞还记录持续帧数和具体几何对，避免只看到笼统的失败状态。

角度字段均表示相对任务 YAML 标称值的均匀采样半径，并取整到 1°；速度以 m/s 表示并保留三位小数；姿态跟随比例保留两位小数。`pipe_bottom` 当前将几何起点和扫掠角各改变 ±10°，焊接速度在 0.003–0.005 m/s 之间；`pipe_top` 始终执行完整 360°，起点改变 ±15°，姿态仅小幅变化。反向任务会交换同一几何圆弧的起终点，不会因为改变方向而换到另一段焊缝。

任务参数按全局 episode 编号分组，默认编号 0–9、10–19 分别共享一组参数。这种定义在多进程、增量采集和失败重试时仍然确定可复现。instruction 始终保持任务 YAML 中的固定文本；实际方向、姿态、圆弧和速度仅保存在 `task_parameters` 与 `resolved_config`。工件位姿、关节初始状态、TCP 偏移和恢复扰动仍逐条独立变化。发生轨迹碰撞的随机组合不能通过 episode 质量门，因此不会进入训练导出。

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

pixi run -e sim sim-collect \
  --config_path=configs/curve_plate.yaml \
  --collection.episodes=5

pixi run -e sim sim-collect \
  --config_path=configs/trihedral_vertical.yaml \
  --collection.episodes=5
```

各任务配置使用独立的数据集目录，不会相互混写。
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
  --collection.episodes=500 \
  --collection.workers=4
```

`collection.workers` 为 1 时保持顺序采集，适合断点调试；大于 1 时使用 `spawn` 多进程，每个进程独立持有 MuJoCo、EGL 和视频编码器。父进程分配唯一 episode 编号并汇总质量，只有当“已通过数量 + 正在执行数量”低于目标时才继续提交，因此不会因并发而额外多采有效 episode。消费级 GPU 建议从 2 开始，根据显存和 GPU 利用率逐步增加到 4；worker 过多会争用渲染 GPU、内存和视频编码 CPU，未必继续加速。交互式非 headless 模式会自动退回单进程。

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

默认配置只生成 LeRobot 视频 feature，不保留逐帧图片。转换器使用唯一的
LeRobot writer 顺序处理 episode，并由 LeRobot 在每个 episode 内并行编码两路相机。
这种方式会生成并及时删除临时图片，但编码进程结束后内存能够被系统完整回收，
适合数千条 episode 的长时间转换。最终数据使用 LeRobot 官方的
`libsvtav1 / CRF 30 / preset 12` 默认配置，在焊缝细节、文件体积和训练解码之间取得平衡。
需要快速人工预览时查看原始 H.264 即可，不必为了播放器兼容性改变训练数据编码。需要图片
feature 时显式添加 `--lerobot_export.save_images=true`，同一目标数据集不能混用视频和图片
schema。

一次合并 `datasets/` 下的所有 raw 数据集：

```bash
pixi run -e data export-lerobot \
  --dataset_glob='datasets/*_raw_v2' \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=YOUR_NAME/weldpath_relative_v1
```

`--datasets='[datasets/raw_a,datasets/raw_b]'` 可显式指定多个源。所有源按路径顺序
追加到同一目标，避免多个 writer 竞争 episode 索引、metadata 和系统内存。

### 上传到 Hugging Face Dataset Hub

先确认 data 环境已登录；令牌只保存在 Hugging Face 凭据库，不要写入 YAML：

```bash
pixi run -e data hf auth whoami
pixi run -e data hf auth login
```

在转换命令中打开 Hub 开关即可创建并上传 Dataset 仓库。默认创建私有数据集：

```bash
# 这是当前笔记本上已挂载、当前用户可写的数据盘；服务器上需替换为实际路径。
export WELDING_DATA_ROOT=/run/media/mintinson/DataDiskD/welding_path_vla
export WELDING_HF_REPO=mintinson/weldpath_relative_v1
mkdir -p "$WELDING_DATA_ROOT/lerobot"

HF_XET_HIGH_PERFORMANCE=1 pixi run -e data export-lerobot \
  --dataset_glob='datasets/*_raw_v2' \
  --output="$WELDING_DATA_ROOT/lerobot/weldpath_relative_v1" \
  --repo_id="$WELDING_HF_REPO" \
  --lerobot_export.push_to_hub=true
```

如果转换已经完成、仅 Hub 上传因网络中断失败，不要重新执行导出。直接上传现有
LeRobot 目录即可；该命令默认最多尝试 10 次，每次间隔 60 秒：

```bash
HF_XET_HIGH_PERFORMANCE=1 pixi run -e data upload-lerobot \
  --dataset="$WELDING_DATA_ROOT/lerobot/weldpath_relative_v1" \
  --repo_id="$WELDING_HF_REPO"
```

再次运行同一命令会复用 Hugging Face 的上传状态和仓库中已经存在的文件，不会
读取 raw episode，也不会重新编码视频。可通过 `--attempts` 和 `--retry_wait_s`
调整弱网重试策略。

不要对整条命令使用 `sudo`。`sudo` 会切换到 root 的 PATH 和凭据环境，
因此找不到安装在用户目录中的 Pixi，也不会沿用当前 Hugging Face
登录。如果其他磁盘的目录不可写，只在初始化该专用目录时调整所有者，
后续仍使用普通用户运行 Pixi。

公开数据集需要显式添加 `--lerobot_export.hub_private=false`。转换器会先
`finalize()` Parquet、视频索引和 metadata，再调用 LeRobot 的
`push_to_hub()`。Hugging Face Xet 上传本身会边读文件边上传，支持并行、去重和中断后重试。

转换和 Hub 上传不同时进行。LeRobot v3 在 `finalize()` 前的 Parquet footer
和元数据尚不完整，此时发布会得到无法稳定加载的中间数据集。因此
Hub 选项不会降低转换阶段的峰值磁盘占用；本地空间不足时，仍需把
`output` 放到容量足够的磁盘。

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

不建议对大数据集启用 `--lerobot_export.streaming_encoding=true`。本机使用
`libsvtav1` 实测时，流式线程编码器在 episode 之间持续保留大块内存：7 个
episode 的峰值 RSS 约为 4.13 GiB；默认的临时帧路径约为 1.28 GiB，且不随
episode 数量线性增长。`parallel_video_encoding=true` 仍会让 LeRobot 同时编码
双相机，不需要项目再建立外层 episode 进程池。

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
  dataset_root: /run/media/mintinson/DataDiskD/welding_path_vla/lerobot/weldpath_relative_v1
  output_dir: outputs/train/smolvla_v2
  batch_size: 16
  steps: 229627
  wandb: true
  wandb_mode: offline
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

DataDiskD 当前数据集包含 4250 个 episode、4,096,853 帧和 15 个任务。LeRobot 按任务
留出最后 10% episode 后，训练集实际为 3,674,023 帧。配置中的最大 step 以约一次完整
训练集遍历为目标：ACT 为 1,837,012，SmolVLA / Trajectory VLA 为 229,627，
Traj-VLA-Qwen 为 918,506，单卡 π0 / π0.5 为 3,674,023，双 A100 有效 batch 4 时为
918,506。数据继续增加后，应重新使用“训练帧数 ÷ 有效 batch size”计算，而不是沿用固定
的 5000 step。

W&B 默认以 offline 模式写入 `<training.output_dir>/wandb`，不会访问网络，也不会重复复制
大型 checkpoint。网络可用后执行：

```bash
pixi run -e train wandb sync outputs/train/NAME/wandb/offline-run-*
```

如需训练时直接在线显示，可临时添加 `--training.wandb_mode=online`。`train.log` 只保存
INFO 以上的配置、loss、学习率、吞吐、显存、评估和 checkpoint 信息，不再记录数据文件
锁和逐视频打开等 DEBUG 噪声。

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
