# 2026-10-19 虚拟控制器采集与 Trajectory-VLA 训练验收

结论：真实 TCP/SDK 读取 → 原生 LeRobot 录制 → 数据整理 → 项目 Trajectory-VLA
训练与 checkpoint 保存已跑通。数据格式与训练链路合格；严格 30 Hz 时序未通过，
真实视觉质量、物理安装坐标和完整路径边界未验收。这批数据仅用于链路测试。

2026-10-20 编码修正说明：正式与 VM 采集配置后来均已统一为 AV1。
本报告中的既有 VM 数据仍是当时录制的 H.264，未被改写；按新版配置重新采集会使用 AV1。

## 测试条件

- 控制器由用户在虚拟机启动，地址 `192.168.56.101:10003`，持续自行运动。
- 本机仅执行连接、读取和断开；没有发送机械臂运动指令。
- `global`、`wrist` 使用显式 `black` 后端，各输出 640×480 uint8 RGB 全黑图像。
- 使用现有 Pixi `dev` 采集、`train` 训练；未创建 `real` 环境。
- 本地采集插件在 `dev` 中通过 Pixi 离线注册；模型使用已有缓存权重，训练禁用 Hub 网络和 WandB。
- 标签包含 `task.extra.test_only=true`、`task.extra.camera_source=black_placeholder`。
  未知运动方向没有填入 forward/reverse，也未虚构工艺参数或阶段标签。

## 数据检查

数据位于 `datasets/real/vm_black_smoke_20261019`，总计 **3 个 episode、1,253 帧**，
包含数据和会话日志约 **1.37 MB**。每条记录为 15 秒墙钟时间截取，不保证恰好覆盖完整路径。

| episode | 帧数 | 文件时间轴长度（帧数 / 30） | 实测首尾查询跨度 | 实测采样率 |
|---|---:|---:|---:|---:|
| 0 | 417 | 13.90 s | 14.981 s | 27.77 Hz |
| 1 | 419 | 13.97 s | 14.967 s | 27.93 Hz |
| 2 | 417 | 13.90 s | 14.972 s | 27.78 Hz |

已逐帧检查：

- 状态维度 13，动作维度 9；数值全部有限。
- 状态四元数为单位四元数，动作 rotation-6D 两行向量单位正交。
- 机器人持续运动：TCP 的 XYZ 范围分别约 174.35、150.61、116.44 mm，状态不是固定假数据。
- Parquet 中每帧状态/动作与 SDK 会话日志一致，episode 内 frame_index 连续。
- 两路视频各完整解码 1,253 帧，尺寸和 RGB 格式正确，所有帧均为全黑占位图像。
- 整理完成并生成 horizon=30、stride=1 的 relative action 统计，现有数据检查及训练入口能读取。

实际平均采样率 **27.827 Hz**，查询间隔 P50/P95/P99 为 **35.55/36.60/50.58 ms**。
原生 Dataset 时间戳按配置的 `frame_index / 30` 生成，没有表示实际 27.827 Hz 的时间轴。
当前采样方式在等待约 33.42 ms 后读取后继状态，再开始下一个独立观测周期；
SDK 查询与写入开销使周期超过 33.33 ms。该差异会影响动作时间语义，不能仅凭 schema
检查通过就视为正式 30 Hz 数据。后续需调整采样调度并重新采集验证；本次保留原始数据和日志。

## 模型训练

模型是项目内实现的 **Trajectory-VLA**，初始化自缓存的 `lerobot/smolvla_base`，
不是 ACT，也没有替换成缩小的假模型。

- 总参数 450,046,176，可训练参数 99,880,992。
- batch=1，预测 horizon=30，双图像缩放至 256×256，BF16，PyAV 解码。
- 使用 3 个 episode 的全部数据作为训练来源，不启用 phase sampling 或验证集。
- 完成 3 次前向、反向、优化器更新；GPU 日志显存约 1.94 GB。

| step | loss | 梯度范数（裁剪前） |
|---|---:|---:|
| 1 | 1.205 | 16.762 |
| 2 | 14.523 | 91.513 |
| 3 | 3.133 | 34.559 |

损失与梯度均有限。保存后的 `model.state_proj.weight` 相对初始化权重有 **12,480** 个元素改变，
最大绝对变化 **0.00030255**，确认发生了实际参数更新。
3 步训练不用于判断收敛或任务能力，黑帧也不能证明视觉策略有效。

checkpoint 位于：
`outputs/train/vm_black_smoke_20261019/checkpoints/000003/pretrained_model/`。
模型权重、处理器及优化器状态均已保存。

## 可复现配置与证据

- 采集配置：`configs/real_record_vm_smoke.yaml`。
- 训练配置：`configs/real_training_vm_smoke.yaml`。
- 原始录制日志：`outputs/validation/vm_black_smoke_20261019/record.log`。
- 数据量、时序和图像检查：同目录 `capture_validation.json`。
- 数据契约检查：同目录 `policy_data_check.json`。
- 训练计划：同目录 `train_plan.json`。
- 训练参数更新检查：同目录 `training_validation.json`。
- 训练日志：`outputs/train/vm_black_smoke_20261019/train.log`。

```bash
pixi run --no-install --frozen -e dev record-real --config_path=configs/real_record_vm_smoke.yaml
pixi run --no-install --frozen -e dev prepare-real-data --dataset=datasets/real/vm_black_smoke_20261019 --require_task_labels=false
pixi run --no-install --frozen -e dev policy-data-check --config_path=configs/real_training_vm_smoke.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 \
  pixi run --no-install --frozen -e train train-policy --config_path=configs/real_training_vm_smoke.yaml
```

以上目录已经有验收结果。重新录制应通过 `dataset.root` 指定一个新目录，
重新训练也应更换 `training.output_dir`，以保留这次结果。
