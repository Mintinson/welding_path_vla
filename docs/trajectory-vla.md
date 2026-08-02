# Trajectory-VLA 本地实现

`trajectory_vla` 以 LeRobot 0.6 官方 SmolVLA 为基线，但模型本身全部位于
`src/welding_path_vla/policies/trajectory_vla/`。它不会继承或调用
`lerobot.policies.smolvla.SmolVLAPolicy`；LeRobot 只负责通用的数据集、processor、
训练循环、checkpoint 和设备工具。

## 模块边界

```text
configuration_trajectory_vla.py  结构、flow、优化器和 scheduler 配置
smolvlm_action_expert.py         SmolVLM + 动作专家双流 Transformer
flow_matching.py                context/action token、训练插值和 Euler 去噪
modeling_trajectory_vla.py       LeRobot policy 与动作队列
processor_trajectory_vla.py      tokenizer、归一化和反归一化
```

训练、在线运行时、离线评估和 robosuite rollout 位于上一级 `policies/` 公共模块中；
`trajectory_vla/` 只保留该算法独有的模型代码。模型差异由
`policies/spec.py` 中的 `TRAJECTORY_VLA` 声明，不再复制整套 pipeline。

研究时优先修改公开接口：

- `SmolVLMActionExpert.embed_image()`：视觉编码或多尺度视觉 token；
- `SmolVLMActionExpert.forward()`：self/cross attention 与层间融合；
- `TrajectoryFlowModel.encode_context()`：图像、语言、状态及未来几何 token；
- `TrajectoryFlowModel.encode_action_tokens()`：轨迹与时间表示；
- `TrajectoryFlowModel.predict_velocity()`：速度场预测；
- `TrajectoryFlowModel.denoise_step()`：单步轨迹修正；
- `TrajectoryVLAPolicy.forward_intermediates()`：取得噪声、时间、插值动作、目标速度、
  预测速度和逐维 loss；
- `TrajectoryVLAPolicy.predict_action_chunk(on_step=...)`：记录每个 Euler 去噪步骤。

官方 `lerobot/smolvla_base` 的 500 个参数名和形状与本地实现完全一致；同权重、同 token
输入的 16 层双流输出也经过逐值验证。源码按 Apache-2.0 许可重写，并在相关文件保留
版权与来源说明。

## 训练

```bash
pixi run -e train policy-data-check --config_path=configs/trajectory_vla.yaml
pixi run -e train train-policy \
  --config_path=configs/trajectory_vla.yaml \
  --dry_run=true
pixi run -e train train-policy --config_path=configs/trajectory_vla.yaml
```

`configs/policies/trajectory_vla.yaml` 默认从官方完整 policy 权重初始化。配置中的
`load_vlm_weights: false` 仅避免在加载完整 policy 前重复加载一次裸 SmolVLM，不会跳过
checkpoint 中的 VLM 权重。日志始终写入
`outputs/train/trajectory_vla_weldpath_relative_v1/train.log`。

中断后把总步数调大并恢复：

```bash
pixi run -e train train-policy \
  --config_path=configs/trajectory_vla.yaml \
  --training.resume=true \
  --training.steps=10000
```

## 评估与任务切换

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/trajectory_vla.yaml \
  --policy.checkpoint=CHECKPOINT

pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/trajectory_vla_l_joint.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/trajectory_vla_pipe_bottom.yaml
pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/trajectory_vla_pipe_top.yaml
```

三个部署入口共享 `configs/deploy/trajectory_vla.yaml` 中的 checkpoint。修改一次即可
让三个工件任务共同切换模型；工件、焊缝、指令和输出目录由各自任务入口组合。
