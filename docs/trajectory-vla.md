# Trajectory-VLA 本地实现

`trajectory_vla` 以 LeRobot 0.6 官方 SmolVLA 为基线，但模型源码位于
`src/welding_path_vla/policies/trajectory_vla/`，不继承或调用官方 `SmolVLAPolicy`。
LeRobot 只负责数据集、processor、训练循环、checkpoint 和通用设备工具。

## 模块边界

```text
configuration_trajectory_vla.py  模型、flow、optimizer 和 scheduler 配置
smolvlm_action_expert.py         SmolVLM + 动作专家双流 Transformer
flow_matching.py                context/action token、训练插值和 Euler 去噪
modeling_trajectory_vla.py       LeRobot policy 与公开研究接口
processor_trajectory_vla.py      tokenizer 及 feature 处理
```

训练、运行时、离线评估和 robosuite rollout 复用上一级 `policies/` 公共模块。模型差异集中
声明在 `policies/spec.py` 的 `TRAJECTORY_VLA`，算法目录不复制生命周期代码。

## 研究接口

- `SmolVLMActionExpert.embed_image()`：视觉编码和视觉 token；
- `SmolVLMActionExpert.forward()`：上下文流与动作专家层间融合；
- `TrajectoryFlowModel.encode_context()`：图像、语言和状态上下文；
- `TrajectoryFlowModel.encode_action_tokens()`：带噪轨迹与时间表示；
- `TrajectoryFlowModel.predict_velocity()`：flow velocity；
- `TrajectoryFlowModel.denoise_step()`：单步 Euler 修正；
- `TrajectoryVLAPolicy.forward_intermediates()`：训练噪声、时间、目标和逐维 loss；
- `TrajectoryVLAPolicy.predict_action_chunk(on_step=...)`：观察每个去噪步骤。

本地实现能够加载 `lerobot/smolvla_base`。配置中的 `load_vlm_weights: false` 表示加载完整
policy checkpoint 时不再重复加载裸 VLM，并不表示丢弃 checkpoint 内的 VLM 权重。

## 训练与恢复

```bash
pixi run -e train policy-data-check --config_path=configs/trajectory_vla.yaml
pixi run -e train train-policy \
  --config_path=configs/trajectory_vla.yaml \
  --dry_run=true
pixi run -e train train-policy --config_path=configs/trajectory_vla.yaml

# 双 A100
pixi run -e train train-policy-2gpu \
  --config_path=configs/trajectory_vla_a100.yaml
```

中断后保持输出目录，并保证总 step 大于 checkpoint step：

```bash
pixi run -e train train-policy \
  --config_path=configs/trajectory_vla.yaml \
  --training.resume=true \
  --training.steps=250000
```

## 评估与部署

```bash
pixi run -e train policy-evaluate \
  --config_path=configs/trajectory_vla.yaml \
  --policy.checkpoint=CHECKPOINT/pretrained_model

pixi run -e policy-sim policy-sim-deploy \
  --config_path=configs/deploy/trajectory_vla_curve_plate.yaml
```

部署目录提供 `l_joint`、`pipe_bottom`、`pipe_top`、`curve_plate`、
`trihedral_horizontal` 和 `trihedral_vertical` 六个入口，它们共享
`configs/deploy/trajectory_vla.yaml` 中的 checkpoint。

基于 Prismatic + Qwen2.5 的另一条研究分支位于
[`policies/traj_vla_qwen/README.md`](../src/welding_path_vla/policies/traj_vla_qwen/README.md)。
该模型使用不同视觉主干和成对层交织结构，不应与本页的 SmolVLA 重写混为同一实现。
