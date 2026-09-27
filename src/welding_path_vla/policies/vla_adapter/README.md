# VLA-Adapter

本目录把官方 VLA-Adapter 的 Prismatic-Qwen2.5-0.5B、ActionQuery 与逐层 Bridge
迁移到项目统一的 LeRobot 生命周期中。训练、断点恢复、离线评估、relative action 和仿真
rollout 均复用 `policies/` 公共实现。

## 实现口径

- 使用官方当前推荐的 Pro Bridge：动作 self-attention、`ActionQuery + state` cross-attention
  和带 `tanh(g)` gate 的 raw patch cross-attention 使用独立 K/V 投影与 RoPE。
- 三路 attention logits 与 Value 的拼接和上游 Pro 一致；不保留未启用的 FiLM，以及
  forward 内临时创建、无法被 optimizer 持久训练的随机 Parameter。
- Qwen chat prompt 在任务前加入上游训练使用的 `What action should the robot take to ...?` 提问。
- 默认 `action_head: l1`，从全零 action latent 直接回归完整动作块，与论文主实验一致。
- `action_head: flow_matching` 是本项目扩展：它把 noisy action 与时间编码送入同一个 Bridge，
  预测直线 Flow 的速度场，并复用项目现有的 Beta 时间采样和 Euler 积分定义。当前官方仓库并未
  发布这一 Flow 头，因此结果应标记为 “VLA-Adapter + Flow Matching”。

## 配置

```yaml
policy:
  family: vla_adapter
  parameters:
    action_head: l1          # 或 flow_matching
    num_action_queries: 64
    num_bridge_layers: 24
    bridge_num_heads: 8
    num_steps: 10            # 只影响 Flow 推理
```

`configs/vla_adapter.yaml` 是默认 L1 入口，`configs/vla_adapter_flow.yaml` 提供 Flow 示例，
`configs/vla_adapter_a100.yaml` 是双 A100 训练入口。首版从 Stanford Prismatic checkpoint
初始化，不导入面向 LIBERO/CALVIN 的 7D 动作头；当前焊接数据的 13D state 和 9D action
由 LeRobot feature schema 自动确定。

VLA-Adapter 策略本身不调用 `torch.compile`。旧 checkpoint 中的 `compile_model` 和
`compile_mode` 字段仅用于兼容配置加载，不再触发策略内部编译。

训练默认冻结视觉编码器和 Qwen 基座原权重，对 Qwen 和视觉编码器的线性层应用 rank-64
LoRA；ActionQuery、状态投影、完整 Bridge、动作头和 Prismatic projector 作为 checkpoint
的完整模块保存。`use_vision_lora: false` 可用于视觉冻结消融，不会关闭 Qwen LoRA。

训练配置默认启用 `training.image_aug: true`，复用 LeRobot 的亮度、对比度、饱和度、色调增强；
验证集和部署输入不增强。上游还使用随机缩放裁剪，但焊缝位置和相机几何对裁剪敏感，
这里不默认加入裁剪或仿射变换。
