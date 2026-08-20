# ADR 0006：Prismatic-Qwen 使用成对层双流交织

状态：已接受

## 背景

Trajectory-VLA Qwen 需要让视觉语言上下文与 flow-matching 动作专家在每层交换信息，同时保留
以后替换 Qwen3 / Qwen3.5 的边界。

## 决策

使用预对齐 Prismatic Qwen2.5 主干，将 16 个 Qwen 层与 16 个轻量动作专家层一一配对，通过
联合注意力逐层交织，而不是在完整 VLM 后追加单独 cross-attention expert。上下文流不能读取
带噪动作，动作流可读取上下文并在推理时复用上下文 KV cache。

默认冻结视觉和语言主干，训练 token merger、projector、状态/动作投影和动作专家；所有主干
保留显式解冻、LoRA 和 gradient checkpointing 接口。

## 后果

结构更接近 SmolVLA / π 的层内交互并暴露研究接口，但实现、显存估算和 checkpoint 兼容更复杂。
模型细节只维护在 `src/welding_path_vla/policies/traj_vla_qwen/README.md`。
