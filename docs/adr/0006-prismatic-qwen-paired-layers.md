# Prismatic-Qwen 采用成对层双流交织

Trajectory-VLA 的 Qwen 变体使用预对齐的 Prismatic Qwen2.5 checkpoint，并让 16 个 Qwen 层与 16 个轻量动作专家层逐层执行联合注意力，而不是在完整 VLM 后附加独立 cross-attention expert。该结构实现和调试成本更高，但保留了 SmolVLA/π 的层内上下文—动作交互与 prefix KV cache；默认冻结视觉和语言主干，只训练视觉 token merger、projector、状态/动作投影与动作专家，所有主干均可通过配置解冻。
