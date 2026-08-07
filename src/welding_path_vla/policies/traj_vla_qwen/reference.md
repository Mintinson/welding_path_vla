# 模型转换与兼容性分析


你说得对。上一条回答把两个概念混在了一起：

1. `Qwen2.5-0.5B` 本身是纯文本 LLM；
2. 但它完全可以作为一个自定义 VLM 中的 **LLM backbone**。

SG-VLA 做的正是第二种，而不是直接把原始 `Qwen2.5-0.5B` 当作完整 VLM 使用。

# 一、SG-VLA 实际采用的是什么结构

SG-VLA 第 3～4 页给出的 VLM 是一个 **Prismatic 风格的组合式 VLM**：

$$
\boxed{
\text{VLM}
=
\text{DINOv2}
+
\text{SigLIP}
+
\text{Vision Projector}
+
\text{Qwen2.5-0.5B}
}
$$

具体分工是：

- DINOv2：提供更强的空间和局部几何特征；
- SigLIP：提供视觉-语义对齐特征；
- Projector：把视觉特征映射到 Qwen 的 token embedding 空间；
- Qwen2.5-0.5B：融合视觉 token 与语言 token，并生成多模态隐表示；
- 可选 Flow Matching Action Expert：从 VLM 隐表示生成连续动作。

论文中的 Figure 2 虽然标成了 `Qwen2.5VL-0.5B`，但正文明确说明它是由双视觉编码器、可训练 Projector 和 Qwen2.5-0.5B 语言主干组合成的 Prismatic VLM，而不是阿里官方提供的一个 0.5B Qwen2.5-VL checkpoint。`Tu et al. - 2026 - SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation.pdf`

它基本继承了 MiniVLA 的路线。MiniVLA 已经公开了一个完成视觉-语言预训练的 Prismatic checkpoint：

```text
Stanford-ILIAD/
prism-qwen25-extra-dinosiglip-224px-0_5b
```

该模型使用 DINOv2 + SigLIP 视觉编码器和 Qwen2.5-0.5B 语言主干，还额外加入了 256 个 token，供后续动作离散化使用。([GitHub](https://github.com/Stanford-ILIAD/openvla-mini))

因此，你真正想做的不是：

```text
SmolVLM2 → 原始 Qwen2.5-0.5B
```

而是：

```text
SmolVLM2
    ↓ 替换为
Prismatic-Qwen VLM
    =
DINOv2 + SigLIP + Projector + Qwen2.5/Qwen3
```

然后把 SmolVLA 的轻量化设计和 Flow Matching Action Expert 移植到这个新 VLM 上。

这个方向是可行的。

---

# 二、推荐的整体架构

建议设计成下面的结构：

```text
全局 RGB ─────┐
腕部 RGB ─────┼──> DINOv2 + SigLIP
可选 Depth ───┘          │
                         ▼
              双视觉特征融合
                         │
              Visual Token Merger
                         │
                    Projector
                         │
语言指令 ───────────> Qwen2.5 / Qwen3
                         │
机器人状态 ──> State Projector
工艺约束 ────> Constraint Projector
                         │
                         ▼
             Multimodal Context Tokens
                         │
             Flow Matching Action Expert
          Cross-Attention + Causal Self-Attention
                         │
                         ▼
             焊缝坐标系残差动作块
```

数学上，可以写成：

## 视觉编码

对于第 $c$ 个相机：

$$
Z_c^{D}
=
E_{\mathrm{DINO}}(I_c),
$$

$$
Z_c^{S}
=
E_{\mathrm{SigLIP}}(I_c).
$$

沿特征维度融合：

$$
Z_c^{V}
=
\operatorname{Concat}
\left(
Z_c^{D},
Z_c^{S}
\right).
$$

经过 token 压缩和 Projector：

$$
\tilde Z_c^V
=
P_V
\left(
M_{\mathrm{token}}(Z_c^V)
\right),
$$

其中：

- $M_{\mathrm{token}}$：视觉 token 压缩器；
- $P_V$：映射到 Qwen hidden dimension 的 Projector。

## 文本和状态

语言 token：

$$
Z^L
=
E_{\mathrm{Qwen}}(\ell).
$$

机器人状态：

$$
z^s
=
W_s s_t+b_s.
$$

工艺条件：

$$
z^c
=
W_c c_t+b_c,
$$

其中可以包括：

$$
c_t=
[
v_d,\alpha_d,\beta_d,d_{\mathrm{safe}}
].
$$

输入 Qwen 的序列为：

$$
X=
[
\tilde Z_1^V;
\tilde Z_2^V;
Z^L;
z^s;
z^c
].
$$

经过截断后的 Qwen：

$$
C_t
=
F_{\mathrm{Qwen}}^{1:N}(X).
$$

最后交给 Flow Matching Expert：

$$
\hat v_\theta
=
v_\theta
\left(
A_t^\tau,
C_t,
\tau
\right).
$$

这就是比较完整的“Qwen-SmolVLA”。

---

# 三、SmolVLA 的哪些轻量化策略可以直接迁移

大部分核心策略都能迁移，但需要区分“思想可迁移”和“代码可直接复用”。

## 1. 视觉 token 压缩：可以迁移，而且很重要

Prismatic 的 224 像素视觉编码器通常会产生较多 patch token。双相机再加 DINOv2 和 SigLIP 后，视觉部分可能成为主要计算负担。

Prismatic 的实现方式是分别运行 DINOv2 和 SigLIP，然后沿 hidden dimension 拼接二者特征，再通过多层 Projector 映射到 LLM 空间。([GitHub](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py))

你可以在 Projector 之前加入 SmolVLA 风格的空间压缩：

$$
Z^V
\in
\mathbb R^{H_p\times W_p\times d_v}
$$

通过 $2\times2$ token merging：

$$
\tilde Z_{i,j}^V
=
\operatorname{MLP}
\left(
[
Z_{2i,2j}^V;
Z_{2i+1,2j}^V;
Z_{2i,2j+1}^V;
Z_{2i+1,2j+1}^V
]
\right).
$$

token 数量缩小为原来的：

$$
\frac14.
$$

例如每个相机原来有约 256 个 token，压缩后变为约 64 个：

$$
256\rightarrow64.
$$

双相机则从约：

$$
512\rightarrow128.
$$

这与 SmolVLA 的视觉 token 数量接近。

需要注意，焊缝是细长目标，不能只做简单平均池化。更推荐：

- Pixel Shuffle + Linear；
- 2D Token Merger；
- Perceiver Resampler；
- 基于 64 个可学习 query 的 Cross-Attention Resampler。

其中 Token Merger 或 Perceiver Resampler 更适合你的焊缝任务。

---

## 2. 截断 Qwen 层：可以迁移，但要重新做消融

Qwen2.5-0.5B 有：

$$
L=24
$$

层，hidden size 为：

$$
d=896.
$$

([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blame/main/config.json?utm_source=chatgpt.com))

可以测试：

$$
N\in\{8,12,16,20,24\}.
$$

第一版建议：

$$
\boxed{N=12\text{ 或 }16}
$$

而不是直接只保留 12 层并认定效果一定最好。

Qwen3-0.6B 有：

$$
L=28,\qquad d=1024,
$$

并使用 16 个 Query Head、8 个 KV Head。([Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B))

可以测试：

$$
N\in\{14,18,22,28\}.
$$

需要注意：MiniVLA 的 Prismatic-Qwen2.5 VLM 是在完整 Qwen 上完成视觉-语言对齐的。直接删除后半层后，中间特征不一定天然适合作为动作条件。因此更稳妥的顺序是：

1. 完整 24 层验证模型；
2. 取不同中间层特征；
3. 训练 Action Expert；
4. 再进行物理截断；
5. 对保留层和 Projector 做少量 LoRA 或联合微调。

---

## 3. 缩窄 Action Expert：可以直接迁移

SmolVLA 将 Action Expert hidden dimension 设置为 VLM 的约 $0.75$ 倍。

对于 Qwen2.5：

$$
d_{\mathrm{expert}}
=
0.75\times896
=
672.
$$

对于 Qwen3：

$$
d_{\mathrm{expert}}
=
0.75\times1024
=
768.
$$

因此可以分别使用：

```text
Qwen2.5:
expert_hidden_size = 672

Qwen3:
expert_hidden_size = 768
```

也可以为了 GPU 友好，统一取：

```text
Qwen2.5: 640 或 704
Qwen3:   768
```

Action Expert 不需要具备完整语言模型宽度，因为其主要任务只是：

$$
\text{多模态条件}
+
\text{噪声动作块}
\rightarrow
\text{动作矢量场}.
$$

---

## 4. Flow Matching：可以迁移

这一部分与具体 VLM 基本无关：

$$
A^\tau
=
\tau A+(1-\tau)\epsilon,
$$

$$
\mathcal L_{\mathrm{FM}}
=
\left\|
v_\theta(A^\tau,C_t,\tau)
-
(\epsilon-A)
\right\|_2^2.
$$

只要 Qwen VLM 能输出：

$$
C_t\in\mathbb R^{B\times L_c\times d_c},
$$

就可以经过一个 Context Projector：

$$
\tilde C_t
=
W_C C_t
$$

送入 Action Expert。

SG-VLA 自己也采用了一个可选的约 100M 参数 Flow Matching Expert，并在第三阶段冻结 VLM、单独训练动作头。`Tu et al. - 2026 - SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation.pdf`

---

## 5. Cross-Attention / Causal Self-Attention：可以迁移

Action Expert 内部仍然可以保持：

```text
Cross-Attention
      ↓
Causal Self-Attention
      ↓
Cross-Attention
      ↓
Causal Self-Attention
```

其中：

$$
Q=H_AW_Q,
\qquad
K=C_tW_K,
\qquad
V=C_tW_V,
$$

$$
H_A'
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V.
$$

Cross-Attention 负责读取 Qwen 的多模态上下文，Causal Self-Attention 负责动作块内部的时序一致性。

这一部分不要求 Qwen 本身和 Action Expert 使用同一种 Attention 实现。

---

## 6. 异步推理：可以原样保留

异步执行与 VLM 类型无关：

```text
Qwen-VLA 正在推理下一动作块
              ∥
机器人正在执行当前动作块
```

只需要满足：

$$
g
\geq
\frac{\ell}{K\Delta t},
$$

其中：

- $K$：动作块长度；
- $\Delta t$：动作步时间；
- $\ell$：模型端到端推理延迟；
- $g$：提前触发推理时的队列比例。

SmolVLA 官方结果中，异步执行主要降低任务时间并提高吞吐量，而不是改变模型本身。([Hugging Face](https://huggingface.co/blog/smolvla))

---

# 四、哪些代码不能直接复用

虽然方法可以迁移，但 LeRobot 中现有的 `SmolVLMWithExpertModel` 不能直接挂载 Prismatic-Qwen。

主要需要重写以下部分。

## 1. 图像处理

SmolVLA 当前依赖：

```python
SmolVLMProcessor
SmolVLM image token
SmolVLM connector
```

Prismatic-Qwen 使用：

```text
DINOv2 transform
SigLIP transform
channel-wise image packing
Prismatic projector
Qwen tokenizer
```

多图像输入时，Prismatic 代码还会分别运行两个视觉编码器，再拼接每个相机的 patch token。([GitHub](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py))

因此要新建：

```python
PrismaticQwenProcessor
```

而不能复用 `SmolVLMProcessor`。

## 2. VLM wrapper

建议定义统一接口：

```python
class MultimodalBackbone(nn.Module):
    def encode_context(
        self,
        images,
        input_ids,
        attention_mask,
        robot_state,
        constraints,
    ) -> torch.Tensor:
        ...
```

然后分别实现：

```python
SmolVLMBackbone
PrismaticQwen25Backbone
PrismaticQwen3Backbone
```

Action Expert 只依赖 `encode_context()` 的输出，不依赖具体 VLM 内部类名。

这是最重要的工程解耦。

## 3. 位置编码和 KV Cache

SmolVLA 当前实现会直接进入 SmolLM2 的每个 Transformer block，并缓存各层 VLM 的 K/V。

Prismatic-Qwen2.5 使用 Qwen2 的：

- GQA；
- QKV bias；
- RoPE；
- 14 个 Query Head；
- 2 个 KV Head。

([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B?utm_source=chatgpt.com))

Qwen3 则是：

- 16 个 Query Head；
- 8 个 KV Head；
- Q/K normalization；
- 28 层。

因此不能直接复用原 SmolVLA 的 head reshape、RoPE 和 cache 代码。

---

# 五、应该采用“解耦式”还是“逐层交织式”

这里有两个实现层级。

## 方案 A：解耦式 Qwen-SmolVLA

```text
Prismatic-Qwen VLM
        ↓
一次性生成 Context Tokens
        ↓
独立 Action Expert
        ↓
Flow Matching
```

即：

$$
C_t
=
F_{\mathrm{Qwen}}(I,\ell),
$$

$$
\hat v
=
v_\theta(A^\tau,C_t,\tau).
$$

### 优点

- 最容易实现；
- Qwen 和 Expert 完全解耦；
- 可以缓存 $C_t$，十次去噪不重复运行 Qwen；
- 容易支持 Qwen2.5 和 Qwen3；
- 容易加入辅助 query 和轨迹损失；
- 真机部署更稳定。

### 缺点

不完全复刻 SmolVLA 中 VLM 层与 Expert 层的逐层对应关系。

### 工作量估计

在现有 MiniVLA/Prismatic checkpoint 和 SmolVLA Expert 代码基础上：

$$
\boxed{\text{约 3～5 周完成稳定原型}}
$$

这是我最推荐的第一版。

---

## 方案 B：逐层交织式 Qwen-SmolVLA

类似原始 SmolVLA：

```text
Qwen Layer 1  ↔ Expert Layer 1
Qwen Layer 2  ↔ Expert Layer 2
Qwen Layer 3  ↔ Expert Layer 3
...
```

每层都联合处理：

- Qwen context；
- Action token；
- Cross-Attention；
- KV cache。

### 优点

- 更接近原始 SmolVLA；
- 能研究不同 Qwen 层与动作层的交互；
- 可以形成更强的架构贡献。

### 缺点

- 需要重写 Qwen2/Qwen3 attention；
- cache、GQA、RoPE、Q/K Norm 都需要处理；
- 截断层数与 Expert 层数需要重新配对；
- 更难兼容 Hugging Face Transformers 更新；
- 更难调试训练不收敛问题。

### 工作量估计

Qwen2.5：

$$
\boxed{\text{约 1～2 个月}}
$$

Qwen3：

$$
\boxed{\text{约 2～3 个月}}
$$

这些是工程估计，不包括大规模实验和真机采集。

对你当前课题，更合理的是先完成方案 A，再把方案 B 作为后续架构增强。

---

# 六、Qwen2.5 和 Qwen3 的难度差别

## Qwen2.5-0.5B：推荐先做

它最大的优势不是只比 Qwen3 小一点，而是已经有现成的多模态对齐权重：

```text
prism-qwen25-extra-dinosiglip-224px-0_5b
```

MiniVLA 官方仓库已经提供：

- Qwen2.5 Prismatic VLM 配置；
- 视觉-语言预训练 checkpoint；
- 多图像支持；
- VLA 训练配置；
- 额外动作 token；
- LIBERO 预训练模型。

([GitHub](https://github.com/Stanford-ILIAD/openvla-mini))

因此你可以跳过最昂贵的步骤：

$$
\text{视觉-语言基础对齐预训练}.
$$

你的工作可以直接从：

$$
\text{已对齐 Prismatic-Qwen2.5 VLM}
\rightarrow
\text{SmolVLA Flow Expert}
$$

开始。

## Qwen3-0.6B：可行，但需要先构造 VLM

目前找到的 MiniVLA 官方实现明确支持的是 Qwen2.5-0.5B，没有看到对应的 Prismatic-Qwen3-0.6B 预训练 checkpoint。Qwen3-0.6B 本身为 28 层、0.6B 的语言模型。([GitHub](https://github.com/Stanford-ILIAD/openvla-mini))

要构建 Qwen3 版本，需要：

1. 在 Prismatic 中增加 `Qwen3LLMBackbone`；
2. 使用 `Qwen3-0.6B-Base` 初始化；
3. 添加图像占位 token 和必要特殊 token；
4. 将视觉 Projector 输出维度改为 1024；
5. 进行视觉-语言对齐预训练；
6. 再进行机器人动作训练；
7. 最后添加 Flow Matching Expert。

由于 Qwen2.5 hidden size 是 896，Qwen3 是 1024，原 Projector 最后一层：

$$
W_P\in\mathbb R^{d_v\times896}
$$

不能直接用于：

$$
\mathbb R^{d_v\times1024}.
$$

可以部分继承 DINOv2、SigLIP 和 Projector 前层，但最后映射层必须重新初始化。

你的两张 A100 40GB 足以做机器人训练和 LoRA 适配，但不适合从头完整复现 SG-VLA/MiniVLA 那种 8 GPU 规模的多模态预训练。MiniVLA 的官方 VLM 预训练示例使用 8 GPU，SG-VLA 的实验也使用了 8 张 A100。([GitHub](https://github.com/Stanford-ILIAD/openvla-mini)) `Tu et al. - 2026 - SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation.pdf`

所以研究顺序应当是：

$$
\boxed{
\text{先 Qwen2.5，后 Qwen3}
}
$$

而不是两条路线并行开发。

---

# 七、SG-VLA 的训练策略很值得直接借鉴

SG-VLA 发现，随机初始化的辅助解码器和 VLM 一起训练会产生较大的噪声梯度，导致性能下降。因此采用了三阶段训练。`Tu et al. - 2026 - SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation.pdf`

这个策略非常适合你的模型。

## Stage 1：辅助头适配

冻结 VLM：

$$
\nabla_{\theta_{\mathrm{VLM}}}\mathcal L_{\mathrm{aux}}
=
0.
$$

只训练：

- 焊缝位置 decoder；
- 焊缝切向 decoder；
- 表面法向 decoder；
- 姿态 decoder；
- 焊缝进度 decoder；
- 分割 decoder。

让这些随机初始化模块先适应固定的 Qwen 特征。

## Stage 2：联合空间表征微调

开放：

- Projector；
- Qwen LoRA；
- 部分视觉 Encoder LoRA；
- 辅助 decoder。

联合损失：

$$
\mathcal L_{\mathrm{repr}}
=
\lambda_{\mathrm{seg}}\mathcal L_{\mathrm{seg}}
+
\lambda_t\mathcal L_{\mathrm{tangent}}
+
\lambda_n\mathcal L_{\mathrm{normal}}
+
\lambda_p\mathcal L_{\mathrm{progress}}
+
\lambda_e\mathcal L_{\mathrm{offset}}.
$$

## Stage 3：独立训练 Flow Matching Expert

冻结整个 VLM：

$$
\nabla_{\theta_{\mathrm{VLM}}}
\mathcal L_{\mathrm{FM}}
=
0.
$$

只训练：

$$
\theta_{\mathrm{expert}}.
$$

SG-VLA 明确报告，并行训练 Flow Matching Expert 和其他目标时去噪损失难以收敛，所以最终单独冻结 VLM 训练动作头。`Tu et al. - 2026 - SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation.pdf`

这比一开始把：

$$
\mathcal L_{\mathrm{FM}}
+
\mathcal L_{\mathrm{seg}}
+
\mathcal L_{\mathrm{pose}}
+
\mathcal L_{\mathrm{smooth}}
$$

全部混在一起训练稳妥得多。

---

# 八、SG-VLA 的 Flow Matching 结果也要客观看待

SG-VLA 的 Flow Matching 配置是：

- Action chunk：8；
- 每次执行前 2 个动作；
- 去噪步骤：10。

它将 Pick 成功率从：

$$
0.13\rightarrow0.27
$$

将 Place 从：

$$
0.70\rightarrow0.80,
$$

但整体平均成功率从：

$$
0.73\rightarrow0.69,
$$

因为 Open/Close 等偏移动和机构操作任务表现下降。`Tu et al. - 2026 - SG-VLA Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation.pdf`

所以这篇论文并没有证明：

> Qwen2.5 + Flow Matching 必然全面优于离散动作。

它证明的是：

> 连续动作 Expert 对精细操作更有利，但对不同动作类型可能存在不同适用性。

你的焊接任务是连续、高精度、姿态约束明显的轨迹生成问题，更接近它受益的 Pick/Place 精细控制，而不是移动底盘的粗粒度动作。因此继续使用 Flow Matching 是合理的；但这是基于任务属性做出的推断，需要通过你的消融实验验证。

---

# 九、对你的具体实现建议

建议将新模型暂时命名为：

```text
Qwen-WeldVLA
```

第一版采用：

```text
Vision:
    DINOv2 + SigLIP
    224 × 224
    2 cameras
    2×2 token merger
    64 tokens / camera

LLM:
    Prismatic Qwen2.5-0.5B
    first 12 or 16 layers
    frozen initially

State:
    TCP pose
    joint position
    joint velocity
    previous action tail

Constraints:
    desired speed
    work angle
    travel angle
    safety clearance

Action Expert:
    hidden size = 672
    interleaved CA / causal SA
    flow steps = 10
    chunk size = 16
    execute prefix = 2-4

Output:
    seam-frame residual SE(3) trajectory chunk
```

动作定义为：

$$
\delta\hat{\boldsymbol\xi}_{t,k}^{S}
\in\mathbb R^6,
$$

最终目标：

$$
\hat{\mathbf T}_{E,t+k}^{B}
=
\mathbf T_{\mathrm{ref}}^{B}(s_{t+k})
\operatorname{Exp}
\left(
\widehat{
\delta\hat{\boldsymbol\xi}_{t,k}^{S}
}
\right).
$$

辅助 query 预测：

$$
\hat{\mathbf t}_{t,k},
\quad
\hat{\mathbf n}_{t,k},
\quad
\hat e_{y,t,k},
\quad
\hat e_{z,t,k},
\quad
\hat\gamma_{t,k}.
$$

分别表示：

- 焊缝切向；
- 表面法向；
- 横向偏差；
- 高度偏差；
- 焊缝进度。

---

# 十、最终推荐路线

## 第一阶段：Qwen2.5 版本

直接使用已有的：

```text
prism-qwen25-extra-dinosiglip-224px-0_5b
```

不要从原始：

```text
Qwen/Qwen2.5-0.5B
```

开始重新做视觉对齐。

然后完成：

1. Prismatic VLM 接口；
2. 视觉 token 压缩；
3. Qwen 层截断；
4. 独立 SmolVLA Action Expert；
5. 焊缝残差动作；
6. 辅助 decoder；
7. 分阶段训练；
8. 异步真机执行。

## 第二阶段：Qwen3 版本

在 Qwen2.5 整体管线稳定后，只替换：

```text
Qwen2.5-0.5B
→
Qwen3-0.6B-Base
```

完成视觉-语言对齐后，再比较：

- 指令组合泛化；
- 中文指令理解；
- 视觉空间表征；
- 推理延迟；
- 显存；
- 轨迹精度。

## 研究价值

单纯把 SmolVLM 换成 Qwen2.5 不足以成为主要创新，因为 MiniVLA、SG-VLA 和 VLA-Adapter 已经采用 Prismatic-Qwen2.5 路线。MiniVLA 已公开对应 VLM，VLA-Adapter 也明确以该 checkpoint 作为其 Qwen2.5-0.5B VLM。([GitHub](https://github.com/Stanford-ILIAD/openvla-mini))

更合适的创新定位是：

> **将 SmolVLA 的视觉 token 压缩、VLM 截断、轻量 Flow Matching Expert 和异步执行机制迁移到空间增强的 Prismatic-Qwen VLM，并面向工业连续轨迹任务设计焊缝坐标系残差动作与几何辅助监督。**

也就是：

$$
\boxed{
\text{Prismatic-Qwen 空间表征}
+
\text{SmolVLA 轻量连续动作生成}
+
\text{焊缝特定几何约束}
}
$$

这个方向技术上成立，也比“单纯替换 VLM”更容易形成清晰的论文贡献。

---

**Sources:**

- [GitHub - Stanford-ILIAD/openvla-mini: OpenVLA: An open-source vision-language-action model for robotic manipulation. · GitHub](https://github.com/Stanford-ILIAD/openvla-mini)
- [config.json · Qwen/Qwen2.5-0.5B-Instruct at main](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blame/main/config.json?utm_source=chatgpt.com)
- [SG-VLA](https://arxiv.org/html/2603.22760v1#bib.bib16)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)



