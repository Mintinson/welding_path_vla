# Welding Path VLA

本上下文描述由已知焊缝几何产生机器人学习数据的焊接轨迹研究领域。

## Language

**Workpiece**:
带有自身局部坐标系、几何和一组可标注焊缝的待焊工件。
_Avoid_: Object, target block

**Seam**:
定义在 Workpiece 局部坐标系中的有方向曲线，以及生成焊枪姿态所需的表面几何。
_Avoid_: Path, line

**Welding Task**:
一次对 Seam 的方向、工艺参数和自然语言指令选择。
_Avoid_: Scene, job

**Reference Trajectory**:
由 Welding Task 按弧长和时间采样得到的目标 TCP 位姿序列。
_Avoid_: Ground truth action

**Command Action**:
专家在某个策略周期表达的期望 SE(3) 增量，即控制意图。
_Avoid_: Executed action

**Executed State**:
仿真或真机实际达到的关节与 TCP 状态；TCP 位姿在项目 world frame 中表达。
_Avoid_: Command

**Episode**:
同一 seed 和 Welding Task 下，从初态到终态的一段完整、可验证时序记录。
_Avoid_: Run, video

**Observation**:
同一策略时刻同步得到的多视角图像、机器人状态和 Welding Task 指令。
_Avoid_: Frame, sample

**Policy**:
把 Observation 映射为一个或多个 Command Action 的可训练决策模型，不直接访问仿真器或真机驱动。
_Avoid_: Controller, model runner

**Prismatic Backbone**:
由 DINOv2、SigLIP、视觉投影器和 Qwen 语言模型组成的多模态上下文编码主干。
_Avoid_: Qwen-VL, vision encoder

**Context Stream**:
在逐层交织模型中承载多视角视觉、任务语言和机器人状态 token 的隐藏状态流。
_Avoid_: Prefix model, VLM output

**Action Expert Stream**:
在逐层交织模型中承载带噪短时动作轨迹和 Flow Matching 时间条件的隐藏状态流。
_Avoid_: Action head, controller

**Paired-Layer Interleaving**:
每个保留的语言模型层与一个动作专家层成对，通过同一次联合注意力交换信息，同时保留各自的投影、残差和 MLP 参数。
_Avoid_: Cross-attention head, alternating layers

**Robot Interface**:
仿真和 Elfin5-Pro 真机共同遵守的 SI 单位状态读取与命令边界，并负责应用真机基座到项目 world frame 的安装外参。
_Avoid_: SDK, backend

**Safety Gate**:
位于 Policy 与真机命令之间、拒绝越界状态或动作的强制检查边界。
_Avoid_: Safety policy, clamp

**Evaluation Report**:
由 Episode 的轨迹、碰撞和任务完成情况计算出的可比较质量结论。
_Avoid_: Score, result
