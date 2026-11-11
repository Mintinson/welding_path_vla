# Welding Path VLA

本上下文描述焊接轨迹研究中由仿真与真机示范产生机器人学习数据的领域。项目全程不开弧，研究目标是焊枪沿焊缝的运动学习与复现。

## Language

**Workpiece**:
带有自身局部坐标系、几何和一组可标注焊缝的待焊工件。
_Avoid_: Object, target block

**Seam**:
定义在 Workpiece 局部坐标系中的有方向曲线，以及生成焊枪姿态所需的表面几何。
_Avoid_: Path, line

**Welding Task**:
一次由自然语言指令定义、沿 Seam 运动的不开弧轨迹任务，可附带方向、速度和焊枪姿态要求。
_Avoid_: Scene, job

**Reference Trajectory**:
由 Welding Task 按弧长和时间采样得到的目标 TCP 位姿序列。
_Avoid_: Ground truth action

**Command Action**:
专家在某个策略周期表达的期望 SE(3) 增量，即控制意图。
_Avoid_: Executed action

**Relative TCP Action Chunk**:
策略根据一次 Observation 预测的有序 TCP 目标序列。块内各目标共享该观测的实测 TCP 位姿作为锚点，增量均在锚点的 TCP 局部坐标系中表达。
_Avoid_: Chained TCP deltas, joint trajectory

**Executed State**:
仿真或真机实际达到的关节与 TCP 状态；TCP 位姿在项目 world frame 中表达。
_Avoid_: Command

**Active TCP Calibration**:
真机控制器当前生效的法兰到工具中心点的位姿标定，是解释实测工具位姿的依据。
_Avoid_: Actual TCP pose, simulation default TCP

**Executed Action Target**:
被动示范中，以当前观测之后下一采样时刻的实际 TCP 位姿定义的动作监督目标；它描述实际运动结果，不代表外部控制器的控制意图。
_Avoid_: Command Action, reference trajectory

**Passive Demonstration Collection**:
仅观测并记录真机运动及多视角图像的示范采集；运动由外部程序、示教器或人工拖动产生，采集端不承担运动控制。
_Avoid_: Teleoperation, policy rollout

**Episode**:
同一 Welding Task 下，从初态到终态的一段完整、可验证时序记录，覆盖接近焊缝、沿缝运动和退出的完整路径。
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
