# VLA与焊接机器人应用

**User:** Anonymous  
**Created:** 7/21/2026 11:30:58  
**Updated:** 7/22/2026 11:57:02  
**Exported:** 7/22/2026 16:49:27  
**Link:** [https://chatgpt.com/g/g-p-6a60221b8c908191895b0b1decc9ffcf/c/6a5ee7e9-b418-83ea-ab6f-df83d59331fb](https://chatgpt.com/g/g-p-6a60221b8c908191895b0b1decc9ffcf/c/6a5ee7e9-b418-83ea-ab6f-df83d59331fb)  

## Response:
7/21/2026, 7:44:56 PM

我会先核对论文和仓库里的数据组织方式，重点确认它到底保存了哪些原始时序、动作是离线如何构造的，以及这种做法是否适合你当前的“末端增量动作块”设定。

这篇工作的关键点正好支持你的想法：它并不是要求采集阶段就固定最终的动作块标签，而是保存整段按时间排列的末端绝对位姿；训练加载时再从当前帧抽取未来轨迹，并转换成相对当前末端坐标系的 delta chunk。这里有一个重要细节：它的“delta chunk”不是逐相邻帧作差。

> **Searching the web**
>
> [GitHub - Tencent-Hunyuan/Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack · GitHub](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)
> [[2606.14409] Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack](https://arxiv.org/abs/2606.14409)
>
> **Searched the web, GitHub, and arXiv**
>
> 
>
> **Cloning repository and extracting delta information**
>
>
> **Cloned repository and displayed log file contents**
>
> 
>
> **Searching the web**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/main/hy_vla/data/umi_dataset.py)
>
> **Searched the web and raw.githubusercontent.com**
>
> 
>
> **Considering trajectory methods**
>
> There are two possible ways to store data for training: 1) save full trajectory and compute actions dynamically; 2) log sim at high frequency and resample at 20Hz, adjusting for semantics. Both need raw data with time-stamps, poses, and states, but care must be taken to ensure the right reference for action labels.
>
> **Considered trajectory methods**
>
> 
>
> **Designing simulation approach**
>
> For tracking actions in a simulation, we could save the whole episode, but focus on precise, synchronized data like TCP poses, joint states, and images. Storing raw observations and task-specific commands will support better action tracking.
>
> **Designed simulation approach**
>
> 
>
> Worked for 3m 30s

## 可以，而且非常适合你的任务

你可以采用一种**动作表示无关的数据记录方式**：

> 采集阶段只保存完整、同步、尽可能原始的绝对轨迹；训练阶段再根据模型需要，动态生成末端相对位姿、逐步增量、动作块或焊缝坐标系残差。

这不仅可行，而且比采集阶段直接把某一种 `action` 写死更加合理。因为你后续很可能比较：

- 基坐标系动作与末端局部坐标系动作；
- 相邻帧增量与相对当前帧的未来轨迹；
- 普通末端动作与焊缝坐标系残差动作；
- 不同动作块长度；
- 10 Hz、20 Hz 等不同策略频率。

只要原始数据中保存了高质量的绝对位姿和时间戳，这些动作表示都可以从同一批数据离线构造。

---

# 一、Hy-VLA 实际上是怎样做的

Hy-VLA 的 UMI 数据保存的是完整的、多相机时序数据以及每一帧的绝对末端位姿。论文将机器人状态表示为相对于系统根坐标系的 EEF 位姿，而未来动作则是在**当前末端坐标系中、相对当前时刻定义的 future delta chunk**。训练时，数据加载器随机选择当前帧，再从整段 episode 中取出未来一段绝对位姿作为动作块。([arXiv](https://arxiv.org/html/2606.14409v2))

假设一段数据保存了绝对末端位姿：

$$
{}^B\mathbf T_{E,0},
{}^B\mathbf T_{E,1},
\dots,
{}^B\mathbf T_{E,N-1},
$$

其中 $B$ 是统一世界或机器人基坐标系，$E$ 是末端坐标系。

在时刻 $t$，选择长度为 $H$ 的未来轨迹：

$$
\left\{
{}^B\mathbf T_{E,t},
{}^B\mathbf T_{E,t+1},
\dots,
{}^B\mathbf T_{E,t+H-1}
\right\}.
$$

Hy-VLA 将每个未来位姿变换到当前末端坐标系：

$$
\Delta\mathbf T_{t,k}
=
\left({}^B\mathbf T_{E,t}\right)^{-1}
{}^B\mathbf T_{E,t+k},
\qquad k=0,\dots,H-1.
$$

其代码中的 `poses_to_relative_matrices()` 正是计算：

$$
\mathbf T_{\mathrm{rel},k}
=
\mathbf T_0^{-1}\mathbf T_k.
$$

平移使用相对变换的三维平移，旋转则使用相对旋转矩阵前两行展开得到的 6D 表示。([GitHub](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/main/hy_vla/utils/transform_utils.py))

UMI 数据加载器也明确执行了以下过程：

1. 从 episode 中抽取未来的 `observation.state`；
2. 将未来末端绝对状态堆叠成轨迹；
3. 调用 `dual_arm_poses_to_relative()`；
4. 在训练加载阶段得到相对动作块。

也就是说，完整的 6-DoF action chunk 并不一定要在采集阶段提前写入数据文件。([GitHub](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/main/hy_vla/data/umi_dataset.py))

---

# 二、一个重要区别：它不是“相邻帧增量”

你需要特别区分下面两种动作。

## 1. 相邻帧增量

每一步都相对于上一步：

$$
\delta\mathbf T_{t,k}
=
\left(
{}^B\mathbf T_{E,t+k-1}
\right)^{-1}
{}^B\mathbf T_{E,t+k}.
$$

动作块为：

$$
\mathcal A_t^{\mathrm{step}}
=
\left[
\delta\boldsymbol\xi_{t,1},
\delta\boldsymbol\xi_{t,2},
\dots,
\delta\boldsymbol\xi_{t,H}
\right].
$$

这更接近你当前理解的 ServoP 单周期增量。

---

## 2. 相对当前帧的未来位姿块

所有未来位姿都相对于当前时刻 $t$：

$$
\Delta\mathbf T_{t,k}
=
\left(
{}^B\mathbf T_{E,t}
\right)^{-1}
{}^B\mathbf T_{E,t+k}.
$$

动作块为：

$$
\mathcal A_t^{\mathrm{RT}}
=
\left[
\Delta\mathbf T_{t,1},
\Delta\mathbf T_{t,2},
\dots,
\Delta\mathbf T_{t,H}
\right].
$$

Hy-VLA 使用的是第二种。

例如一维运动中，机器人每帧向前移动 1 mm：

| 未来时刻 | 相邻帧增量 | 相对当前帧位移 |
|---:|---:|---:|
| $t+1$ | 1 mm | 1 mm |
| $t+2$ | 1 mm | 2 mm |
| $t+3$ | 1 mm | 3 mm |
| $t+4$ | 1 mm | 4 mm |

因此，Hy-VLA 的 delta chunk 更准确地说是：

> **相对当前末端坐标系表示的未来轨迹块。**

而不是：

> 每个元素都是下一个控制周期的增量。

---

# 三、这种表示为什么适合你的焊接轨迹任务

## 1. 对全局坐标系变化具有不变性

假设把整个系统的全局坐标系变换为 $\mathbf G$：

$$
\tilde{\mathbf T}_{E,t}
=
\mathbf G\mathbf T_{E,t}.
$$

相对动作仍然为：

$$
\begin{aligned}
\tilde{\mathbf T}_{E,t}^{-1}
\tilde{\mathbf T}_{E,t+k}
&=
(\mathbf G\mathbf T_{E,t})^{-1}
(\mathbf G\mathbf T_{E,t+k})\\
&=
\mathbf T_{E,t}^{-1}
\mathbf G^{-1}\mathbf G
\mathbf T_{E,t+k}\\
&=
\mathbf T_{E,t}^{-1}
\mathbf T_{E,t+k}.
\end{aligned}
$$

因此，纯粹改变世界坐标系定义不会改变动作标签。

---

## 2. 不需要对位姿轨迹做数值微分

相邻帧速度通常通过：

$$
\mathbf v_t
\approx
\frac{\mathbf p_{t+1}-\mathbf p_t}{\Delta t}
$$

计算，测量噪声会被除以很小的 $\Delta t$，导致速度和增量标签抖动。

相对当前帧的未来位姿块直接使用：

$$
\mathbf T_t^{-1}\mathbf T_{t+k},
$$

不需要先计算速度，更适合连续平滑轨迹。

---

## 3. 模型直接预测未来轨迹形状

对于曲线焊缝，未来轨迹块包含局部曲率信息：

$$
\left\{
\Delta\mathbf p_{t,1},
\dots,
\Delta\mathbf p_{t,H}
\right\}.
$$

模型不只是预测“下一步向哪里走”，而是在一次推理中预测未来一段曲线。

这对：

- 速度规划；
- 曲率变化；
- 转角减速；
- 姿态连续变化；
- 碰撞预测；
- 动作块平滑拼接；

都更加方便。

---

# 四、但你的仿真数据不能只保存 RGB 和 TCP 位姿

“以后再计算动作”成立的前提是：**原始数据中已经保存了足够的信息**。

建议将每条 episode 分成两个时间流。

## 1. 高频控制流：100 Hz

保存数值状态，不保存图像：

```text
control_timestamp
joint_position
joint_velocity
joint_acceleration
tcp_pose_base
flange_pose_base

expert_reference_tcp_pose
expert_command_tcp_pose
safe_command_tcp_pose
joint_command

workpiece_pose_base
minimum_collision_distance
collision_flag
joint_limit_margin
control_phase
```

其中要区分：

$$
\mathbf T_t^{\mathrm{ref}},
\qquad
\mathbf T_t^{\mathrm{cmd}},
\qquad
\mathbf T_t^{\mathrm{exec}}.
$$

它们分别代表：

- `reference`：轨迹规划器生成的理想参考位姿；
- `command`：专家控制器根据当前误差生成的命令；
- `executed`：MuJoCo 实际执行后的末端位姿。

---

## 2. 策略观测流：20 Hz

保存：

```text
policy_timestamp
global_rgb
wrist_rgb

joint_position
joint_velocity
tcp_pose_base

workpiece_pose_base
seam_progress
task_instruction
desired_speed
desired_work_angle
desired_travel_angle
phase

control_frame_index
```

其中 `control_frame_index` 指出该图像对应哪个 100 Hz 控制状态。

这样做以后，可以在训练阶段自由选择：

- 10 Hz 动作；
- 20 Hz 动作；
- 50 Hz 动作；
- 长度为 8、16、32 的动作块。

不过，图像只保存为 20 Hz 时，就无法构造高于 20 Hz 的视觉闭环策略。高频控制仍可以通过插值或底层控制器完成。

---

# 五、应该从哪一条轨迹计算训练 action

这取决于数据来源。

## 情况 1：脚本专家生成的仿真数据

你同时拥有：

$$
\mathbf T_t^{\mathrm{ref}},
\quad
\mathbf T_t^{\mathrm{cmd}},
\quad
\mathbf T_t^{\mathrm{exec}}.
$$

建议主要训练标签使用：

$$
\boxed{\mathbf T_t^{\mathrm{safe\_cmd}}}
$$

也就是：

> 经过闭环纠偏、限幅和安全投影后，真正发送给控制器的目标轨迹。

原因是模型部署时需要输出的是控制目标，而不是预测“机器人在动力学作用下最终发生了什么”。

建议保留三种标签视图：

```text
action.source = reference
action.source = safe_command
action.source = executed
```

正式模型优先使用 `safe_command`。

---

## 情况 2：拖拽示教数据

拖拽示教中没有显式的机器人目标命令。这时可以使用实际测得的未来 TCP 轨迹：

$$
{}^B\mathbf T_{E,t+k}^{\mathrm{exec}}
$$

构造动作块。

这与 UMI 的逻辑比较接近：人手末端轨迹本身就被视为示范动作轨迹。

---

## 情况 3：键盘或三维鼠标控制

最好同时保存：

- 遥操作设备原始输入；
- 发送给机器人的目标命令；
- 机器人实际执行位姿。

训练标签优先使用经过控制映射和安全过滤后的机器人命令，而不是原始键盘或三维鼠标输入。

---

# 六、推荐的原始 episode 格式

```text
episode_000001/
├── metadata.json
├── control.npz
├── policy.npz
├── global.mp4
├── wrist.mp4
└── scene.json
```

## `metadata.json`

```json
{
  "episode_id": 1,
  "seed": 12345,
  "robot": "elfin5pro",
  "workpiece_id": "part_001",
  "seam_id": "upper_fillet",
  "instruction": "从左向右沿上方焊缝运动",
  "direction": "forward",
  "desired_speed_mps": 0.02,
  "work_angle_rad": 0.785398,
  "travel_angle_rad": 0.174533,
  "simulation_hz": 500,
  "control_hz": 100,
  "policy_hz": 20,
  "length_unit": "meter",
  "angle_unit": "radian",
  "quaternion_order": "xyzw",
  "success": true
}
```

## `control.npz`

```text
timestamp                       (Nc,)
joint_position                  (Nc, 6)
joint_velocity                  (Nc, 6)

tcp_pose_executed_base          (Nc, 7)
tcp_pose_reference_base         (Nc, 7)
tcp_pose_command_base           (Nc, 7)
tcp_pose_safe_command_base      (Nc, 7)

workpiece_pose_base             (Nc, 7)
seam_frame_base                 (Nc, 7)

minimum_distance                (Nc,)
collision                       (Nc,)
phase                           (Nc,)
```

## `policy.npz`

```text
timestamp                       (Np,)
control_index                   (Np,)
tcp_pose_base                   (Np, 7)
joint_position                  (Np, 6)
joint_velocity                  (Np, 6)
seam_progress                   (Np,)
desired_speed                   (Np,)
desired_work_angle              (Np,)
desired_travel_angle            (Np,)
```

图像单独编码为 MP4，可以显著减少存储空间。

---

# 七、训练阶段如何动态构造 Hy-VLA 风格动作块

下面是一个单臂版本。

```python
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

def poses_xyzw_to_matrices(poses: np.ndarray) -> np.ndarray:
    """
    Convert pose sequence [x, y, z, qx, qy, qz, qw] to SE(3).

    Args:
        poses: Shape (N, 7), quaternion order xyzw.

    Returns:
        Shape (N, 4, 4).
    """
    poses = np.asarray(poses, dtype=np.float64)

    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected (N, 7), got {poses.shape}")

    rotations = Rotation.from_quat(poses[:, 3:7]).as_matrix()

    transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        repeats=len(poses),
        axis=0,
    )
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = poses[:, :3]

    return transforms

def rotation_to_6d_rows(rotation: np.ndarray) -> np.ndarray:
    """
    Hy-VLA repository convention:
    flatten the first two rows of a rotation matrix.

    Args:
        rotation: Shape (..., 3, 3).

    Returns:
        Shape (..., 6).
    """
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)

def build_rt_relative_chunk(
    absolute_poses: np.ndarray,
    current_index: int,
    horizon: int,
    *,
    stride: int = 1,
    include_current: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a future pose chunk relative to the current EEF frame.

    Output per step:
        [relative_xyz(3), relative_rotation_6d(6)]

    Returns:
        actions: Shape (horizon, 9)
        valid_mask: Shape (horizon,)
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    transforms = poses_xyzw_to_matrices(absolute_poses)
    num_frames = len(transforms)

    if not 0 <= current_index < num_frames:
        raise IndexError(current_index)

    first_offset = 0 if include_current else 1

    raw_indices = (
        current_index
        + (np.arange(horizon) + first_offset) * stride
    )

    valid_mask = raw_indices < num_frames
    indices = np.clip(raw_indices, 0, num_frames - 1)

    current_inv = np.linalg.inv(transforms[current_index])
    relative = current_inv[None] @ transforms[indices]

    relative_position = relative[:, :3, 3]
    relative_rotation_6d = rotation_to_6d_rows(
        relative[:, :3, :3]
    )

    actions = np.concatenate(
        [relative_position, relative_rotation_6d],
        axis=-1,
    ).astype(np.float32)

    return actions, valid_mask
```

使用：

```python
actions, mask = build_rt_relative_chunk(
    absolute_poses=tcp_pose_safe_command_base,
    current_index=t,
    horizon=16,
    stride=1,
)
```

假设策略数据频率为 20 Hz，那么：

$$
H=16
$$

对应：

$$
16\times0.05=0.8\text{ s}
$$

的未来轨迹。

---

# 八、是否应该像 Hy-VLA 一样包含当前帧

Hy-VLA 数据加载器从当前索引 `c_id` 开始构造未来动作，所以第一个相对位姿为：

$$
\mathbf T_t^{-1}\mathbf T_t
=
\mathbf I.
$$

对应：

$$
\Delta\mathbf p=\mathbf 0,
\qquad
\Delta\mathbf R=\mathbf I.
$$

如果你严格复用其模型代码和预训练权重，最好遵循相同语义。

但对于你自己设计的模型，更自然的定义是：

$$
k=1,\dots,H,
$$

也就是第一个动作对应 $t+1$，而不是当前时刻。

建议采用：

```text
Hy-VLA 兼容实验：
include_current = true

自己的 WeldPath-VLA：
include_current = false
```

必须把这个定义写入数据集配置，不能在不同实验中无意改变。

---

# 九、模型输出后如何变成 ServoP 单步增量

假设模型输出的是相对当前时刻的累计轨迹：

$$
\Delta\mathbf T_{t,1},
\Delta\mathbf T_{t,2},
\dots,
\Delta\mathbf T_{t,H}.
$$

对应的绝对目标位姿是：

$$
{}^B\mathbf T_{E,t+k}^{\mathrm{target}}
=
{}^B\mathbf T_{E,t}
\Delta\mathbf T_{t,k}.
$$

如果 Elfin5Pro 接口需要相邻增量，则转换为：

$$
\delta\mathbf T_{t,1}
=
\Delta\mathbf T_{t,1},
$$

$$
\delta\mathbf T_{t,k}
=
\Delta\mathbf T_{t,k-1}^{-1}
\Delta\mathbf T_{t,k},
\qquad k\geq2.
$$

也就是：

```text
模型：
相对当前帧的未来轨迹块

执行层：
转成连续目标位姿
→ 安全投影
→ 100 Hz 插值
→ ServoP 小增量
```

不要直接把模型输出的累计位移逐项当作 ServoP 增量，否则会产生越来越大的重复运动。

---

# 十、你的数据还能离线构造另外两种动作

同一份绝对轨迹可以生成不同动作表示。

## 1. 相邻帧 $SE(3)$ 增量

$$
\delta\boldsymbol\xi_{t,k}
=
\operatorname{Log}
\left(
\mathbf T_{t+k-1}^{-1}
\mathbf T_{t+k}
\right)^\vee.
$$

得到 6 维：

$$
[
\Delta x,\Delta y,\Delta z,
\Delta\phi_x,\Delta\phi_y,\Delta\phi_z
].
$$

适合直接的末端增量策略。

---

## 2. Hy-VLA 风格累计相对轨迹

$$
\mathbf a_{t,k}
=
\phi\left(
\mathbf T_t^{-1}\mathbf T_{t+k}
\right).
$$

适合动作块和轨迹生成。

---

## 3. 焊缝参考轨迹残差

设几何专家生成的参考位姿为：

$$
\mathbf T_{t+k}^{\mathrm{ref}},
$$

安全专家的真实命令为：

$$
\mathbf T_{t+k}^{\mathrm{cmd}}.
$$

残差为：

$$
\mathbf E_{t,k}
=
\left(
\mathbf T_{t+k}^{\mathrm{ref}}
\right)^{-1}
\mathbf T_{t+k}^{\mathrm{cmd}}.
$$

模型学习：

$$
\delta\boldsymbol\xi_{t,k}^{\mathrm{res}}
=
\operatorname{Log}(\mathbf E_{t,k})^\vee.
$$

这正好适合你后续研究：

> 普通 VLA 动作表示与焊缝参考轨迹残差表示的消融实验。

因为所有动作都来自同一批原始 episode，所以实验不会受到数据采集差异干扰。

---

# 十一、速度和姿态要求必须显式保存

即使以后能从绝对轨迹计算实际速度：

$$
v_t^{\mathrm{exec}}
=
\frac{
\|\mathbf p_{t+1}-\mathbf p_t\|
}{
t_{t+1}-t_t
},
$$

仍然必须保存任务要求中的期望速度：

$$
v_d(s).
$$

原因是：

$$
v_t^{\mathrm{exec}}
\neq v_d(s_t)
$$

可能来自：

- 控制器跟踪误差；
- 转角减速；
- 安全过滤；
- 动作限幅；
- 碰撞规避；
- 仿真动力学。

姿态同理，需要同时保存：

```text
desired_work_angle
desired_travel_angle
reference_orientation
executed_orientation
```

不要只从最终执行轨迹反推出任务要求。

---

# 十二、末尾动作块不要只重复最后一帧

Hy-VLA 的开源加载器在未来索引超出 episode 时，会把索引裁剪到最后一帧，即：

$$
i_k=\min(t+k,N-1).
$$

这样可以构造固定长度动作块。([GitHub](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/main/hy_vla/data/umi_dataset.py))

你的数据也可以重复最后位姿，但必须同时返回：

$$
m_{t,k}\in\{0,1\}
$$

作为有效动作掩码：

$$
m_{t,k}
=
\begin{cases}
1,&t+k<N,\\
0,&t+k\geq N.
\end{cases}
$$

训练损失使用：

$$
\mathcal L
=
\frac{
\sum_k m_{t,k}
\|\hat{\mathbf a}_{t,k}-\mathbf a_{t,k}\|^2
}{
\sum_k m_{t,k}+\epsilon
}.
$$

否则模型会在大量 episode 末尾样本中学到长时间静止。

还应单独保存：

```text
phase = approach / track / retreat
episode_done
seam_progress
```

---

# 十三、训练时计算，还是预处理时计算

建议采用三层结构。

## 第一层：不可变原始数据

```text
absolute TCP poses
joint states
timestamps
images
commands
scene geometry
task conditions
```

原始层永远不修改。

## 第二层：动作构造适配器

```python
ActionBuilder(
    source="safe_command",
    frame="eef",
    representation="rt_relative",
    rotation="rot6d_rows",
    horizon=16,
    stride=1,
)
```

它负责从完整 episode 生成训练样本。

## 第三层：可选缓存

第一次训练时可以动态计算；当动作表示固定后，缓存到：

```text
derived/
└── rt_relative_h16_stride1/
```

位姿矩阵运算本身开销很低，真正耗时的通常是视频解码，因此早期实验动态计算即可。

不过计算归一化统计量时，必须使用与训练完全相同的 `ActionBuilder` 配置。

---

# 十四、与 LeRobot 的关系

Hy-VLA 的公开数据采用兼容 LeRobot v3.0 schema 的 Lance 格式，数据加载器能够按帧或按 episode 读取完整轨迹。([GitHub](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA))

你不必照搬 Lance。更适合你的结构是：

```text
原始 HDF5/Zarr/NPZ + MP4
        ↓
ActionBuilder
        ↓
LeRobot adapter / Hy-VLA adapter / 自定义 PyTorch Dataset
```

LeRobot 中可以保存：

```text
observation.state = 当前绝对 TCP 和关节状态
action = 单帧安全命令或绝对目标位姿
```

但模型训练时不要被 LeRobot 的默认 `action` 字段限制。自定义数据集包装器可以读取整个 episode，再生成任意形式的动作块。

---

# 十五、对你当前项目的具体推荐

第一版仿真数据按照以下方式记录：

### 原始状态

$$
[
\mathbf q_t,\dot{\mathbf q}_t,
{}^B\mathbf T_{E,t}^{\mathrm{exec}}
]
$$

### 专家轨迹

$$
[
{}^B\mathbf T_{E,t}^{\mathrm{ref}},
{}^B\mathbf T_{E,t}^{\mathrm{cmd}},
{}^B\mathbf T_{E,t}^{\mathrm{safe}}
]
$$

### 几何信息

$$
[
{}^B\mathbf T_{O,t},
{}^B\mathbf T_{S,t},
s_t,
d_{\min,t}
]
$$

### 视觉信息

$$
[
I_t^{global},
I_t^{wrist}
]
$$

### 任务信息

$$
[
l,
v_d,
\alpha_d,
\beta_d,
d_{\mathrm{safe}}
]
$$

首个模型使用：

$$
\boxed{
\mathbf a_{t,k}
=
\phi\left[
\left(
{}^B\mathbf T_{E,t}^{\mathrm{exec}}
\right)^{-1}
{}^B\mathbf T_{E,t+k}^{\mathrm{safe}}
\right]
}
$$

也就是：

- 当前真实 TCP 作为相对参考；
- 未来安全专家命令作为动作目标；
- 输出相对当前末端坐标系的未来轨迹块；
- 平移 3 维；
- 旋转 6D；
- 单臂共 9 维。

随后再从同一批数据构造：

- 相邻 $SE(3)$ 增量；
- 基坐标系动作；
- 焊缝局部坐标系动作；
- 焊缝参考轨迹残差；

做动作表示消融。

## 最终结论

**完全可以参考 Hy-VLA 的方式，而且应该这样做。**

但更准确的表述不是：

> “先不保存动作，训练时再凭空计算动作。”

而是：

> **采集并保存完整的绝对状态轨迹、专家参考轨迹、实际控制命令和时间同步信息；训练阶段根据所选动作语义，从完整 episode 中构造模型特定的动作块。**

这会让你的数据集从一开始就不依赖 SmolVLA、Hy-VLA 或 LeRobot 的特定动作定义，也为后续最重要的“动作坐标系与轨迹表示消融”保留空间。

---

**Sources:**

- [1 Introduction](https://arxiv.org/html/2606.14409v2)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/main/hy_vla/utils/transform_utils.py)
- [GitHub - Tencent-Hunyuan/Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack · GitHub](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)