# 动作表示与当前 pipeline

## 1. 统一术语

项目现在遵循 `action_repre.md` 中的 LeRobot 官方术语：

- `absolute action`：每个 future target 都是世界坐标系中的绝对目标。
- `relative action`：一个 chunk 内的所有 future target 都相对**预测时刻的同一个状态**。
- `delta action`：第 `t` 个动作相对第 `t` 个状态，序列中每一步的参考状态不同。

因此函数统一为：

```python
build_delta_action(episode)
build_relative_actions(episode, frame_index, horizon, stride)
```

旧名称的对应关系是：

| 旧名称 | 新名称 | 实际语义 |
|---|---|---|
| `build_relative_actions` | `build_delta_action` | sequential delta |
| `build_relative_action_chunk` | `build_relative_actions` | shared-anchor relative trajectory |

旧的 `include_current` 已删除。当前帧本身不是 future target，不应人为占用动作块的一位。

## 2. 9D EE 动作定义

策略内部学习的 relative action 为：

```text
[dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]
```

- 前三维是目标位置在预测时刻 TCP 坐标系中的位置差，单位为米。
- 后六维是相对旋转矩阵的前两行，即 rotation-6D rows。
- 一个 `[H, 9]` chunk 中的 H 个目标使用同一个 TCP 位姿作为锚点。

设预测时刻 TCP 世界位姿为 `(R₀, p₀)`，第 `k` 个绝对目标为 `(Rₖ, pₖ)`：

```text
relative_position[k] = R₀ᵀ (pₖ - p₀)
relative_rotation[k] = R₀ᵀ Rₖ
```

这是一段相对当前末端的短时轨迹，不是逐步速度，也不是相邻目标之差。

## 3. 为什么 LeRobot 数据集仍保存 absolute target

这是有意遵循 LeRobot 官方推荐的数据流：

```text
原始绝对轨迹
  → LeRobot 数据集保存 absolute EE target
  → 采样 future chunk
  → relative processor 以当前观测为共同锚点
  → normalization
  → policy
```

数据集的 `action` schema 是：

```text
[x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]
```

前三维名称没有 `d`，明确表示磁盘中的值是世界系绝对目标。不能在导出时把每一行预先变成 delta，因为 LeRobot 后续才会根据 `action_horizon` 组合 future chunk；过早转换会丢失“整个 chunk 共享同一锚点”的语义。

## 4. 数据转换发生了什么

`scripts/export_lerobot.py` 执行以下工作：

1. 从原始 episode 的 `safe_command`、`reference` 或 `executed` 读取绝对目标。
2. 写入 13D 状态：六关节角、TCP 世界位置、TCP wxyz 四元数。
3. 写入 9D absolute EE action。
4. 根据 `action_horizon` 和 `action_stride` 构造训练时会看到的 relative chunks。
5. 用 relative chunks 重算 `meta/stats.json` 中的 action 统计量。
6. 在 `meta/welding_path_vla_export.json` 固定动作类型、存储形式、horizon 和 stride。

导出命令：

```bash
pixi run -e data export-lerobot \
  --config_path=configs/smolvla.yaml \
  --dataset=datasets/weldpath_raw_v2 \
  --output=datasets/weldpath_lerobot_relative_v1 \
  --repo_id=huayan/weldpath_relative_v1
```

增量导出仍使用 `lerobot_export.incremental=true` 以及 episode 起止配置。新增数据后会对完整目标数据集重新计算 relative action 统计量，保证归一化一致。

## 5. 训练时模型接收和输出什么

LeRobot 根据策略配置取得 `[B, H, 9]` 的 absolute future targets。项目的 `RelativeEEActionsProcessorStep` 位于 normalizer 前：

```text
absolute [B, H, 9]
  → SE(3) relative [B, H, 9]
  → normalized relative [B, H, 9]
  → policy loss
```

所以 ACT、SmolVLA、π0、π0.5 和 TrajectoryVLA 的训练目标都已经统一为 relative action。模型的原始数值输出是**归一化后的 relative action chunk**，不是世界坐标目标。

数据校验会同时检查：

- 状态维数为 13，动作维数为 9；
- 磁盘 action 名称为 absolute schema；
- manifest 声明 `relative_action`；
- 数据集 horizon、stride 与 policy 配置完全一致。

## 6. 测试与离线 evaluate

离线 evaluate 使用与训练相同的 preprocessor：ground truth absolute chunk 会先变成 relative chunk，再归一化；policy prediction 与监督值因此在同一空间中计算 loss 和 MAE。

`normalized_action_mae` 的含义是“归一化 relative action 空间的误差”。它适合比较模型和排查收敛，但不是米或角度单位的最终轨迹指标。仿真 rollout 中的 CTE、姿态误差和速度误差才是物理量评价。

## 7. 部署时如何恢复动作

在线预测顺序为：

```text
当前观测 TCP 作为锚点
  → policy 预测 normalized relative chunk
  → unnormalize
  → 用该锚点一次性恢复整个 absolute world chunk
  → 队列执行前 n_action_steps 个目标
  → 重新观测并预测下一块
```

这里必须一次性恢复整个 chunk。如果每执行一步都用新的 TCP 再解码同一个 relative chunk，锚点会漂移，后续目标就不再是模型原本预测的轨迹。

rollout 保存的 `action` 现在是 postprocessor 输出的世界系 9D 绝对目标；`command_tcp_position` 是经过单步速度限制后真正交给 IK 的目标。二者不同可以直接暴露安全限幅行为。

## 8. 为什么 `build_relative_actions()` 更适合本项目

焊接 VLA 的输出目标是一段短时轨迹，模型需要同时表达未来的方向、曲率、速度分布和平滑性。共享锚点表示具有以下优势：

- 整个 chunk 是一个明确的几何对象。
- 不需要先积分一串 delta 才知道远端目标。
- chunk 内误差不会因逐步积分而累计。
- 圆弧、转角和避障轨迹可以直接相对当前末端表达。
- 更适合对整段轨迹施加平滑、碰撞和焊缝约束。

`build_delta_action()` 仍保留用于旧数据诊断和动作表示消融，但不再进入默认的数据转换、训练、测试或部署路径。

## 9. 兼容性说明

旧 LeRobot 数据集的 action 数值和统计量属于旧 delta 语义，不能继续增量写入；旧 checkpoint 的 processor 也无法证明使用了共享锚点 relative action。项目会明确拒绝这两类输入，而不是静默执行错误动作。

完成迁移需要：

1. 重新导出到 `datasets/weldpath_lerobot_relative_v1`。
2. 使用新配置重新训练各 baseline。
3. 部署新 checkpoint；其 `policy_preprocessor.json` 和 `policy_postprocessor.json` 会保存项目的 relative/absolute processor。

这次不提供旧 checkpoint 的自动兼容层，因为同名动作曾代表不同数学含义，自动猜测会让实验结果失去可复现性。
