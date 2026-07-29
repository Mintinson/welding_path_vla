# 工件与焊缝任务

仿真把“工件外形”和“沿哪条焊缝执行”分开配置：

- `workpiece.kind` 选择几何；
- `task.seam_id` 选择该工件上的焊缝；
- `task.direction` 选择正向或反向；
- `WorkpieceObject.seam()` 返回统一的 `SeamPath`。

`SeamPath` 负责按进度采样位置、切向和法向，也负责把实际 TCP 投影回有限焊缝。专家轨迹、
数据采集、自然退出和 ACT rollout 因此不再假设焊缝一定是直线。

## 现有工件

| 配置 | 工件 | 焊缝 |
| --- | --- | --- |
| `configs/default.yaml` | L 形双板 | `straight_fillet` 直线角焊缝 |
| `configs/pipe_bottom.yaml` | 空心圆管 + 方形底板 | `pipe_bottom` 管底圆弧 |
| `configs/pipe_top.yaml` | 空心圆管 + 方形底板 | `pipe_top` 管口圆弧 |

圆管由 32 个环向碰撞/外观分段构成，因此内部保持空心，不会像实心 cylinder 一样错误阻挡
内壁空间。`pipe_bottom` 默认走 90° 前侧圆弧；`pipe_top` 默认走完整 360° 管口圆周。
可通过 `task.arc_start_deg`、`task.arc_sweep_deg` 和 `task.direction` 修改范围与方向。
扩大范围后应重新运行整条轨迹的 IK 和碰撞检查，不能只确认起点可达。

```bash
pixi run -e sim sim-view --config_path=configs/pipe_bottom.yaml
pixi run -e sim sim-collect \
  --config_path=configs/pipe_top.yaml \
  --collection.episodes=5
```

圆弧上的切向和径向法向会随进度连续变化。`task.orientation_follow_ratio` 控制焊枪姿态
跟随程度：`0` 保持起点姿态，`1` 完全跟随局部标架。管口整圆默认取 `0`，避免末端无必要地
旋转一整圈；`task.work_angle_deg`、`task.travel_angle_deg` 和 `task.tool_roll_deg` 分别控制
工作角、行走角和滚转角。

## 新增工件

只在确有新工件时扩展以下三个位置：

1. 在 `WorkpieceConfig` 增加必要的实体尺寸；
2. 在 `WorkpieceObject` 增加几何构造和 `seam()` 分支；
3. 在 `configs/` 增加独立 YAML，并使用独立 `collection.dataset_root`。

机器人模型、Arena、控制器、录制器和策略接口不应随工件变化。新增焊缝若仍是直线或圆弧，
直接复用现有路径类；只有真实几何无法表达时才新增新的 `SeamPath` 实现。
