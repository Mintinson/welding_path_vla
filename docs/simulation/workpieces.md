# 工件与焊缝任务

仿真把“工件外形”和“沿哪条焊缝执行”分开配置：

- `workpiece.kind` 选择几何；
- `task.seam_id` 选择该工件上的焊缝；
- `task.direction` 选择正向或反向；
- `WorkpieceObject.seam()` 返回统一的 `SeamPath`。

`SeamPath` 负责按进度采样位置、切向和法向，也负责把实际 TCP 投影回有限焊缝。专家轨迹、
数据采集、自然退出和 ACT rollout 因此不再假设焊缝一定是直线。

## 可视焊缝状态

每条需要执行的焊缝按真实任务长度绘制为 5 mm 左右的黑色分段，线宽为 5 mm。TCP 距离某段
小于 `task.weld_success_distance_m`（默认 6 mm）后，该段永久变为白色，直到环境重置。
这只改变渲染颜色，不参与碰撞、轨迹投影或质量指标。

水平双焊缝只绘制两条真实直线；绕开三板死角的圆角属于连续转移动作，不显示为待焊黑线。

共享同一工件的候选任务会同时显示。例如圆管始终显示 `pipe_bottom` 和 `pipe_top`，三面角
始终显示 `horizontal_pair` 和 `vertical_corner`；专家只会把当前任务经过的焊缝逐段变白，
其余候选线保持黑色。策略因此必须结合任务文本判断目标，而不能从“场景里唯一一条线”猜测任务。

## 现有工件

| 配置 | 工件 | 焊缝 |
| --- | --- | --- |
| `configs/tasks/l_joint.yaml` | L 形双板 | `straight_fillet` 直线角焊缝 |
| `configs/tasks/pipe_bottom.yaml` | 空心圆管 + 方形底板 | `pipe_bottom` 管底圆弧 |
| `configs/tasks/pipe_top.yaml` | 空心圆管 + 方形底板 | `pipe_top` 管口圆弧 |
| `configs/tasks/curve_plate.yaml` | 带可见周期曲线的平板 | `curve_seam` 正弦/余弦焊缝 |
| `configs/tasks/trihedral_horizontal.yaml` | 三块互相垂直的平板 | 连续连接 `floor_x`、`floor_y` 的 `horizontal_pair` |
| `configs/tasks/trihedral_vertical.yaml` | 三块互相垂直的平板 | `vertical_corner`、`floor_x`、`floor_y` 三条内角直线 |

圆管由 32 个环向碰撞/外观分段构成，因此内部保持空心，不会像实心 cylinder 一样错误阻挡
内壁空间。`pipe_bottom` 默认走 90° 前侧圆弧；`pipe_top` 默认走完整 360° 管口圆周。
可通过 `task.arc_start_deg`、`task.arc_sweep_deg` 和 `task.direction` 修改范围与方向。
采集器会在录制前逐帧运行连续 IK、关节限位、速度连续性和碰撞预检，不能只确认起点可达。圆弧姿态使用相邻帧旋转增量累积，避免跨过 180° 时 SLERP 最短路径翻转。

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

曲线平板使用 `configs/curve_plate.yaml`。焊缝按照真实弧长均匀采样，振幅、周期数、
正弦/余弦类型和执行方向记录在 `task_parameters`；`orientation_follow_ratio=0` 保证单个
episode 的跟踪姿态固定，只考察策略的平面曲线执行能力。

```bash
pixi run -e sim sim-view --config_path=configs/curve_plate.yaml
pixi run -e sim sim-collect \
  --config_path=configs/curve_plate.yaml \
  --collection.episodes=50
```

三面角工件完整显示两条水平内角焊缝和一条竖直内角焊缝。
`configs/trihedral_vertical.yaml` 选择 `vertical_corner`，用于考察自下而上的竖焊能力；
`configs/trihedral_horizontal.yaml` 则从 `floor_x` 外端进入，以 `trihedral_turn_radius_m` 为半径
绕过三板交汇死角，
再沿 `floor_y` 一次运行到另一侧外端。两条直线共用 `task.seam_length_m`，因此每组 episode
会同步改变两边的有效长度，整个过程只有一次接近和一次退出。

三个板件尺寸分别由 `trihedral_floor_size_m`、`trihedral_wall_x_size_m` 和
`trihedral_wall_y_size_m` 表示。非厚度方向尺寸、焊缝长度、工件位姿、焊枪姿态和速度均
采用小范围成组随机化；竖缝使用 `trihedral_corner_margin_m` 避开数学尖角，水平双焊缝使用
独立的 `trihedral_turn_radius_m` 为枪颈转向留出空间。

```bash
pixi run -e sim sim-view --config_path=configs/trihedral_vertical.yaml
pixi run -e sim sim-collect \
  --config_path=configs/trihedral_vertical.yaml \
  --collection.episodes=50
pixi run -e sim sim-view --config_path=configs/trihedral_horizontal.yaml
pixi run -e sim sim-collect \
  --config_path=configs/trihedral_horizontal.yaml \
  --collection.episodes=50
```

## 新增工件

只在确有新工件时扩展以下四个位置：

1. 在 `WorkpieceConfig` 增加必要的实体尺寸；
2. 在 `WorkpieceObject` 增加几何构造和 `seam()` 分支；
3. 在 `configs/tasks/` 增加任务模块，并使用独立 `collection.dataset_root`；
4. 在 `configs/deploy/` 增加引用该任务的部署入口。

机器人模型、Arena、控制器、录制器和策略接口不应随工件变化。新增焊缝若仍是直线、圆弧
或周期曲线，直接复用现有路径类；只有真实几何无法表达时才新增 `SeamPath` 实现。
