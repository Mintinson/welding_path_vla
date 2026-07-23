# Photo-aligned simulation layout

当前场景依据单张实验台照片建立，采用以下近似基准：

- 桌面顶面高度：`0.29 m`
- 机械臂底座位置：`[0.0, 0.0, 0.29] m`
- 机械臂底座 yaw：`-90 deg`（绕底座竖直向上的局部 Z 轴；与 world Z 同向）
- 工件位置：`[0.45, 0.0, 0.2925] m`
- 全局相机位置：`[1.25, 0.0, 1.05] m`
- 全局相机垂直视场角：`55 deg`
- 腕部相机垂直视场角：`85 deg`

全局相机位于机器人正前上方，与真实布局的中心支架一致，并始终朝向工件中心。

这些数值来自照片比例而非手眼标定。桌面、底座、工件和相机位置由 `configs/default.yaml` 的 `scene` 与 `camera` 控制。开始 sim-to-real 实验前，应使用实测外参、相机内参和桌面坐标替换这些近似值。

工件默认只进行 XY 平移和 yaw 随机化，Z 固定在桌面支撑高度；未经夹具模型约束的正负 Z 随机会让工件嵌入桌面，因此默认关闭。随机到 IK 不可达或 staging 碰撞的位姿属于无效场景样本，采集器会按相同 episode seed 继续重采样，最多尝试 `randomization.max_sampling_attempts` 次。该 rejection sampling 保证被录制的 episode 都有高精度、无碰撞的有效起始姿态。

腕部相机安装在 link6 的负 Y 侧，光心局部位置为 `[0, -0.080, 0.134] m`，与 mesh 上突出的安装螺钉同轴，并位于安装板的前表面。它是固定相机而非跟踪相机，光轴由 YAML 中的目标点对准标定 TCP。

Elfin STL 在 link5 与 link6 的 joint6 交界处没有独立轴环，运行 MJCF 使用纯视觉的 `elfin_joint6_ring` 黑色圆环补齐外观；它不参与碰撞检测。基座 yaw 同时作用于 base mesh 和整条运动链，episode 中的 base-frame 位移由 world-frame 位移显式转换得到。
