"""直线与圆弧焊缝的统一几何接口。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from welding_path_vla.core.geometry import normalize


@dataclass(slots=True)
class SeamFrame:
    """焊缝某一进度处的局部标架。

    Attributes:
        position: TCP 参考位置，世界坐标，单位为米。
        tangent: 沿任务执行方向的单位切向。
        normal: 焊枪接近侧的单位法向。
    """

    position: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        """返回列向量依次为切向、副法向和法向的旋转矩阵。"""
        binormal = normalize(np.cross(self.normal, self.tangent))
        return np.column_stack([self.tangent, binormal, self.normal])


@dataclass(slots=True)
class SeamProjection:
    """TCP 到有限焊缝的投影结果。

    Attributes:
        progress: 裁剪到 ``[0, 1]`` 的有向进度。
        raw_progress: 未裁剪进度，用于判断是否越过端点。
        position: 焊缝上最近的参考位置。
        distance_m: TCP 到参考位置的欧氏距离。
    """

    progress: float
    raw_progress: float
    position: np.ndarray
    distance_m: float


class SeamPath:
    """工件无关的有向焊缝接口。"""

    seam_id: str
    length_m: float

    def sample(self, progress: float) -> SeamFrame:
        """返回指定归一化进度处的焊缝标架。"""
        raise NotImplementedError

    def project(
        self,
        position: np.ndarray,
        hint_progress: float | None = None,
    ) -> SeamProjection:
        """将世界坐标位置投影到有限焊缝。"""
        raise NotImplementedError

    @property
    def start(self) -> SeamFrame:
        """返回任务起点标架。"""
        return self.sample(0.0)

    @property
    def end(self) -> SeamFrame:
        """返回任务终点标架。"""
        return self.sample(1.0)


class StraightSeamPath(SeamPath):
    """具有固定法向的有限直线焊缝。"""

    def __init__(
        self,
        seam_id: str,
        start: np.ndarray,
        end: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        """创建有向直线。

        Args:
            seam_id: 稳定的焊缝标识。
            start: 世界坐标起点。
            end: 世界坐标终点。
            normal: 世界坐标焊接法向。
        """
        self.seam_id = seam_id
        self.start_position = np.asarray(start, dtype=np.float64)
        self.end_position = np.asarray(end, dtype=np.float64)
        self.vector = self.end_position - self.start_position
        self.length_m = float(np.linalg.norm(self.vector))
        self.tangent = self.vector / self.length_m
        self.normal = normalize(np.asarray(normal, dtype=np.float64))

    def sample(self, progress: float) -> SeamFrame:
        """在线段上按进度插值位置。"""
        alpha = float(np.clip(progress, 0, 1))
        return SeamFrame(
            self.start_position + alpha * self.vector,
            self.tangent.copy(),
            self.normal.copy(),
        )

    def project(
        self,
        position: np.ndarray,
        hint_progress: float | None = None,
    ) -> SeamProjection:
        """解析计算点到有限线段的投影。"""
        del hint_progress
        raw = float(np.dot(position - self.start_position, self.vector) / self.length_m**2)
        progress = float(np.clip(raw, 0, 1))
        closest = self.sample(progress).position
        return SeamProjection(progress, raw, closest, float(np.linalg.norm(position - closest)))


class RoundedCornerSeamPath(SeamPath):
    """连接两条互相垂直直线的圆角路径。

    两条直线分别位于局部 X、Y 轴上，圆角绕开三板交汇的不可达死角。正向执行时
    从 X 轴外端进入，经圆角连续转向后从 Y 轴外端离开。
    """

    def __init__(
        self,
        seam_id: str,
        center: np.ndarray,
        rotation: np.ndarray,
        straight_length_m: float,
        corner_radius_m: float,
        work_angle_rad: float,
        clearance_m: float,
        reverse: bool,
    ) -> None:
        """创建水平双焊缝的连续 TCP 路径。

        Args:
            seam_id: 稳定的焊缝标识。
            center: 三板交汇点的世界坐标。
            rotation: 工件局部坐标到世界坐标的旋转矩阵。
            straight_length_m: 每条水平直线的有效焊接长度。
            corner_radius_m: 绕开三板交汇死角的转弯半径。
            work_angle_rad: 焊枪相对底板法向的工作角。
            clearance_m: TCP 沿焊接法向离开理论焊缝的净空。
            reverse: 是否从 Y 轴外端反向执行到 X 轴外端。
        """
        self.seam_id = seam_id
        self.center = np.asarray(center, dtype=np.float64)
        self.workpiece_rotation = np.asarray(rotation, dtype=np.float64)
        self.straight_length_m = straight_length_m
        self.corner_radius_m = corner_radius_m
        self.work_angle_rad = work_angle_rad
        self.clearance_m = clearance_m
        self.reverse = reverse
        self.horizontal_clearance_m = clearance_m * np.sin(work_angle_rad)
        self.vertical_clearance_m = clearance_m * np.cos(work_angle_rad)
        self.effective_radius_m = corner_radius_m - self.horizontal_clearance_m
        if self.effective_radius_m <= 0:
            raise ValueError("corner radius must exceed the horizontal TCP clearance")
        self.arc_length_m = np.pi * self.effective_radius_m / 2
        self.length_m = 2 * straight_length_m + self.arc_length_m

    def local_frame(self, distance_m: float) -> SeamFrame:
        """返回正向弧长位置对应的工件局部标架。"""
        distance = float(np.clip(distance_m, 0, self.length_m))
        radius = self.corner_radius_m
        offset = self.horizontal_clearance_m
        height = self.vertical_clearance_m
        work_sin = np.sin(self.work_angle_rad)
        work_cos = np.cos(self.work_angle_rad)
        if distance <= self.straight_length_m:
            position = np.array([radius + self.straight_length_m - distance, offset, height])
            tangent = np.array([-1.0, 0.0, 0.0])
            normal = np.array([0.0, work_sin, work_cos])
        elif distance < self.straight_length_m + self.arc_length_m:
            angle = (distance - self.straight_length_m) / self.effective_radius_m
            position = np.array(
                [
                    radius - self.effective_radius_m * np.sin(angle),
                    radius - self.effective_radius_m * np.cos(angle),
                    height,
                ]
            )
            tangent = np.array([-np.cos(angle), np.sin(angle), 0.0])
            normal = np.array([work_sin * np.sin(angle), work_sin * np.cos(angle), work_cos])
        else:
            line_distance = distance - self.straight_length_m - self.arc_length_m
            position = np.array([offset, radius + line_distance, height])
            tangent = np.array([0.0, 1.0, 0.0])
            normal = np.array([work_sin, 0.0, work_cos])
        return SeamFrame(position, tangent, normal)

    def sample(self, progress: float) -> SeamFrame:
        """按总弧长进度采样位置，并保持圆角处切向和法向连续。"""
        alpha = float(np.clip(progress, 0, 1))
        distance = self.length_m * (1 - alpha if self.reverse else alpha)
        local = self.local_frame(distance)
        direction = -1.0 if self.reverse else 1.0
        return SeamFrame(
            self.center + self.workpiece_rotation @ local.position,
            normalize(direction * self.workpiece_rotation @ local.tangent),
            normalize(self.workpiece_rotation @ local.normal),
        )

    def project(
        self,
        position: np.ndarray,
        hint_progress: float | None = None,
    ) -> SeamProjection:
        """投影到两条直线或连接圆角，并用进度提示消除拐角歧义。"""
        local = self.workpiece_rotation.T @ (position - self.center)
        radius = self.corner_radius_m
        line = self.straight_length_m

        first_raw = (radius + line - local[0]) / line
        first_alpha = float(np.clip(first_raw, 0, 1))
        first_distance = first_alpha * line

        angle = float(np.clip(np.arctan2(radius - local[0], radius - local[1]), 0, np.pi / 2))
        arc_distance = line + angle * self.effective_radius_m

        second_raw = (local[1] - radius) / line
        second_alpha = float(np.clip(second_raw, 0, 1))
        second_distance = line + self.arc_length_m + second_alpha * line
        candidates = [first_distance, arc_distance, second_distance]
        projections = [self.local_frame(distance).position for distance in candidates]
        distances = [float(np.linalg.norm(local - projected)) for projected in projections]
        target = 0.5 if hint_progress is None else hint_progress
        progresses = [distance / self.length_m for distance in candidates]
        if self.reverse:
            progresses = [1 - progress for progress in progresses]
        best = min(
            range(3),
            key=lambda index: (round(distances[index], 10), abs(progresses[index] - target)),
        )

        geometric_raw = candidates[best] / self.length_m
        if best == 0 and first_raw < 0:
            geometric_raw = first_raw * line / self.length_m
        elif best == 2 and second_raw > 1:
            geometric_raw = (line + self.arc_length_m + second_raw * line) / self.length_m
        raw_progress = 1 - geometric_raw if self.reverse else geometric_raw
        progress = float(np.clip(raw_progress, 0, 1))
        world_point = self.center + self.workpiece_rotation @ projections[best]
        return SeamProjection(progress, raw_progress, world_point, distances[best])


class SinusoidalSeamPath(SeamPath):
    """平板上的正弦或余弦焊缝，以真实弧长均匀采样。"""

    def __init__(
        self,
        seam_id: str,
        center: np.ndarray,
        rotation: np.ndarray,
        length_m: float,
        height_m: float,
        amplitude_m: float,
        frequency: float,
        curve_kind: str,
        reverse: bool,
    ) -> None:
        """创建平面周期曲线焊缝。

        Args:
            seam_id: 稳定的焊缝标识。
            center: 工件局部原点的世界坐标。
            rotation: 工件局部坐标到世界坐标的旋转矩阵。
            length_m: 曲线在局部 Y 方向覆盖的长度。
            height_m: TCP 曲线相对工件原点的高度。
            amplitude_m: 局部 X 方向振幅。
            frequency: 整段曲线包含的周期数量。
            curve_kind: ``sine`` 或 ``cosine``。
            reverse: 是否从几何终点向起点执行。
        """
        self.seam_id = seam_id
        self.center = np.asarray(center, dtype=np.float64)
        self.workpiece_rotation = np.asarray(rotation, dtype=np.float64)
        self.span_m = length_m
        self.height_m = height_m
        self.amplitude_m = amplitude_m
        self.frequency = frequency
        self.curve_kind = curve_kind
        self.reverse = reverse

        self.u_lookup = np.linspace(0.0, 1.0, 1001)
        self.local_lookup = self.local_points(self.u_lookup)
        segment_lengths = np.linalg.norm(np.diff(self.local_lookup, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.length_m = float(cumulative[-1])
        self.progress_lookup = cumulative / self.length_m

    def curve_values(self, u: np.ndarray) -> np.ndarray:
        """返回参数位置对应的局部 X 坐标。"""
        phase = 2 * np.pi * self.frequency * u
        function = np.cos if self.curve_kind == "cosine" else np.sin
        return self.amplitude_m * function(phase)

    def curve_derivatives(self, u: float) -> float:
        """返回局部 X 坐标对归一化参数的导数。"""
        phase = 2 * np.pi * self.frequency * u
        scale = 2 * np.pi * self.frequency * self.amplitude_m
        return float(
            -scale * np.sin(phase) if self.curve_kind == "cosine" else scale * np.cos(phase)
        )

    def local_points(self, u: np.ndarray) -> np.ndarray:
        """批量生成曲线的工件局部坐标。"""
        return np.column_stack(
            [
                self.curve_values(u),
                self.span_m * (u - 0.5),
                np.full_like(u, self.height_m),
            ]
        )

    def sample(self, progress: float) -> SeamFrame:
        """按弧长进度返回曲线位置、切向和固定平板法向。"""
        alpha = float(np.clip(progress, 0, 1))
        geometric_progress = 1.0 - alpha if self.reverse else alpha
        u = float(np.interp(geometric_progress, self.progress_lookup, self.u_lookup))
        local_position = self.local_points(np.array([u]))[0]
        direction = -1.0 if self.reverse else 1.0
        local_tangent = direction * np.array([self.curve_derivatives(u), self.span_m, 0.0])
        return SeamFrame(
            self.center + self.workpiece_rotation @ local_position,
            normalize(self.workpiece_rotation @ local_tangent),
            normalize(self.workpiece_rotation @ np.array([0.0, 0.0, 1.0])),
        )

    def project(
        self,
        position: np.ndarray,
        hint_progress: float | None = None,
    ) -> SeamProjection:
        """投影到稠密折线，并返回与执行方向一致的弧长进度。"""
        del hint_progress
        local = self.workpiece_rotation.T @ (position - self.center)
        nearest = int(np.argmin(np.linalg.norm(self.local_lookup - local, axis=1)))
        candidates = range(max(0, nearest - 1), min(len(self.local_lookup) - 1, nearest + 1))
        best_distance = float("inf")
        best_progress = 0.0
        best_point = self.local_lookup[nearest]
        for index in candidates:
            start = self.local_lookup[index]
            vector = self.local_lookup[index + 1] - start
            alpha = float(np.clip(np.dot(local - start, vector) / np.dot(vector, vector), 0, 1))
            closest = start + alpha * vector
            distance = float(np.linalg.norm(local - closest))
            if distance < best_distance:
                best_distance = distance
                best_progress = float(
                    self.progress_lookup[index]
                    + alpha * (self.progress_lookup[index + 1] - self.progress_lookup[index])
                )
                best_point = closest
        progress = 1.0 - best_progress if self.reverse else best_progress
        world_point = self.center + self.workpiece_rotation @ best_point
        return SeamProjection(progress, progress, world_point, best_distance)


class CircularSeamPath(SeamPath):
    """圆管外壁上的有限圆弧焊缝路径生成器。

    该类基于参数方程定义一条位于圆柱面上的三维圆弧焊缝。
    支持 TCP（工具中心点）相对于工件表面的安全间隙偏移，并能计算
    沿路径的位姿（位置、切向、法向）以及空间点到路径的投影。
    """

    def __init__(
        self,
        seam_id: str,
        center: np.ndarray,
        rotation: np.ndarray,
        radius_m: float,
        height_m: float,
        start_rad: float,
        sweep_rad: float,
        work_angle_rad: float,
        clearance_m: float,
    ) -> None:
        """初始化圆弧焊缝参数，并计算考虑TCP间隙的有效几何参数。

        坐标系约定：
        - 世界坐标系 (World Frame): MuJoCo 或仿真环境的全局坐标系。
        - 工件坐标系 (Workpiece Frame): 以 `center` 为原点，`rotation` 定义的局部坐标系。
          在工件坐标系中，圆管的轴线默认平行于 Z 轴。

        Args:
            seam_id: 焊缝的唯一标识符，用于日志或状态机管理。
            center: 工件坐标系原点在世界坐标系中的位置 [x, y, z]。
            rotation: 3x3 旋转矩阵，将工件坐标系下的向量变换到世界坐标系。
            radius_m: 圆管的外半径（单位：米）。
            height_m: 圆弧所在平面在工件坐标系中的 Z 轴高度（单位：米）。
            start_rad: 圆弧在工件坐标系 XY 平面中的起始角度（弧度）。
                       X轴正方向为 0，逆时针为正。
            sweep_rad: 圆弧扫掠角度（弧度）。符号决定方向：
                       > 0 表示逆时针执行，< 0 表示顺时针执行。
            work_angle_rad: 焊枪工作角（弧度）。定义为圆管径向与竖直方向（Z轴）
                            之间的夹角。0 表示焊枪垂直向下，π/2 表示水平。
            clearance_m: TCP 相对于圆管几何表面的安全间隙（单位：米）。
                         正值表示工具在管外，负值表示工具陷入管内。
        """
        self.seam_id = seam_id
        self.center = np.asarray(center, dtype=np.float64)
        self.workpiece_rotation = np.asarray(rotation, dtype=np.float64)
        self.radius_m = radius_m
        self.height_m = height_m
        self.start_rad = start_rad
        self.sweep_rad = sweep_rad
        self.work_angle_rad = work_angle_rad
        self.clearance_m = clearance_m

        # --- 有效几何参数计算 (核心逻辑) ---
        # 由于存在工作角 (work_angle_rad)，TCP 的实际运动轨迹是一个
        # 半径和高度都与原始圆管不同的圆锥或圆柱面。
        #
        # 1. 有效半径: 原始半径 + 间隙在水平方向（径向）的投影
        #    clearance * sin(work_angle) 是因为间隙是沿着法向（与工作角有关）的
        self.effective_radius_m = radius_m + clearance_m * np.sin(work_angle_rad)
        # 2. 有效高度: 原始高度 + 间隙在竖直方向的投影
        #    clearance * cos(work_angle) 是间隙在 Z 轴上的分量
        self.effective_height_m = height_m + clearance_m * np.cos(work_angle_rad)
        # 3. 焊缝总长度: 弧长 = 圆心角 * 半径
        #    使用 abs 确保长度为正
        self.length_m = abs(sweep_rad) * self.effective_radius_m

    def sample(self, progress: float) -> SeamFrame:
        """根据归一化进度 (0~1) 采样焊缝上的位姿。

        该方法计算当前进度下 TCP 在世界坐标系中的位置、切向量和法向量。
        切向量沿运动方向，法向量指向工具接近方向（焊枪轴线方向）。

        Args:
            progress: 归一化的进度值，范围 [0, 1]。
                      0 对应起点，1 对应终点。

        Returns:
            SeamFrame: 包含世界坐标系下的位置、单位切向量和单位法向量。
        """
        alpha = float(np.clip(progress, 0, 1))
        angle = self.start_rad + alpha * self.sweep_rad

        # 径向向量: 从圆心指向当前角度位置 (XY平面内)
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        direction = np.sign(self.sweep_rad)  # 确定运动方向 (顺/逆时针)
        # 切向量: 垂直于径向。(-sin, cos) 是 (cos, sin) 逆时针旋转90度的结果。
        # 乘以 direction 以保证切向量始终指向运动的前方。
        tangent = direction * np.array([-np.sin(angle), np.cos(angle), 0.0])
        # 法向量 (工具轴向): 由工作角决定。
        # 它是径向的水平分量与竖直分量的合成。
        # sin(work_angle)*radial: 水平方向分量（指向管外或管内）
        # cos(work_angle)*[0,0,1]: 竖直方向分量
        normal = np.sin(self.work_angle_rad) * radial + np.cos(self.work_angle_rad) * np.array(
            [0.0, 0.0, 1.0]
        )
        # 局部位置: 有效半径处的圆周上的点 + 有效高度
        local_position = np.array(
            [
                self.effective_radius_m * np.cos(angle),
                self.effective_radius_m * np.sin(angle),
                self.effective_height_m,
            ]
        )
        # --- 变换到世界坐标系并返回 ---
        return SeamFrame(
            self.center + self.workpiece_rotation @ local_position,
            normalize(self.workpiece_rotation @ tangent),
            normalize(self.workpiece_rotation @ normal),
        )

    def project(
        self,
        position: np.ndarray,
        hint_progress: float | None = None,
    ) -> SeamProjection:
        """将世界坐标系中的一个点投影到圆弧焊缝上，反求进度。

        主要用于传感器反馈（如视觉、力控）将实际测量位置映射回理论路径。
        利用 hint_progress 解决圆管闭合处的角度跳变问题（unwrap）。

        Args:
            position: 待投影点的世界坐标 [x, y, z]。
            hint_progress: 可选的进度提示（通常为上一时刻的进度）。
                           用于判断当前点位于圆弧的哪一圈，避免 2π 跳变。

        Returns:
            SeamProjection: 包含归一化进度、原始进度、最近点坐标及距离误差。
        """
        # 1. 将点转换到工件坐标系，以便计算角度
        # P_local = R^T * (P_world - Center)
        local = self.workpiece_rotation.T @ (position - self.center)
        # 2. 计算点在 XY 平面上的极角 (atan2(y, x))
        angle = float(np.arctan2(local[1], local[0]))
        # 获取扫掠方向
        direction = float(np.sign(self.sweep_rad))
        span = abs(self.sweep_rad)
        # 3. 角度解缠 (Unwrapping)
        #    计算当前角度相对于起点的有向角，并归一化到 [0, 2π)
        #    direction 用于处理顺时针/逆时针的符号一致性
        wrapped = (direction * (angle - self.start_rad)) % (2 * np.pi)
        # 4. 生成候选进度列表
        #    因为圆是闭合的，同一个点可能对应三个理论进度：
        #    - 前一整圈 (raw - 2pi)
        #    - 当前圈 (raw)
        #    - 后一整圈 (raw + 2pi)
        #    除以 span 将其转化为归一化的进度值
        candidates = np.array(
            [(wrapped - 2 * np.pi) / span, wrapped / span, (wrapped + 2 * np.pi) / span]
        )
        # 5. 选择最佳候选
        #    如果没有提示，默认选中间值 (0.5附近)。
        #    如果有提示（hint_progress），选择与提示值最接近的那个候选，
        #    这样可以保证在跨过 0/1 边界时进度是连续的，而不是突然跳变。
        target = 0.5 if hint_progress is None else hint_progress
        raw = float(candidates[np.argmin(np.abs(candidates - target))])

        # 6. 最终裁剪与最近点计算
        #    将原始进度限制在 [0, 1] 范围内，得到最终的 progress
        progress = float(np.clip(raw, 0, 1))
        #    使用计算出的 progress 重新采样，得到理论上最近的路径点
        closest = self.sample(progress).position
        return SeamProjection(progress, raw, closest, float(np.linalg.norm(position - closest)))
