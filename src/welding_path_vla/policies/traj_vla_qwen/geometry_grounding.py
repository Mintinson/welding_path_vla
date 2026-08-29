"""
轨迹监督的双相机焊缝接地标签、预测头与辅助损失。

该模块为焊接路径 VLA 模型提供几何监督信号，包括：
1. 在动作归一化前，利用未来 TCP 轨迹生成图像 patch 上的焊缝走廊标签（seam patch labels）；
2. 预测局部切向（tangent）和相对姿态（orientation）的辅助损失；
3. 对应的预测头与损失函数。

标签仅依赖 LeRobot 已有的绝对末端执行器动作、当前 TCP 状态和相机标定，
无需额外标注数据集。相机遵循 MuJoCo 约定：局部 ``-Z`` 为光轴、``+Y`` 为图像上方。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.converters import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE
from torch import Tensor, nn

from welding_path_vla.core.geometry import (
    quaternion_to_matrix,
    rotation_from_6d_rows,
    rotation_to_6d_rows,
)

# 定义观察数据中存储几何标签的键名
SEAM_LABELS = "observation.geometry.seam_patch_labels"  # 焊缝走廊二值标签 (B, C, P)
SEAM_VALID = "observation.geometry.seam_valid"  # 是否至少有一个有效走廊 patch (B, C)
TANGENT_TARGET = "observation.geometry.tangent"  # 局部切向目标 (B, 3)
TANGENT_VALID = "observation.geometry.tangent_valid"  # 切向是否有效 (B,)
ORIENTATION_TARGET = "observation.geometry.orientation_6d"  # 相对姿态 6D 表示 (B, 6)
ORIENTATION_VALID = "observation.geometry.orientation_valid"  # 姿态是否有效 (B,)


def pose_parts(pose: tuple[float, ...], like: Tensor) -> tuple[Tensor, Tensor]:
    """
    把配置中的 xyz + wxyz 位姿转换为与输入张量同设备、同精度的张量。

    Args:
        pose: 长度为 7 的元组，前 3 个为平移 (x, y, z)，后 4 个为四元数 (w, x, y, z)
        like: 参考张量，用于确定输出张量的设备和数据类型

    Returns:
        (position, rotation_matrix) 元组，position 形状 (3,)，rotation 形状 (3,3)
    """
    values = torch.as_tensor(pose, dtype=like.dtype, device=like.device)
    return values[:3], quaternion_to_matrix(values[3:])


def transform_pose(
    parent_position: Tensor,
    parent_rotation: Tensor,
    child_position: Tensor,
    child_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    组合 ``world_from_parent`` 与 ``parent_from_child`` 刚体位姿，得到 ``world_from_child``。

    Args:
        parent_position: 父坐标系原点在世界系中的位置，形状 (B, 3)
        parent_rotation: 父坐标系到世界系的旋转矩阵，形状 (B, 3, 3)
        child_position: 子坐标系原点在父坐标系中的位置，形状 (B, 3)
        child_rotation: 子坐标系到父坐标系的旋转矩阵，形状 (B, 3, 3)

    Returns:
        (position, rotation) 元组，表示子坐标系原点在世界系中的位置和旋转矩阵
    """
    position = parent_position + (parent_rotation @ child_position[..., None])[..., 0]
    return position, parent_rotation @ child_rotation


@ProcessorStepRegistry.register(name="welding_geometry_grounding_targets")
@dataclass
class GeometryGroundingTargetProcessorStep(ProcessorStep):
    """
    在动作归一化前由未来 TCP 目标在线生成几何监督。

    该步骤作为 LeRobot 处理器链的一部分，在训练时读取当前观察（包括状态和相机图像）
    以及对应的未来动作序列，计算出焊缝走廊标签、切向和姿态目标，并添加到观察字典中。
    部署时若无动作数据，则原样返回观测。
    """

    # Camera configuration: Two camera button names (usually global camera and wrist camera).
    camera_keys: tuple[str, str]
    # Camera Vertical Field of View (degrees)
    camera_fovy_deg: tuple[float, float]
    # Global camera pose in the world frame (xyz + wxyz)
    global_camera_pose_world: tuple[float, ...]
    # Wrist camera pose in TCP coordinate system (xyz + wxyz)
    wrist_camera_pose_tcp: tuple[float, ...]
    # Target image size (used to scale the projected points to the DINO input size, default 224x224)
    target_size: tuple[int, int] = (224, 224)
    # Patch grid size (e.g., 16 means the image is divided into 16x16 patches)
    patch_grid: int = 16
    # Corridor radius (pixels), used to determine whether the patch intersects
    # with the trajectory line segment.
    corridor_radius_px: float = 8.0

    def camera_poses(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        返回 batch 中每个样本的全局相机和腕部相机的世界系位姿。

        Args:
            state: 观测状态张量，形状 (B, S)，其中 S 至少包含 TCP 位置 (6:9) 和四元数 (9:13)

        Returns:
            (camera_positions, camera_rotations) 元组，
            camera_positions 形状 (B, 2, 3)，camera_rotations 形状 (B, 2, 3, 3)
            维度 1 的索引 0 对应全局相机，索引 1 对应腕部相机
        """
        batch = state.shape[0]
        # 全局相机位姿：配置固定，扩展到 batch
        global_position, global_rotation = pose_parts(self.global_camera_pose_world, state)
        global_position = global_position.expand(batch, -1)
        global_rotation = global_rotation.expand(batch, -1, -1)
        # TCP 在世界系中的位姿：从状态中提取平移和四元数
        tcp_position = state[:, 6:9]
        tcp_rotation = quaternion_to_matrix(state[:, 9:13])
        # 腕部相机相对于 TCP 的位姿（配置）
        wrist_position, wrist_rotation = pose_parts(self.wrist_camera_pose_tcp, state)
        # 组合得到腕部相机在世界系中的位姿
        wrist_position, wrist_rotation = transform_pose(
            tcp_position,
            tcp_rotation,
            wrist_position.expand(batch, -1),
            wrist_rotation.expand(batch, -1, -1),
        )
        # 堆叠成 (B, 2, 3) 和 (B, 2, 3, 3)
        return (
            torch.stack((global_position, wrist_position), dim=1),
            torch.stack((global_rotation, wrist_rotation), dim=1),
        )

    def project_points(
        self,
        points_world: Tensor,
        camera_positions: Tensor,
        camera_rotations: Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> tuple[Tensor, Tensor]:
        """
        将世界坐标系中的 3D 点投影到每个相机的图像平面，并缩放到目标尺寸 (target_size)。

        Args:
            points_world: 世界系中的点，形状 (B, T, 3)（T 为轨迹点数，包括当前点）
            camera_positions: 相机世界系位置，形状 (B, C, 3)
            camera_rotations: 相机世界系旋转矩阵，形状 (B, C, 3, 3)
            image_sizes: 每个相机原始图像尺寸 [(H, W), ...]

        Returns:
            (coordinates, in_front) 元组
            coordinates: 投影后的像素坐标，形状 (B, C, T, 2)，已缩放到 target_size
            in_front: 布尔张量，形状 (B, C, T)，表示点是否在相机前方（深度 > 阈值）
        """
        # 计算点相对于相机的位置：p_cam = R_cam^T * (p_world - p_cam_world)
        relative = points_world[:, None] - camera_positions[:, :, None]  # (B, C, T, 3)
        camera_points = torch.einsum(
            "bcij,bcpj->bcpi",
            camera_rotations.transpose(-1, -2),  # 转置旋转矩阵得到世界到相机的变换
            relative,
        )  # (B, C, T, 3)

        depth = -camera_points[..., 2]
        safe_depth = depth.clamp_min(torch.finfo(points_world.dtype).eps)

        coordinates = []
        target_height, target_width = self.target_size
        for camera, (height, width) in enumerate(image_sizes):
            # 根据垂直视场角计算焦距（以原始图像像素为单位)
            focal = 0.5 * height / math.tan(math.radians(self.camera_fovy_deg[camera]) / 2)

            # 归一化坐标：x = X / depth, y = Y / depth
            x = camera_points[:, camera, :, 0] / safe_depth[:, camera]
            y = camera_points[:, camera, :, 1] / safe_depth[:, camera]
            # 像素坐标（原始图像）
            u_orig = focal * x + width / 2
            v_orig = height / 2 - focal * y  # 注意图像坐标系 y 轴向下
            # 缩放到目标尺寸
            u = u_orig * target_width / width
            v = v_orig * target_height / height
            coordinates.append(torch.stack((u, v), dim=-1))  # (B, T, 2)

        # 堆叠所有相机： (B, C, T, 2), 深度大于极小值认为点在相机前方
        return torch.stack(coordinates, dim=1), depth > 1e-6

    def patch_corridor(self, coordinates: Tensor, point_valid: Tensor) -> tuple[Tensor, Tensor]:
        """
        根据轨迹线段到每个 patch 中心的距离生成二值走廊标签。

        走廊定义：对于每个相机的每个 patch，若其中心到任何轨迹线段的最小距离小于阈值
        （阈值 = corridor_radius_px + patch 半对角线），则标记为正。

        Args:
            coordinates: 投影到图像的轨迹点坐标，形状 (B, C, T, 2)，按时间顺序排列（T 包括当前点）
            point_valid: 布尔张量，形状 (B, C, T)，表示每个轨迹点是否有效（在相机前方且动作非 pad）

        Returns:
            (labels, labels_any) 元组
            labels: 二值张量，形状 (B, C, P)，P = patch_grid^2，表示每个 patch 是否在走廊内
            labels_any: 布尔张量，形状 (B, C)，表示每个相机是否至少有一个有效 patch
        """
        # 轨迹线段端点：start = 前 T-1 个点，end = 后 T-1 个点
        start, end = coordinates[:, :, :-1], coordinates[:, :, 1:]  # (B, C, T-1, 2)
        # 线段有效性：两端点均有效
        segments = point_valid[:, :, :-1] & point_valid[:, :, 1:]  # (B, C, T-1)

        # 生成 patch 中心坐标（在 target_size 图像上均匀分布）
        patch_size = self.target_size[0] / self.patch_grid
        axis = (
            torch.arange(self.patch_grid, device=coordinates.device, dtype=coordinates.dtype) + 0.5
        )
        axis = axis * patch_size  # 每个 patch 的中心坐标
        vertical, horizontal = torch.meshgrid(axis, axis, indexing="ij")
        # 将二维网格展平为 (P, 2)，其中 P = patch_grid^2
        centers = torch.stack((horizontal.flatten(), vertical.flatten()), dim=-1)  # (P, 2)

        # 计算每个 patch 中心到每条线段的最短距离
        direction = end - start  # (B, C, T-1, 2)
        # center_delta: (B, C, T-1, P, 2)，patch 中心相对于线段起点的向量
        center_delta = centers[None, None, None] - start[:, :, :, None]
        # 计算线段上投影参数 t，并限制在 [0,1]
        denominator = direction.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (B, C, T-1, 1)
        fraction = (center_delta * direction[:, :, :, None]).sum(dim=-1) / denominator
        # 最近点 = start + t * direction
        closest = start[:, :, :, None] + fraction.clamp(0, 1)[..., None] * direction[:, :, :, None]
        # 距离 = ||patch中心 - 最近点||
        distance = torch.linalg.vector_norm(centers[None, None, None] - closest, dim=-1)
        # 无效线段对应的距离置为无穷大
        distance = distance.masked_fill(~segments[..., None], torch.inf)

        # patch 与 corridor 相交即可为正；半对角线补偿中心点近似。
        threshold = self.corridor_radius_px + patch_size / math.sqrt(2)
        # 每个相机是否至少有一个正 patch
        labels = distance.amin(dim=2) <= threshold
        return labels, labels.any(dim=-1)

    def targets(
        self,
        state: Tensor,
        actions: Tensor,
        action_is_pad: Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> dict[str, Tensor]:
        """
        构造 batch 中每个样本的几何监督标签。

        Args:
            state: 当前状态张量，形状 (B, S)，包含 TCP 位置和四元数
            actions: 动作序列张量，形状 (B, T_a, 9)，每步为 [dx,dy,dz, 6D_rotation]（相对或绝对？）
            action_is_pad: 布尔张量，形状 (B, T_a)，标记动作是否为填充（无效）
            image_sizes: 每个相机原始图像尺寸列表

        Returns:
            字典，包含 SEAM_LABELS, SEAM_VALID, TANGENT_TARGET, TANGENT_VALID,
            ORIENTATION_TARGET, ORIENTATION_VALID 等键
        """
        # 有效动作 mask
        valid_actions = ~action_is_pad.bool()
        # 构建轨迹点：当前 TCP 位置 + 每个动作的平移部分（假设动作为绝对位置或增量？）
        # 注意：这里 actions[..., :3] 通常为绝对末端执行器位置（在 LeRobot 中默认动作是绝对状态）
        positions = torch.cat((state[:, None, 6:9], actions[..., :3]), dim=1)  # (B, 1+T_a, 3)

        # 轨迹点有效性：当前点总是有效，后续点取决于动作是否有效
        point_steps = torch.cat(
            (
                torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device),
                valid_actions,
            ),
            dim=1,
        )  # (B, 1+T_a)

        # 获取相机位姿
        camera_positions, camera_rotations = self.camera_poses(state)

        # 投影轨迹点到图像
        coordinates, in_front = self.project_points(
            positions,
            camera_positions,
            camera_rotations,
            image_sizes,
        )  # coordinates: (B, C, 1+T_a, 2), in_front: (B, C, 1+T_a)

        # 生成焊缝走廊标签
        seam_labels, seam_valid = self.patch_corridor(
            coordinates,
            in_front & point_steps[:, None],
        )  # seam_labels: (B, C, P), seam_valid: (B, C)

        # ---- 构造切向目标 ----
        # 最后一个有效动作的平移位置
        counts = valid_actions.sum(dim=1)  # (B,)
        last = (counts - 1).clamp_min(0)  # 最后一个有效动作的索引，若无有效动作则为 0
        last_positions = actions[..., :3].gather(
            1,
            last[:, None, None].expand(-1, 1, 3),
        )[:, 0]  # (B, 3)

        # 当前 TCP 旋转矩阵
        tcp_rotation = quaternion_to_matrix(state[:, 9:13])  # (B, 3, 3)

        # 切向向量 = 从当前 TCP 指向最终位置的向量在 TCP 坐标系下的表示
        tangent = (tcp_rotation.transpose(-1, -2) @ (last_positions - state[:, 6:9])[..., None])[
            ..., 0
        ]  # (B, 3)

        # 切向有效性：至少一个有效动作且向量模长足够大
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
        tangent_valid = (counts > 0) & (tangent_norm > 1e-6)  # (B,)
        # 归一化切向
        tangent = tangent / tangent_norm.clamp_min(1e-6)[:, None]

        # ---- 构造相对姿态目标 ----
        # 第一个有效动作的旋转（6D 表示）
        first = valid_actions.to(torch.int64).argmax(dim=1)  # 第一个有效动作索引
        first_orientation = actions[..., 3:9].gather(
            1,
            first[:, None, None].expand(-1, 1, 6),
        )[:, 0]  # (B, 6)
        # 转换为旋转矩阵
        target_rotation = rotation_from_6d_rows(first_orientation)  # (B, 3, 3)
        # 相对旋转 = TCP 旋转的转置 @ 目标旋转（即目标在 TCP 坐标系下的旋转）
        relative_rotation = tcp_rotation.transpose(-1, -2) @ target_rotation  # (B, 3, 3)

        # 返回标签字典
        return {
            SEAM_LABELS: seam_labels.to(torch.float32),  # 转换为 float 用于 BCE
            SEAM_VALID: seam_valid,
            TANGENT_TARGET: tangent.to(torch.float32),
            TANGENT_VALID: tangent_valid,
            ORIENTATION_TARGET: rotation_to_6d_rows(relative_rotation).to(torch.float32),
            ORIENTATION_VALID: counts > 0,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        处理器步骤的调用入口。训练 batch 有动作时添加标签；部署观测保持原样。

        Args
            transition: EnvTransition 对象，包含 observation、action 等字段

        Returns:
            更新后的 EnvTransition（添加了几何标签）
        """
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        if observation is None or action is None or OBS_STATE not in observation:
            return transition
        state = cast(Tensor, observation[OBS_STATE])
        actions = cast(Tensor, action)
        # 获取填充标记，若缺失则假定全 False
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        default_padding = torch.zeros(actions.shape[:2], dtype=torch.bool)
        action_is_pad = complementary.get("action_is_pad", default_padding)
        action_is_pad = torch.as_tensor(action_is_pad, device=actions.device)
        image_sizes = []
        for key in self.camera_keys:
            height, width = cast(Tensor, observation[key]).shape[-2:]
            image_sizes.append((int(height), int(width)))

        # 复制 transition 并添加几何标签到观察中
        updated = cast(Any, transition.copy())
        updated[TransitionKey.OBSERVATION] = {
            **observation,
            **self.targets(state, actions, action_is_pad, image_sizes),
        }
        return updated

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        """辅助张量不参与 LeRobot feature 推断和归一化。"""
        return features


class GeometryGroundingHeads(nn.Module):
    """
    将三个几何 readout 映射为 seam、tangent 与 orientation 预测的模块。

    输入为辅助输出中的 readout_tokens 和 patch_tokens，输出预测结果，
    供损失函数使用。
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = hidden_size**-0.5
        self.seam_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.seam_patch = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tangent = nn.Linear(hidden_size, 3)
        self.orientation = nn.Linear(hidden_size, 6)

    def forward(self, auxiliary: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        前向传播，输入辅助特征字典，输出预测结果。

        Args:
            auxiliary: 包含 "geometry.readout_tokens" 和 "geometry.patch_tokens" 的字典。
                - readout_tokens: 形状 (B, 3, hidden_size)
                        ，三个 token 分别用于 seam、tangent、orientation
                - patch_tokens: 形状 (B, C, P, hidden_size)，图像 patch 的特征

        Returns:
            字典，包含：
                - "geometry.seam_logits": 形状 (B, C, P)，每个 patch 的 logits
                - "geometry.tangent_prediction": 形状 (B, 3)，归一化切向预测
                - "geometry.orientation_6d": 形状 (B, 6)，姿态 6D 表示
        """
        readout = auxiliary["geometry.readout_tokens"]
        patches = auxiliary["geometry.patch_tokens"]
        seam_logits = (
            torch.einsum(
                "bd,bcpd->bcp",
                self.seam_query(readout[:, 0]),
                self.seam_patch(patches),
            )
            * self.scale
        )
        return {
            "geometry.seam_logits": seam_logits,
            "geometry.tangent_prediction": F.normalize(self.tangent(readout[:, 1]), dim=-1),
            "geometry.orientation_6d": self.orientation(readout[:, 2]),
        }


def masked_sample_mean(values: Tensor, mask: Tensor) -> Tensor:
    """
    将任意尾部维度按 mask 约简为逐样本均值。

    Args:
        values: 形状 (B, ...) 的张量
        mask: 与 values 形状相同的布尔张量，True 表示计入均值

    Returns:
        形状为 (B,) 的张量，每个样本在 mask 为 True 的位置的平均值
    """
    weights = mask.to(values.dtype)
    dimensions = tuple(range(1, values.ndim))
    return (values * weights).sum(dim=dimensions) / weights.sum(dim=dimensions).clamp_min(1)


def geometry_grounding_losses(
    auxiliary: dict[str, Tensor],
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    # ---- Seam 损失：逐 patch 的二值交叉熵，仅在有效 patch 上计算 ----
    logits = auxiliary["geometry.seam_logits"]  # (B, C, P)
    labels = batch[SEAM_LABELS].to(logits)  # (B, C, P) 二值标签
    patch_mask = auxiliary[
        "geometry.patch_mask"
    ].bool()  # (B, C, P) 有效 patch mask（可能来自模型输出）
    seam_mask = batch[SEAM_VALID].bool()[..., None] & patch_mask  # 结合有效相机和有效 patch
    # 逐元素 BCE（不 reduction）
    seam_values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    # 对每个样本，平均有效 patch 上的损失
    seam = masked_sample_mean(seam_values, seam_mask)

    # ---- 切向损失：1 - 余弦相似度，仅在切向有效时计算 ----
    tangent_prediction = auxiliary["geometry.tangent_prediction"]  # (B, 3)
    tangent_target = batch[TANGENT_TARGET].to(tangent_prediction)  # (B, 3)
    tangent_values = 1 - (tangent_prediction * tangent_target).sum(dim=-1)  # (B,)
    tangent_mask = batch[TANGENT_VALID].bool()  # (B,)
    tangent = tangent_values * tangent_mask.to(tangent_values.dtype)

    # ---- 姿态损失：SO(3) 测地线距离（旋转角度） ----
    predicted_rotation = rotation_from_6d_rows(auxiliary["geometry.orientation_6d"])  # (B, 3, 3)
    target_rotation = rotation_from_6d_rows(
        batch[ORIENTATION_TARGET].to(predicted_rotation)
    )  # (B, 3, 3)
    # 相对旋转矩阵：R_rel = R_pred^T @ R_target
    relative = predicted_rotation.transpose(-1, -2) @ target_rotation  # (B, 3, 3)
    # 计算旋转角度的余弦值
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).clamp(-1, 1)  # (B,)
    # 计算旋转向量的反对称部分（用于得到 sin 分量）
    skew = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )  # (B, 3)
    # 旋转角 = atan2(||skew||, cosine)
    orientation_values = torch.atan2(torch.linalg.vector_norm(skew, dim=-1), cosine)  # (B,)
    orientation_mask = batch[ORIENTATION_VALID].bool()
    orientation = orientation_values * orientation_mask.to(orientation_values.dtype)

    return {"seam": seam, "tangent": tangent, "orientation": orientation}


__all__ = [
    "ORIENTATION_TARGET",
    "ORIENTATION_VALID",
    "SEAM_LABELS",
    "SEAM_VALID",
    "TANGENT_TARGET",
    "TANGENT_VALID",
    "GeometryGroundingHeads",
    "GeometryGroundingTargetProcessorStep",
    "geometry_grounding_losses",
]
