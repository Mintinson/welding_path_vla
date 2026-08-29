"""LeRobot 策略在项目中的声明式差异。"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class LeRobotPolicySpec:
    """一个策略接入公共训练、评估和部署流程所需的最小信息。

    Attributes:
        family: 项目 YAML 使用的策略名称。
        policy_type: LeRobot 配置内部使用的名称。
        display_name: 日志和错误信息使用的名称。
        config_class_path: 配置类的完整导入路径。
        policy_class_path: 模型类的完整导入路径。
        config_mode: 从零构造、读取官方配置或用本地配置承接预训练权重。
        pretrained_model: 默认基础 checkpoint；从零训练时为 ``None``。
        language: 在线观测是否包含自然语言任务。
        processor_adds_batch: processor 是否负责添加 batch 维。
        return_uint8: 训练数据是否保持 uint8 图像交给 processor。
        explicit_mixed_precision: 是否由项目显式配置 Accelerator 混合精度。
        implementation: 报告中区分官方模型与项目本地模型。
        evaluation_values: 除总 loss 外需要汇总的模型日志字段。
        plan_fields: 训练计划中额外公开的配置字段。
    """

    family: str
    policy_type: str
    display_name: str
    config_class_path: str
    policy_class_path: str
    config_mode: Literal["scratch", "pretrained", "local_pretrained"]
    pretrained_model: str | None = None
    language: bool = True
    processor_adds_batch: bool = False
    return_uint8: bool = True
    explicit_mixed_precision: bool = True
    implementation: str = "official"
    evaluation_values: tuple[str, ...] = ()
    plan_fields: tuple[tuple[str, str], ...] = ()

    def config_class(self) -> type[Any]:
        """惰性加载配置类，避免配置检查提前导入大模型依赖。"""
        module, name = self.config_class_path.rsplit(".", 1)
        return getattr(import_module(module), name)

    def policy_class(self) -> type[Any]:
        """只在训练或推理时加载模型类。"""
        module, name = self.policy_class_path.rsplit(".", 1)
        return getattr(import_module(module), name)


ACT = LeRobotPolicySpec(
    family="act",
    policy_type="act",
    display_name="ACT",
    config_class_path="lerobot.policies.act.configuration_act.ACTConfig",
    policy_class_path="lerobot.policies.act.modeling_act.ACTPolicy",
    config_mode="scratch",
    language=False,
    return_uint8=False,
    evaluation_values=("l1_loss", "kld_loss"),
)

SMOLVLA = LeRobotPolicySpec(
    family="smolvla",
    policy_type="smolvla",
    display_name="SmolVLA",
    config_class_path="lerobot.policies.smolvla.configuration_smolvla.SmolVLAConfig",
    policy_class_path="lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    config_mode="pretrained",
    pretrained_model="lerobot/smolvla_base",
    plan_fields=(
        ("image_resolution", "resize_imgs_with_padding"),
        ("inference_steps", "num_steps"),
    ),
)

TRAJECTORY_VLA = LeRobotPolicySpec(
    family="trajectory_vla",
    policy_type="trajectory_vla",
    display_name="Trajectory-VLA",
    config_class_path=(
        "welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla.TrajectoryVLAConfig"
    ),
    policy_class_path=(
        "welding_path_vla.policies.trajectory_vla.modeling_trajectory_vla.TrajectoryVLAPolicy"
    ),
    config_mode="local_pretrained",
    pretrained_model="lerobot/smolvla_base",
    implementation="local",
    plan_fields=(
        ("image_resolution", "resize_imgs_with_padding"),
        ("inference_steps", "num_steps"),
    ),
)

TRAJ_VLA_QWEN = LeRobotPolicySpec(
    family="traj_vla_qwen",
    policy_type="traj_vla_qwen",
    display_name="Trajectory-VLA Qwen",
    config_class_path=(
        "welding_path_vla.policies.traj_vla_qwen.configuration_traj_vla_qwen.TrajVLAQwenConfig"
    ),
    policy_class_path=(
        "welding_path_vla.policies.traj_vla_qwen.modeling_traj_vla_qwen.TrajVLAQwenPolicy"
    ),
    config_mode="scratch",
    implementation="local",
    plan_fields=(
        ("image_resolution", "resize_imgs_with_padding"),
        ("inference_steps", "num_steps"),
        ("language_model", "language_model_name"),
        ("prismatic_backbone", "prismatic_repo_id"),
        ("language_layers", "num_vlm_layers"),
        ("expert_layers", "num_expert_layers"),
        ("dense_geometry", "use_geometry_branch"),
        ("geometry_queries", "geometry_num_queries"),
        ("geometry_grounding", "use_geometry_grounding"),
        ("motion_latent", "use_motion_latent"),
        ("training_stage", "training_stage"),
    ),
)

PI0 = LeRobotPolicySpec(
    family="pi0",
    policy_type="pi0",
    display_name="π0",
    config_class_path="lerobot.policies.pi0.configuration_pi0.PI0Config",
    policy_class_path="lerobot.policies.pi0.modeling_pi0.PI0Policy",
    config_mode="pretrained",
    pretrained_model="lerobot/pi0_base",
    processor_adds_batch=True,
    explicit_mixed_precision=False,
    plan_fields=(
        ("image_resolution", "image_resolution"),
        ("inference_steps", "num_inference_steps"),
    ),
)

PI05 = LeRobotPolicySpec(
    family="pi0_5",
    policy_type="pi05",
    display_name="π0.5",
    config_class_path="lerobot.policies.pi05.configuration_pi05.PI05Config",
    policy_class_path="lerobot.policies.pi05.modeling_pi05.PI05Policy",
    config_mode="pretrained",
    pretrained_model="lerobot/pi05_base",
    processor_adds_batch=True,
    explicit_mixed_precision=False,
    plan_fields=PI0.plan_fields,
)

POLICY_SPECS = {
    spec.family: spec for spec in (ACT, SMOLVLA, TRAJECTORY_VLA, TRAJ_VLA_QWEN, PI0, PI05)
}


def get_policy_spec(family: str) -> LeRobotPolicySpec:
    """返回策略规格，并为未知名称提供一致错误。"""
    try:
        return POLICY_SPECS[family]
    except KeyError as error:
        raise ValueError(f"policy pipeline is not implemented: {family}") from error


__all__ = [
    "ACT",
    "PI0",
    "PI05",
    "POLICY_SPECS",
    "SMOLVLA",
    "TRAJECTORY_VLA",
    "TRAJ_VLA_QWEN",
    "LeRobotPolicySpec",
    "get_policy_spec",
]
