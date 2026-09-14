from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.lerobot_training import make_train_config
from welding_path_vla.policies.runtime import apply_inference_overrides
from welding_path_vla.policies.spec import XVLA, get_policy_spec


def test_xvla_uses_official_lerobot_implementation() -> None:
    """X-VLA 应只注册官方类，不在项目中复制模型实现。"""
    assert get_policy_spec("xvla") is XVLA
    assert XVLA.pretrained_model == "lerobot/xvla-base"
    assert XVLA.config_class().__name__ == "XVLAConfig"
    assert XVLA.policy_class().__name__ == "XVLAPolicy"
    assert XVLA.processor_adds_batch
    assert not XVLA.use_imagenet_stats


def test_xvla_a100_profile_covers_training_split_once() -> None:
    """The user-selected 1,000,000 steps cover the current v3 split at least once."""
    config = AppConfig.load("configs/xvla_a100.yaml")
    parameters = config.policy.parameters

    assert config.policy.family == "xvla"
    assert parameters["pretrained_model"] == "lerobot/xvla-base"
    assert parameters["action_mode"] == "auto"
    assert parameters["max_action_dim"] == 20
    assert parameters["num_image_views"] == 3
    assert parameters["dtype"] == "bfloat16"
    assert not parameters["freeze_vision_encoder"]
    assert not parameters["freeze_language_encoder"]
    assert parameters["train_policy_transformer"]
    assert parameters["train_soft_prompts"]
    assert config.training.batch_size == 4
    assert config.training.dataset_root == "datasets/lerobot/weldpath_lerobot_relative_v3"
    assert config.training.dataset_repo_id == "mintinson/weldpath_relative_v1"
    assert config.training.steps == 1_000_000
    assert parameters["scheduler_decay_steps"] == 1_000_000
    assert config.training.peft is None
    train_config = make_train_config(config.policy, config.training, XVLA)
    assert not train_config.dataset.use_imagenet_stats


def test_xvla_denoising_steps_can_be_overridden_for_deployment() -> None:
    """统一部署参数应映射到 X-VLA 的去噪步数字段。"""

    class Config:
        chunk_size = 30
        n_action_steps = 8
        num_denoising_steps = 10

    config = Config()
    apply_inference_overrides(config, action_steps=6, inference_steps=5)

    assert config.n_action_steps == 6
    assert config.num_denoising_steps == 5
