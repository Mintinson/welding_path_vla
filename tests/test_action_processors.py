import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.processor import UnnormalizerProcessorStep
from lerobot.processor.converters import TransitionKey
from scipy.spatial.transform import Rotation

from welding_path_vla.core.geometry import (
    absolute_ee_actions_from_relative,
    relative_ee_actions_from_absolute,
)
from welding_path_vla.policies.action_processors import (
    AbsoluteEEActionsProcessorStep,
    Float32ActionProcessorStep,
    RelativeEEActionsProcessorStep,
    connect_relative_processors,
    require_relative_checkpoint,
)


def rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """取旋转矩阵前两行，得到项目使用的 rotation-6D。"""
    return matrix[:2].reshape(6)


def test_relative_action_chunk_roundtrip_uses_one_tcp_anchor() -> None:
    """整段 future targets 应共享预测时刻 TCP，而不是逐步更换锚点。"""
    anchor_position = torch.tensor([[0.4, -0.2, 0.3]], dtype=torch.float64)
    anchor_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    anchor_quaternion_xyzw = Rotation.from_matrix(anchor_rotation).as_quat()
    anchor_quaternion = torch.tensor(
        [[anchor_quaternion_xyzw[3], *anchor_quaternion_xyzw[:3]]], dtype=torch.float64
    )
    targets = np.stack(
        (
            np.r_[0.4, -0.1, 0.3, rotation_6d(anchor_rotation)],
            np.r_[
                0.4,
                0.0,
                0.3,
                rotation_6d(Rotation.from_euler("z", 120, degrees=True).as_matrix()),
            ],
        )
    )
    absolute = torch.tensor(targets[None], dtype=torch.float64, requires_grad=True)

    relative = relative_ee_actions_from_absolute(absolute, anchor_position, anchor_quaternion)
    restored = absolute_ee_actions_from_relative(relative, anchor_position, anchor_quaternion)

    np.testing.assert_allclose(relative.detach()[0, :, :3], [[0.1, 0, 0], [0.2, 0, 0]], atol=1e-8)
    torch.testing.assert_close(restored, absolute)
    gradient = torch.autograd.grad(relative.square().sum(), absolute)[0]
    assert torch.all(torch.isfinite(gradient))


def test_old_checkpoint_is_not_silently_resumed(tmp_path: Path) -> None:
    """旧 checkpoint 缺少动作契约时必须要求重新训练。"""
    config = tmp_path / "policy_preprocessor.json"
    config.write_text(json.dumps({"steps": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="旧 delta action"):
        require_relative_checkpoint(tmp_path)

    config.write_text(
        json.dumps({"steps": [{"registry_name": "welding_relative_ee_actions"}]}),
        encoding="utf-8",
    )
    require_relative_checkpoint(tmp_path)


def test_absolute_processor_aligns_anchor_with_policy_action() -> None:
    """动作解码统一使用 FP32，并把锚点放到动作所在设备。"""
    relative_step = RelativeEEActionsProcessorStep()
    relative_step.anchor_positions = torch.zeros((1, 3), dtype=torch.float32)
    relative_step.anchor_quaternions = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    processor = AbsoluteEEActionsProcessorStep(relative_step)
    action = torch.tensor([[[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0]]], dtype=torch.float64)

    converted = processor({TransitionKey.ACTION: action})

    absolute = converted[TransitionKey.ACTION]
    assert absolute.dtype == torch.float32
    torch.testing.assert_close(absolute[..., :3], action[..., :3].float())


def test_absolute_processor_keeps_submillimeter_motion_in_float32() -> None:
    """BF16 policy action 解码时不能把世界坐标和小步位移一起量化。"""
    relative_step = RelativeEEActionsProcessorStep()
    relative_step.anchor_positions = torch.tensor([[0.3003, 0.1001, 0.2961]], dtype=torch.float32)
    relative_step.anchor_quaternions = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    processor = AbsoluteEEActionsProcessorStep(relative_step)
    action = torch.tensor(
        [[[0.0007, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.bfloat16
    )

    absolute = processor({TransitionKey.ACTION: action})[TransitionKey.ACTION]

    assert absolute.dtype == torch.float32
    torch.testing.assert_close(
        absolute[..., :3],
        torch.tensor([[[0.3010, 0.1001, 0.2961]]], dtype=torch.float32),
        atol=2e-4,
        rtol=0,
    )


def test_connect_inserts_float32_before_unnormalizer() -> None:
    """FP32 边界必须位于反归一化之前，避免统计量缩放仍在 BF16 中执行。"""
    from types import SimpleNamespace

    preprocessor = SimpleNamespace(steps=[RelativeEEActionsProcessorStep()])
    postprocessor = SimpleNamespace(
        steps=[UnnormalizerProcessorStep(features={}, norm_map={}, stats={})]
    )

    connect_relative_processors(preprocessor, postprocessor, require_saved=False)

    float32_index = next(
        index
        for index, step in enumerate(postprocessor.steps)
        if isinstance(step, Float32ActionProcessorStep)
    )
    unnormalizer_index = next(
        index
        for index, step in enumerate(postprocessor.steps)
        if isinstance(step, UnnormalizerProcessorStep)
    )
    absolute_index = next(
        index
        for index, step in enumerate(postprocessor.steps)
        if isinstance(step, AbsoluteEEActionsProcessorStep)
    )
    assert float32_index < unnormalizer_index < absolute_index
