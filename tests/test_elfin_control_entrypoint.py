"""验证八字入口、相机回退及 ServoP 下发边界。"""

import json
import sys
from importlib import import_module
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot_robot_welding import (
    BlackCamera,
    BlackCameraConfig,
    ElfinWelding,
    ElfinWeldingConfig,
    adapter,
)
from scipy.spatial.transform import Rotation

from welding_path_vla.core.geometry import rotation_to_6d_rows
from welding_path_vla.dataset.actions import ABSOLUTE_ACTION_NAMES


class FakeCPS:
    """只实现连接、观测和实际用到的伺服指令。"""

    def __init__(self):
        self.pose = [-191.673, 365.479, 164.605, 0.0, 0.0, 0.0]
        self.host = None
        self.calls = []
        self.targets = []

    def HRIF_Connect(self, box, host, port):
        self.host = host
        return 0

    def HRIF_DisConnect(self, box):
        self.calls.append("disconnect")
        return 0

    def HRIF_ReadActPos(self, box, robot, result):
        result.extend([0.0] * 6 + self.pose + [0.0] * 12)
        return 0

    def HRIF_StartServo(self, box, robot, period, lookahead):
        self.calls.append("start")
        assert period == 0.01
        return 0

    def HRIF_PushServoP(self, box, robot, target, tcp, ucs):
        self.calls.append("push")
        self.targets.append(target)
        self.pose = target
        return 0

    def HRIF_GrpStop(self, box, robot):
        self.calls.append("stop")
        return 0


@pytest.fixture
def devices(monkeypatch):
    """保留一路图像，另一相机连接失败后交给适配器回退。"""

    class Camera(BlackCamera):
        def read_latest(self, max_age_ms=500):
            return np.full_like(super().read_latest(), 123)

    class UnavailableCamera(BlackCamera):
        def connect(self, warmup=True):
            raise OSError("camera unavailable")

    cps = FakeCPS()
    cameras = {
        "global": Camera(BlackCameraConfig()),
        "wrist": UnavailableCamera(BlackCameraConfig()),
    }
    monkeypatch.setattr(adapter, "CPSClient", lambda: cps)
    monkeypatch.setattr(adapter, "make_cameras_from_configs", lambda configs: dict(cameras))
    return cps, cameras


@pytest.mark.parametrize("execute", [False, True])
def test_figure_eight_cli(monkeypatch, capsys, devices, tmp_path, execute):
    """使用录制设备配置及项目策略字段执行八字。"""
    monkeypatch.syspath_prepend("scripts")
    command = import_module("test_elfin_control")
    cps, _ = devices
    if not execute:
        cps.pose[1] = 490.0  # 只读演练不会在观测或规划时判定越界。
    profile = tmp_path / "policy.yaml"
    profile.write_text(
        f"includes: [{Path('configs/base.yaml').resolve()}]\n"
        "policy: {action_horizon: 5, action_steps: 3, action_stride: 2}\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_elfin_control.py",
            "--host=192.168.56.101",
            f"--app_config={profile}",
            "--duration_s=0.4",
            f"--execute={str(execute).lower()}",
        ],
    )
    command.main()
    report = json.loads(capsys.readouterr().out)
    assert cps.host == "192.168.56.101"
    assert (report["state_size"], report["chunks"]) == (13, 2)
    assert (report["action_hz"], report["action_horizon"], report["action_steps"]) == (15, 5, 3)
    assert report["cameras"]["wrist"] == "black"
    if execute:
        assert cps.calls[0] == "start" and cps.calls[-2:] == ["stop", "disconnect"]
        assert len(cps.targets) > 12
    else:
        assert cps.calls == ["disconnect"]


@pytest.mark.parametrize("outside", [False, True])
def test_checkpoint_cli_uses_same_servo_and_interpolates_orientation(
    monkeypatch, capsys, devices, outside
):
    """换成 checkpoint 后，ServoP 链路仍执行策略输出的平移和旋转。"""
    monkeypatch.syspath_prepend("scripts")
    command = import_module("test_elfin_control")
    runtime_module = import_module("welding_path_vla.policies.runtime")
    cps, _ = devices
    predictions = []
    rotation = rotation_to_6d_rows(Rotation.from_euler("z", -60, degrees=True).as_matrix())

    def predict(observation):
        predictions.append(observation)
        action = np.r_[observation.state[6:9], rotation]
        if outside:
            action[0] += 1.0
        return np.broadcast_to(action, (8, 9))

    def load(checkpoint, device, spec, **options):
        assert checkpoint == "checkpoint" and options["action_steps"] == 8
        return SimpleNamespace(
            reset=lambda: None,
            policy=SimpleNamespace(config=SimpleNamespace(chunk_size=30)),
            predict_absolute_chunk=predict,
        )

    monkeypatch.setattr(runtime_module.LeRobotRuntime, "from_pretrained", load)
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_elfin_control.py", "--checkpoint=checkpoint", "--duration_s=0.4", "--execute=true"],
    )
    if outside:
        with pytest.raises(ValueError, match="ServoP 目标超出 Base TCP 边界"):
            command.main()
        assert all(target[1] <= 450 for target in cps.targets)
    else:
        command.main()
        report = json.loads(capsys.readouterr().out)
        assert report["chunks"] == len(predictions) == 2
        assert predictions[0].instruction and set(predictions[0].images) == {"global", "wrist"}
        assert any(0.1 < target[5] < 29.9 for target in cps.targets)
        assert cps.targets[-1][5] == pytest.approx(30)
    assert cps.calls[-2:] == ["stop", "disconnect"]


@pytest.mark.parametrize("outside", [False, True])
def test_native_rollout_preserves_chunk_anchor_and_stops_servo(
    monkeypatch, devices, tmp_path, outside
):
    """经过官方参数解析和 BaseStrategy，用同一 TCP 锚点解码块内两点。"""
    import torch
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
    )
    from lerobot.rollout import context
    from lerobot.scripts import lerobot_rollout

    from welding_path_vla.policies.action_processors import (
        AbsoluteEEActionsProcessorStep,
        RelativeEEActionsProcessorStep,
    )
    from welding_path_vla.policies.trajectory_vla.configuration_trajectory_vla import (
        TrajectoryVLAConfig,
    )

    cps, _ = devices
    TrajectoryVLAConfig(device="cpu", chunk_size=3, n_action_steps=2).save_pretrained(tmp_path)
    PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device="cpu"),
            RelativeEEActionsProcessorStep(),
        ]
    ).save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    PolicyProcessorPipeline(steps=[AbsoluteEEActionsProcessorStep()]).save_pretrained(
        tmp_path, config_filename="policy_postprocessor.json"
    )

    def load(checkpoint, config):
        def predict(observation):
            assert observation["observation.state"].shape == (1, 13)
            actions = torch.tensor([0.0, 0, 0, 1, 0, 0, 0, 1, 0]).repeat(1, 3, 1)
            actions[0, :, 0] = torch.tensor([1.0 if outside else 0.001, 0.003, 1.0])
            return actions

        policy = SimpleNamespace(config=config, predict_action_chunk=predict, reset=lambda: None)
        policy.to = lambda device: policy
        policy.eval = lambda: policy
        return policy

    sent = []
    send_action = ElfinWelding.send_action

    def capture_action(robot, action):
        sent.append(action)
        return send_action(robot, action)

    monkeypatch.setattr(ElfinWelding, "send_action", capture_action)
    monkeypatch.setattr(
        context, "get_policy_class", lambda name: SimpleNamespace(from_pretrained=load)
    )
    monkeypatch.setattr(
        lerobot_rollout,
        "ProcessSignalHandler",
        lambda **_: SimpleNamespace(shutdown_event=Event()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-rollout",
            "--strategy.type=base",
            "--robot.type=elfin_welding",
            "--robot.active=true",
            f"--policy.path={tmp_path}",
            "--device=cpu",
            "--duration=0.35",
            "--play_sounds=false",
            "--return_to_initial_position=false",
            "--task=Weld along the seam.",
        ],
    )
    if outside:
        with pytest.raises(ValueError, match="ServoP 目标超出 Base TCP 边界"):
            lerobot_rollout.main()
        assert all(-400 <= target[0] <= 200 for target in cps.targets)
    else:
        lerobot_rollout.main()
        assert len(sent) >= 2
        assert sent[0]["y.pos"] - sent[1]["y.pos"] == pytest.approx(0.002, abs=1e-6)
        assert len(cps.targets) > len(sent)
    assert cps.calls[:2] == ["start", "push"] and cps.calls[-2:] == ["stop", "disconnect"]


@pytest.mark.parametrize("outside", [(-400.1, 300, 100), (0, 450.1, 100), (0, 300, -15.1)])
def test_active_robot_rejects_only_out_of_bounds_send(devices, outside):
    """读取越界反馈不会中止，越界目标在 PushServoP 前停止会话。"""
    cps, cameras = devices
    cps.pose[0] = -500
    robot = ElfinWelding(ElfinWeldingConfig(active=True))
    with robot:
        observation = robot.get_observation()
        assert observation["global"][0, 0, 0] == 123
        assert robot.cameras["global"] is cameras["global"]
        assert not observation["wrist"].any() and cps.calls == []

        def action(position_mm):
            position = robot.reader.base_rotation @ (np.array(position_mm) / 1000)
            position += robot.reader.base_position
            return dict(zip(ABSOLUTE_ACTION_NAMES, (*position, 1, 0, 0, 0, 1, 0), strict=True))

        robot.start_servo()
        robot.send_action(action((-250, 320, 200)))
        robot.send_action(action((-400, 200, -15)))
        with pytest.raises(ValueError, match="ServoP 目标超出 Base TCP 边界"):
            robot.send_action(action(outside))
        assert cps.calls == ["start", "push", "push", "stop"]
        with pytest.raises(RuntimeError, match="请先启动 ServoP"):
            robot.send_action(action((-250, 320, 200)))
        np.testing.assert_allclose(cps.targets[0][:3], [-250, 320, 200])
        assert cps.targets[1][:3] == [-400, 200, -15]
    assert cps.calls == ["start", "push", "push", "stop", "disconnect"]
