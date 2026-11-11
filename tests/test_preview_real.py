"""验证官方预览循环只读、不触碰数据集并正确清理资源。"""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from lerobot.scripts import lerobot_teleoperate
from lerobot_robot_welding import adapter
from test_real_capture import make_robot


@pytest.fixture
def preview(monkeypatch, tmp_path):
    """仅替换设备 I/O 和图形后端，保留原生循环与生产适配器。"""
    monkeypatch.syspath_prepend("scripts")
    command = import_module("preview_real")
    _, _, cps = make_robot(monkeypatch, tmp_path / "unused")
    created, observations, visualization = [], [], []
    robot_class = command.ElfinWelding

    def make_device(config):
        robot = robot_class(config)
        created.append(robot)
        return robot

    monkeypatch.setattr(command, "ElfinWelding", make_device)
    monkeypatch.setattr(
        command, "init_visualization", lambda *a, **kw: visualization.append("init")
    )
    monkeypatch.setattr(
        command, "shutdown_visualization", lambda mode: visualization.append("close")
    )
    monkeypatch.setattr(
        lerobot_teleoperate,
        "log_visualization_data",
        lambda mode, **values: observations.append(values["observation"]),
    )
    settings = yaml.safe_load(Path("configs/real_record.yaml").read_text())
    dataset = tmp_path / "dataset"
    settings["dataset"]["root"] = str(dataset)
    settings["dataset"]["rgb_encoder"]["vcodec"] = "not_a_codec"
    settings["task_conditions"]["direction"] = None
    settings["display_data"] = False
    for camera in settings["robot"]["cameras"].values():
        camera.update(width=32, height=24)
    path = tmp_path / "record.yaml"
    path.write_text(yaml.safe_dump(settings))
    return SimpleNamespace(
        command=command,
        cps=cps,
        created=created,
        observations=observations,
        visualization=visualization,
        path=path,
        dataset=dataset,
    )


@pytest.mark.parametrize("existing_dataset", [False, True])
def test_official_loop_preview_is_read_only(preview, existing_dataset):
    """预览输出双相机和状态，只调用连接/读取/断开且不创建或更改数据。"""
    if existing_dataset:
        meta = preview.dataset / "meta"
        meta.mkdir(parents=True)
        (meta / "welding_path_vla_export.json").write_text('{"keep": true}')
        (meta / "stats.json").write_text('{"keep": true}')
    before = {p: p.read_bytes() for p in preview.dataset.rglob("*") if p.is_file()}
    preview.command.preview_devices(
        preview.command.PreviewArguments(record_config=preview.path, duration_s=0.1)
    )
    assert len(preview.observations) >= 2
    for obs in preview.observations:
        assert obs["global"].shape == obs["wrist"].shape == (24, 32, 3)
        assert "joint_1" in obs and "tcp_qw" in obs
    assert set(preview.cps.calls) == {"connect", "read", "disconnect"}
    assert preview.cps.calls[-1] == "disconnect"
    assert preview.visualization == ["init", "close"]
    robot = preview.created[0]
    assert robot.dataset_root is None and robot.log is None
    assert not robot.connected and not adapter.CAPTURE_SESSIONS
    assert all(not camera.is_connected for camera in robot.cameras.values())
    assert before == {p: p.read_bytes() for p in preview.dataset.rglob("*") if p.is_file()}
    assert preview.dataset.exists() == existing_dataset


@pytest.mark.parametrize("error", [RuntimeError("预览后端错误"), KeyboardInterrupt()])
def test_preview_loop_failure_releases_devices(preview, monkeypatch, error):
    """可视化失败或用户中断时，关闭图形后端及全部设备，不写日志。"""

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(lerobot_teleoperate, "log_visualization_data", fail)
    with pytest.raises(type(error)):
        preview.command.preview_devices(
            preview.command.PreviewArguments(record_config=preview.path)
        )
    assert preview.visualization == ["init", "close"]
    assert preview.cps.calls[-1] == "disconnect"
    assert not preview.created[0].connected and not adapter.CAPTURE_SESSIONS
    assert all(not camera.is_connected for camera in preview.created[0].cameras.values())
    assert not preview.dataset.exists()


def test_preview_camera_connect_failure_cleans_up(preview, monkeypatch):
    """相机连接失败也断开 SDK 和已连接的相机，不启动可视化。"""
    original = preview.command.ElfinWelding

    def make_failing_device(config):
        robot = original(config)

        def fail():
            raise RuntimeError("USB camera unavailable")

        monkeypatch.setattr(robot.cameras["wrist"], "connect", fail)
        return robot

    monkeypatch.setattr(preview.command, "ElfinWelding", make_failing_device)
    with pytest.raises(RuntimeError, match="USB"):
        preview.command.preview_devices(
            preview.command.PreviewArguments(record_config=preview.path)
        )
    assert preview.cps.calls[-1] == "disconnect"
    assert all(not camera.is_connected for camera in preview.created[0].cameras.values())
    assert not preview.visualization and not preview.dataset.exists()


def test_preview_rejects_non_passive_teleop_before_connect(preview):
    """只允许项目被动适配器，避免使用运动控制型设备。"""
    settings = yaml.safe_load(preview.path.read_text())
    settings["teleop"] = {"type": "keyboard"}
    preview.path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="只读预览"):
        preview.command.preview_devices(
            preview.command.PreviewArguments(record_config=preview.path)
        )
    assert not preview.cps.calls and not preview.created
