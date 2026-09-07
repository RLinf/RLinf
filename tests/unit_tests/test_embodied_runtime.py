# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import rlinf.envs.realworld.common.camera as camera_module
from rlinf.envs.realworld.common.camera import CameraInfo
from rlinf.envs.realworld.common.camera.embodied_runtime_camera import (
    EmbodiedRuntimeCamera,
)
from rlinf.scheduler import EmbodiedRuntimeCLI
from rlinf.scheduler.hardware.robots.franka import FrankaConfig, FrankaRobot


def _result(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload).encode())


class _RuntimeCLI:
    def __init__(self, tool: str):
        self.tool = tool

    @staticmethod
    def is_enabled(tool: str) -> bool:
        return tool in ("camctr", "rosctr")

    def list_robots(self):
        return [
            {
                "robotId": "franka-0",
                "params": {"robot_ip": "172.16.0.2"},
            }
        ]

    def list_cameras(self):
        return [
            {
                "cameraId": "video0",
                "serialNumber": "cam-123",
            }
        ]

    def resolve_robot_id(self, robot_ip):
        assert robot_ip == "172.16.0.2"
        return "franka-0"

    def resolve_camera_id(self, identifier):
        assert identifier == "cam-123"
        return "video0"


def _load_ros_controller(monkeypatch):
    rospy = types.ModuleType("rospy")
    setattr(rospy, "init_node", MagicMock())
    setattr(rospy, "Publisher", MagicMock())
    setattr(rospy, "Subscriber", MagicMock())
    setattr(rospy, "Message", object)
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    module = importlib.import_module("rlinf.envs.realworld.common.ros.ros_controller")
    return importlib.reload(module), rospy


def test_scheduler_hardware_has_no_cross_package_imports():
    hardware_root = Path(__file__).resolve().parents[2] / "rlinf/scheduler/hardware"
    violations = []
    for path in hardware_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "rlinf."
            ):
                violations.append(f"{path.name}:{node.lineno}:{node.module}")
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}:{node.lineno}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith("rlinf.")
                )
    assert violations == []


def test_cli_discovers_mounted_binary(monkeypatch, tmp_path):
    executable = tmp_path / "camctr"
    executable.touch()
    monkeypatch.setenv("RLINF_EMBODIED_PATH", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert EmbodiedRuntimeCLI.find_executable("camctr") == str(executable)


def test_cli_discovers_default_runtime_mount(monkeypatch):
    monkeypatch.delenv("RLINF_EMBODIED_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(
        "pathlib.Path.is_file", lambda path: str(path) == "/opt/rlinf/bin/rosctr"
    )

    assert EmbodiedRuntimeCLI.find_executable("rosctr") == "/opt/rlinf/bin/rosctr"


def test_cli_resolves_runtime_inventory(monkeypatch):
    run = MagicMock(
        side_effect=[
            _result(
                {
                    "cameras": [
                        {
                            "cameraId": "video0",
                            "name": "wrist",
                            "serialNumber": "cam-123",
                        }
                    ]
                }
            ),
            _result(
                {
                    "robots": [
                        {
                            "robotId": "franka-0",
                            "params": {"robot_ip": "172.16.0.2"},
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    cli = EmbodiedRuntimeCLI("camctr", executable="/opt/rlinf/bin/camctr")

    assert cli.resolve_camera_id("cam-123") == "video0"
    cli.tool = "rosctr"
    assert cli.resolve_robot_id("172.16.0.2") == "franka-0"


def test_cli_does_not_treat_broken_controller_as_uninstalled(monkeypatch, tmp_path):
    executable = tmp_path / "rosctr"
    executable.touch()
    monkeypatch.setenv("RLINF_EMBODIED_ROS_CLI", str(executable))
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("rosctr", 5)),
    )

    assert EmbodiedRuntimeCLI.is_installed("rosctr")
    assert not EmbodiedRuntimeCLI.is_available("rosctr")


def test_cli_respects_disabled_runtime_manager(monkeypatch, tmp_path):
    executable = tmp_path / "camctr"
    executable.touch()
    monkeypatch.setenv("RLINF_EMBODIED_PATH", str(tmp_path))
    monkeypatch.setenv("RLINF_EMBODIED_RUNTIME_ENABLED", "1")
    monkeypatch.delenv("RLINF_EMBODIED_CAMERA_ENABLED", raising=False)

    assert EmbodiedRuntimeCLI.is_installed("camctr")
    assert not EmbodiedRuntimeCLI.is_enabled("camctr")

    monkeypatch.setenv("RLINF_EMBODIED_CAMERA_ENABLED", "1")
    assert EmbodiedRuntimeCLI.is_enabled("camctr")


def test_enabled_runtime_reports_missing_cli(monkeypatch):
    monkeypatch.setenv("RLINF_EMBODIED_RUNTIME_ENABLED", "1")
    monkeypatch.setenv("RLINF_EMBODIED_ROS_ENABLED", "1")
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert EmbodiedRuntimeCLI.is_enabled("rosctr")
    with pytest.raises(FileNotFoundError, match="rosctr"):
        EmbodiedRuntimeCLI("rosctr")


def test_cli_rejects_ambiguous_robot(monkeypatch):
    monkeypatch.setattr(
        EmbodiedRuntimeCLI,
        "list_robots",
        lambda _: [{"robotId": "left"}, {"robotId": "right"}],
    )
    cli = EmbodiedRuntimeCLI("rosctr", executable="rosctr")

    with pytest.raises(ValueError, match="Could not select"):
        cli.resolve_robot_id()


def test_cli_does_not_fall_back_on_robot_ip_mismatch(monkeypatch):
    monkeypatch.setattr(
        EmbodiedRuntimeCLI,
        "list_robots",
        lambda _: [{"robotId": "franka-0", "params": {"robot_ip": "172.16.0.2"}}],
    )
    cli = EmbodiedRuntimeCLI("rosctr", executable="rosctr")

    with pytest.raises(ValueError, match="172.16.0.3"):
        cli.resolve_robot_id("172.16.0.3")


def test_cli_rejects_ambiguous_camera_name(monkeypatch):
    monkeypatch.setattr(
        EmbodiedRuntimeCLI,
        "list_cameras",
        lambda _: [
            {"cameraId": "video0", "name": "wrist"},
            {"cameraId": "video1", "name": "wrist"},
        ],
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(ValueError, match="ambiguous"):
        cli.resolve_camera_id("wrist")


def test_cli_reports_timeout(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("camctr", 10)),
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(RuntimeError, match="timed out"):
        cli.run("list")


def test_cli_reports_command_failure_without_stderr(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(2, "camctr")),
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(RuntimeError, match="failed: 2"):
        cli.run("list")


def test_cli_reports_invalid_json(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=b"not-json")),
    )
    cli = EmbodiedRuntimeCLI("camctr", executable="camctr")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        cli.run_json("list")


def test_runtime_camera_opens_and_decodes_watch_stream(monkeypatch):
    jpeg = b"\xff\xd8payload\xff\xd9"
    cli = MagicMock()
    cli.executable = "/opt/rlinf/bin/camctr"
    cli.resolve_camera_id.return_value = "video0"
    cli.run_json.side_effect = [
        {"camera": {"state": "closed"}},
        {"encoding": "jpeg"},
    ]
    process = MagicMock()
    process.stdout.fileno.return_value = 10

    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.os.read",
        lambda fd, size: jpeg,
    )
    decoded = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr("cv2.imdecode", lambda data, flags: decoded)
    camera = EmbodiedRuntimeCamera(
        CameraInfo(name="wrist", serial_number="cam-123", fps=15)
    )
    monkeypatch.setattr(camera, "_capture_frames", lambda: None)

    camera.open()
    ok, frame = camera._read_frame()
    camera.close()

    assert ok
    assert frame is decoded
    cli.run_json.assert_called_with(
        "open",
        "video0",
        "--width",
        "640",
        "--height",
        "480",
        "--fps",
        "15",
        "--encoding",
        "jpeg",
    )
    cli.run.assert_called_once_with("close", "video0")


def test_runtime_camera_rejects_incompatible_existing_stream(monkeypatch):
    cli = MagicMock()
    cli.resolve_camera_id.return_value = "video0"
    cli.run_json.side_effect = [
        {"camera": {"state": "closed"}},
        {"encoding": "h264"},
    ]
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    camera = EmbodiedRuntimeCamera(CameraInfo(name="wrist", serial_number="cam-123"))

    with pytest.raises(RuntimeError, match="incompatible encoding"):
        camera.open()
    cli.run.assert_called_once_with("close", "video0")


def test_camera_factory_prefers_enabled_runtime(monkeypatch):
    runtime_camera = object()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.EmbodiedRuntimeCLI.is_enabled",
        lambda _: True,
    )
    constructor = MagicMock(return_value=runtime_camera)
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.EmbodiedRuntimeCamera", constructor
    )
    info = CameraInfo(name="wrist", serial_number="cam-123")

    camera = camera_module.create_camera(info)

    assert camera is runtime_camera
    constructor.assert_called_once_with(info)


def test_runtime_camera_handles_jpeg_marker_bytes_in_scan_data(monkeypatch):
    cli = MagicMock()
    cli.resolve_camera_id.return_value = "video0"
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    camera = EmbodiedRuntimeCamera(CameraInfo(name="wrist", serial_number="cam-123"))
    camera._buffer.extend(
        b"\xff\xd8\xff\xda\x00\x08\x00\x00\x00\x00\x00\x00abc\xff\x00\xd9def\xff\xd9"
    )

    assert camera._find_jpeg_end() == len(camera._buffer)


def test_runtime_camera_closes_existing_runtime_state(monkeypatch):
    cli = MagicMock()
    cli.executable = "/opt/rlinf/bin/camctr"
    cli.resolve_camera_id.return_value = "video0"
    cli.run_json.side_effect = [
        {"camera": {"state": "error"}},
        {"encoding": "jpeg"},
    ]
    process = MagicMock()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    camera = EmbodiedRuntimeCamera(CameraInfo(name="wrist", serial_number="cam-123"))
    monkeypatch.setattr(camera, "_capture_frames", lambda: None)

    camera.open()

    cli.run.assert_called_once_with("close", "video0")
    camera.close()


def test_franka_hardware_is_discovered_from_runtime(monkeypatch):
    monkeypatch.delenv("ROBOT_IP", raising=False)
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )

    resource = FrankaRobot.enumerate(node_rank=0)

    assert resource is not None
    assert resource.count == 1
    config = resource.infos[0].config
    assert config.robot_ip == "172.16.0.2"
    assert config.embodied_runtime_robot_id == "franka-0"
    assert config.camera_serials == ["cam-123"]


def test_franka_hardware_uses_explicit_runtime_robot_id(monkeypatch):
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )
    config = FrankaConfig(
        node_rank=0,
        embodied_runtime_robot_id="franka-0",
        camera_serials=["cam-123"],
    )

    resource = FrankaRobot.enumerate(node_rank=0, configs=[config])

    assert resource is not None
    assert resource.infos[0].config.robot_ip == "172.16.0.2"


def test_franka_hardware_rejects_empty_runtime_inventory(monkeypatch):
    monkeypatch.setattr(_RuntimeCLI, "list_robots", lambda _: [])
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )

    with pytest.raises(ValueError, match="No Franka robots"):
        FrankaRobot.enumerate(node_rank=0)


def test_franka_hardware_is_absent_without_config_or_runtime(monkeypatch):
    monkeypatch.delenv("ROBOT_IP", raising=False)
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI.is_enabled",
        lambda _: False,
    )

    assert FrankaRobot.enumerate(node_rank=0, configs=[]) is None


def test_franka_hardware_requires_runtime_robot_ip(monkeypatch):
    monkeypatch.setattr(
        _RuntimeCLI,
        "list_robots",
        lambda _: [{"robotId": "franka-0", "params": {}}],
    )
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )

    with pytest.raises(ValueError, match="params.robot_ip"):
        FrankaRobot.enumerate(node_rank=0)


def test_ros_controller_connects_through_runtime(monkeypatch):
    module, rospy = _load_ros_controller(monkeypatch)
    runtime = MagicMock()
    runtime.resolve_robot_id.return_value = "franka-0"
    runtime.run_json.side_effect = [
        {"rosMasterUri": "http://10.0.0.2:11311", "state": "running"},
        {"rosMasterUri": "http://10.0.0.2:11311"},
    ]
    runtime_cli = MagicMock(return_value=runtime)
    runtime_cli.is_enabled.return_value = True
    monkeypatch.setattr(module, "EmbodiedRuntimeCLI", runtime_cli)
    popen = MagicMock()
    monkeypatch.setattr(module.psutil, "Popen", popen)

    controller = module.ROSController(robot_ip="172.16.0.2")
    controller.start_runtime_mode(
        "impedance", {"robot_ip": "172.16.0.2", "load_gripper": "true"}
    )

    popen.assert_not_called()
    rospy.init_node.assert_called_once()
    runtime.run_json.assert_called_with(
        "start",
        "franka-0",
        "impedance",
        "--arg",
        "robot_ip=172.16.0.2",
        "--arg",
        "load_gripper=true",
    )


def test_ros_controller_falls_back_without_runtime(monkeypatch):
    module, _ = _load_ros_controller(monkeypatch)
    monkeypatch.setattr(module.EmbodiedRuntimeCLI, "is_enabled", lambda _: False)
    monkeypatch.setattr(module.psutil, "process_iter", lambda: [])
    popen = MagicMock()
    monkeypatch.setattr(module.psutil, "Popen", popen)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    module.ROSController()

    popen.assert_called_once()
    assert popen.call_args.args[0] == ["roscore"]
