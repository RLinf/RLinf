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

"""Unit tests for realworld hardware backend compatibility helpers."""

from omegaconf import DictConfig
import pytest

from rlinf.envs.realworld.common import camera as camera_pkg
from rlinf.envs.realworld.common.camera import Camera, CameraInfo, RealSenseCamera
from rlinf.envs.realworld.common.camera import zed_camera as zed_camera_module
from rlinf.envs.realworld.common.gripper import create_gripper
from rlinf.scheduler.cluster.config import ClusterConfig
from rlinf.scheduler.hardware.robots.franka import FrankaConfig


def test_camera_backward_compatibility_alias_and_defaults():
    assert Camera is RealSenseCamera

    info = CameraInfo(name="wrist_1", serial_number="123")
    assert info.camera_type == "realsense"


def test_create_camera_dispatches_to_realsense(monkeypatch):
    class DummyRealSenseCamera:
        def __init__(self, camera_info):
            self.camera_info = camera_info

    monkeypatch.setattr(camera_pkg, "RealSenseCamera", DummyRealSenseCamera)

    info = CameraInfo(name="wrist_1", serial_number="123", camera_type="realsense")
    camera = camera_pkg.create_camera(info)

    assert isinstance(camera, DummyRealSenseCamera)
    assert camera.camera_info is info


def test_create_camera_dispatches_to_zed(monkeypatch):
    class DummyZEDCamera:
        def __init__(self, camera_info):
            self.camera_info = camera_info

    monkeypatch.setattr(zed_camera_module, "ZEDCamera", DummyZEDCamera)

    info = CameraInfo(name="wrist_1", serial_number="123", camera_type="zed")
    camera = camera_pkg.create_camera(info)

    assert isinstance(camera, DummyZEDCamera)
    assert camera.camera_info is info


def test_create_camera_rejects_unknown_backend():
    info = CameraInfo(name="wrist_1", serial_number="123", camera_type="mystery")

    with pytest.raises(ValueError, match="Unsupported camera_type"):
        camera_pkg.create_camera(info)


def test_create_gripper_validates_required_arguments():
    with pytest.raises(ValueError, match="ROSController instance must be provided"):
        create_gripper(gripper_type="franka", ros=None)

    with pytest.raises(ValueError, match="gripper_connection"):
        create_gripper(gripper_type="robotiq", port=None)


def test_cluster_config_parses_new_franka_backend_fields():
    config = DictConfig(
        {
            "num_nodes": 2,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "franka",
                    "node_ranks": "0,1",
                    "hardware": {
                        "type": "Franka",
                        "configs": [
                            {
                                "node_rank": 0,
                                "robot_ip": "10.10.10.1",
                                "camera_serials": ["111"],
                                "camera_type": "zed",
                                "gripper_type": "robotiq",
                                "gripper_connection": "/dev/ttyUSB0",
                                "controller_node_rank": 1,
                                "disable_validate": True,
                            }
                        ],
                    },
                }
            ],
        }
    )

    cluster_cfg = ClusterConfig.from_dict_cfg(config)
    hw_cfgs = cluster_cfg.get_node_hw_configs_by_rank(0)

    assert len(hw_cfgs) == 1
    hw_cfg = hw_cfgs[0]
    assert isinstance(hw_cfg, FrankaConfig)
    assert hw_cfg.camera_type == "zed"
    assert hw_cfg.gripper_type == "robotiq"
    assert hw_cfg.gripper_connection == "/dev/ttyUSB0"
    assert hw_cfg.controller_node_rank == 1
