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

"""Configuration-only YAM resource checks; no devices are opened."""

import copy

import pytest

from rlinf.robotics.discovery import RobotDiscovery
from rlinf.robotics.robots.dual_yam import (
    DualYamConfig,
    DualYamDiscovery,
    DualYamRobot,
    YamCameraConfig,
    YamDeviceConfig,
)
from rlinf.scheduler.hardware import NodeHardwareConfig


def _station(prefix, *, node_rank=0):
    def device(side):
        return {
            "channel": f"{prefix}_{side}",
            "gripper_type": "yam_gripper",
            "ee_mass": None,
            "gripper_limits": None,
        }

    return {
        "node_rank": node_rank,
        "left_follower": device("left"),
        "right_follower": device("right"),
        "cameras": [
            {"name": name, "serial": f"{prefix}_{name}"}
            for name in ("top_rgb", "left_rgb", "right_rgb")
        ],
    }


def test_two_follower_three_view_station_registers_without_importing_sdk():
    node = NodeHardwareConfig(type="DualYam", configs=[_station("station")])
    station = node.configs[0]

    assert RobotDiscovery.registry["DualYam"].config_cls is DualYamConfig
    assert RobotDiscovery.registry["DualYam"].robot_cls is DualYamRobot
    assert RobotDiscovery.registry["DualYam"].discovery_cls is DualYamDiscovery
    assert list(station.devices) == ["left_follower", "right_follower"]
    assert all(
        isinstance(device, YamDeviceConfig) for device in station.devices.values()
    )
    assert all(isinstance(camera, YamCameraConfig) for camera in station.cameras)
    assert [camera.name for camera in station.cameras] == [
        "top_rgb",
        "left_rgb",
        "right_rgb",
    ]


def test_station_requires_both_followers_and_unique_resources():
    values = _station("missing")
    del values["right_follower"]
    with pytest.raises(ValueError, match="right_follower.*must name a CAN device"):
        DualYamConfig(**values)

    values = _station("missing")
    values["cameras"] = []
    with pytest.raises(ValueError, match="at least one camera"):
        DualYamConfig(**values)

    values = _station("duplicate")
    values["right_follower"]["channel"] = values["left_follower"]["channel"]
    with pytest.raises(ValueError, match="CAN channels must be unique"):
        DualYamConfig(**values)


def test_discovery_assigns_only_local_stations_and_rejects_shared_camera():
    local = DualYamConfig(**_station("local", node_rank=0))
    remote = DualYamConfig(**_station("remote", node_rank=1))
    resource = DualYamDiscovery.enumerate(0, [local, remote])
    assert resource.count == 1
    assert resource.infos[0].config is local
    assert DualYamDiscovery.enumerate(2, [local, remote]) is None

    second_values = copy.deepcopy(_station("second"))
    second_values["cameras"][0]["serial"] = local.cameras[0].serial
    second = DualYamConfig(**second_values)
    with pytest.raises(ValueError, match="camera serial.*shared"):
        DualYamDiscovery.enumerate(0, [local, second])
