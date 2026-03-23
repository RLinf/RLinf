# Copyright 2026 Ying-Chun Lee
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

"""Portal client compatibility layer for YAM follower servers."""

from __future__ import annotations

from typing import Any

import numpy as np
import portal
from i2rt.robots.robot import RobotType
from yam_realtime.utils.portal_utils import Client as YamRealtimeClient


class YAMFollowerClient(YamRealtimeClient):
    """Client for `minimum_gello.py` follower servers.

    The i2rt follower server exposes a small fixed set of portal methods
    directly and does not implement yam_realtime's
    `get_supported_remote_methods` handshake. This adapter matches the subset of
    the robot protocol that `YAMEnv` / `RobotEnv` actually use.
    """

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._host = host
        self._port = port
        self._client = portal.Client(f"{host}:{port}")
        self._supported_remote_methods = {
            "num_dofs": False,
            "get_joint_pos": False,
            "command_joint_pos": False,
            "command_joint_state": False,
            "get_observations": False,
            "zero_torque_mode": False,
        }
        self._use_future = False

    def get_joint_state(self) -> dict[str, np.ndarray]:
        observations = self.get_observations()
        return {
            "pos": np.asarray(observations["joint_pos"], dtype=np.float32),
            "vel": observations.get(
                "joint_vel",
                np.zeros_like(observations["joint_pos"], dtype=np.float32),
            ),
        }

    def joint_pos_spec(self) -> Any:
        from dm_env.specs import Array

        return Array(shape=(self.num_dofs(),), dtype=np.float32)

    def joint_state_spec(self) -> dict[str, Any]:
        from dm_env.specs import Array

        return {
            "pos": Array(shape=(self.num_dofs(),), dtype=np.float32),
            "vel": Array(shape=(self.num_dofs(),), dtype=np.float32),
        }

    def get_robot_info(self) -> dict[str, Any]:
        return {}

    def get_robot_type(self) -> RobotType:
        return RobotType.ARM
