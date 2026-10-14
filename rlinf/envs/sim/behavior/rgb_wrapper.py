# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import (
    HEAD_RESOLUTION,
    ROBOT_CAMERA_NAMES,
    WRIST_RESOLUTION,
)


class RGBWrapper(EnvironmentWrapper):
    """A RGB-only wrapper for OmniGibson."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
        env.load_observation_space()
