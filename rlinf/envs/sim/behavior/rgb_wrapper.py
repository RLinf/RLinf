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

import os

import omnigibson as og
from omegaconf import OmegaConf
from omnigibson.envs import Environment, EnvironmentWrapper


class RGBWrapper(EnvironmentWrapper):
    """A RGB-only wrapper for OmniGibson."""

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Resolution is configured before sensor creation; set the head camera's field of view in each scene.
        r1pro = OmegaConf.load(os.path.join(og.root_path, "eval", "r1pro.yaml"))
        head_sensor = r1pro.eval.camera_sensor_names.head
        for scene in env.scenes:
            scene.robots[0].sensors[head_sensor].horizontal_aperture = 40.0
