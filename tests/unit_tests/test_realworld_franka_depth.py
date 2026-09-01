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

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.common.camera import CameraInfo
from rlinf.envs.realworld.franka.franka_env import FrankaEnv


def _build_franka_space(camera_infos):
    env = object.__new__(FrankaEnv)
    env.config = SimpleNamespace(
        camera_observation_size=8,
        camera_resize=True,
        enable_camera_depth=True,
        ee_pose_limit_min=np.full(6, -np.inf),
        ee_pose_limit_max=np.full(6, np.inf),
        end_effector_type="franka_gripper",
    )
    env._camera_infos = camera_infos
    env._init_action_obs_spaces()
    return env


def test_franka_depth_space_only_contains_depth_capable_cameras():
    env = _build_franka_space(
        [
            CameraInfo(
                "realsense", "camera-1", camera_type="realsense", enable_depth=True
            ),
            CameraInfo("lumos", "camera-2", camera_type="lumos"),
        ]
    )

    assert set(env.observation_space["frames"].spaces) == {"realsense", "lumos"}
    assert set(env.observation_space["depths"].spaces) == {"realsense"}


def test_franka_depth_requires_a_depth_capable_camera():
    with pytest.raises(ValueError, match="none of the configured cameras support depth"):
        _build_franka_space(
            [CameraInfo("lumos", "camera-1", camera_type="lumos")]
        )
