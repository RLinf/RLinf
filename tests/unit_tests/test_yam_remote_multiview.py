# Copyright 2026 Shirui Chen
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

"""Tests for YAM remote multi-view observation handling."""

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.remote.remote_env import _proto_to_obs
from rlinf.envs.remote.robot_server import RobotEnvServicer, _obs_to_proto
from rlinf.envs.yam.yam_env import YAMEnv


def test_yam_env_dummy_exposes_all_available_camera_roles():
    cfg = OmegaConf.create(
        {
            "is_dummy": True,
            "img_height": 32,
            "img_width": 32,
            "main_camera": "front",
            "camera_cfgs": {
                "front_camera": {},
                "left_wrist_camera": {},
                "right_wrist_camera": {},
                "rear_camera": {},
            },
            "task_description": "pick and place",
        }
    )

    env = YAMEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    assert env._main_camera_name == "front_camera"
    assert env._wrist_camera_names == ["left_wrist_camera", "right_wrist_camera"]
    assert env._extra_view_camera_names == ["rear_camera"]

    obs, _ = env.reset()

    assert tuple(obs["main_images"].shape) == (1, 32, 32, 3)
    assert tuple(obs["wrist_images"].shape) == (1, 2, 32, 32, 3)
    assert tuple(obs["extra_view_images"].shape) == (1, 1, 32, 32, 3)
    assert obs["task_descriptions"] == ["pick and place"]


def test_multiview_observation_round_trip_preserves_optional_images():
    obs = {
        "states": np.arange(14, dtype=np.float32)[np.newaxis, :],
        "main_images": np.full((1, 8, 8, 3), fill_value=11, dtype=np.uint8),
        "wrist_images": np.stack(
            [
                np.full((8, 8, 3), fill_value=22, dtype=np.uint8),
                np.full((8, 8, 3), fill_value=33, dtype=np.uint8),
            ],
            axis=0,
        )[np.newaxis, :],
        "extra_view_images": np.full((1, 1, 8, 8, 3), fill_value=44, dtype=np.uint8),
        "task_descriptions": ["pick and place"],
    }

    proto = _obs_to_proto(obs, img_h=8, img_w=8, compress=False)
    restored_obs = _proto_to_obs(proto)

    np.testing.assert_array_equal(restored_obs["states"].numpy(), obs["states"])
    np.testing.assert_array_equal(
        restored_obs["main_images"].numpy(), obs["main_images"]
    )
    np.testing.assert_array_equal(
        restored_obs["wrist_images"].numpy(), obs["wrist_images"]
    )
    np.testing.assert_array_equal(
        restored_obs["extra_view_images"].numpy(), obs["extra_view_images"]
    )
    assert restored_obs["task_descriptions"] == ["pick and place"]


def test_robot_env_servicer_reports_multiview_counts():
    class DummyEnv:
        observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
                ),
                "main_images": gym.spaces.Box(
                    low=0, high=255, shape=(32, 32, 3), dtype=np.uint8
                ),
                "wrist_images": gym.spaces.Box(
                    low=0, high=255, shape=(2, 32, 32, 3), dtype=np.uint8
                ),
                "extra_view_images": gym.spaces.Box(
                    low=0, high=255, shape=(1, 32, 32, 3), dtype=np.uint8
                ),
            }
        )
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        _max_episode_steps = 100
        _control_rate_hz = 10.0
        auto_reset = False
        ignore_terminations = True

    response = RobotEnvServicer(DummyEnv()).GetSpaces(None, None)

    assert response.state_dim == 14
    assert response.action_dim == 14
    assert response.img_height == 32
    assert response.img_width == 32
    assert response.img_channels == 3
    assert response.num_wrist_images == 2
    assert response.num_extra_view_images == 1
