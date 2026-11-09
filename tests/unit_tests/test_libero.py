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

"""LIBERO observation preprocessing and episode-local history contracts."""

from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image


@pytest.mark.parametrize("extract_image", [get_libero_image, get_libero_wrist_image])
@pytest.mark.parametrize("is_libero_mem", [False, True])
def test_camera_orientation_matches_demonstrations(extract_image, is_libero_mem):
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    obs = {"agentview_image": image, "robot0_eye_in_hand_image": image}

    actual = extract_image(obs, is_libero_mem=is_libero_mem)

    expected = image[::-1] if is_libero_mem else image[::-1, ::-1]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(extract_image(obs), image[::-1, ::-1])


@pytest.fixture
def history_env():
    pytest.importorskip("libero.libero.envs")
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    env = LiberoEnv.__new__(LiberoEnv)
    env.num_envs = 2
    env.history_length = 3
    env.history_stride = 2
    env.history_fps = 20.0
    env._elapsed_steps = np.zeros(2, dtype=np.int32)
    env._observation_history = [deque(maxlen=5), deque(maxlen=5)]
    for step in range(7):
        env._elapsed_steps[:] = step
        env._append_history(
            {
                "main_images": torch.full((2, 2, 2, 3), step),
                "wrist_images": torch.full((2, 2, 2, 3), step + 10),
            },
            range(2),
        )
    return env


def test_history_samples_oldest_to_current_after_buffer_rollover(history_env):
    history = history_env._get_history_obs()

    assert history["history_main_images"][0, :, 0, 0, 0].tolist() == [2, 4, 6]
    assert history["history_wrist_images"][0, :, 0, 0, 0].tolist() == [12, 14, 16]
    assert history["history_frame_mask"].tolist() == [[True, True, True]] * 2
    torch.testing.assert_close(
        history["history_time_offsets"], torch.tensor([[-0.2, -0.1, 0.0]] * 2)
    )


def test_partial_reset_clears_only_the_reset_environment_history(history_env):
    env = history_env
    env._is_start = False
    env.is_eval = True
    env.is_libero_mem = True
    env.short_memory = {"num_frames": 3, "stride": 2, "fps": 20}
    env.cfg = OmegaConf.create({"num_steps_wait": 1, "reset_gripper_open": True})
    env.seed = 0
    env.cumsum_trial_id_bins = [2]
    env.task_ids = np.array([0, 0])
    env.trial_ids = np.array([0, 1])
    env.task_descriptions = ["task", "task"]
    env.prev_step_reward = np.zeros(2)
    env._init_metrics()
    raw_obs = {
        "agentview_image": np.full((2, 2, 3), 9, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), 19, dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    env.current_raw_obs = [raw_obs.copy(), raw_obs.copy()]
    env.task_suite = Mock()
    env.task_suite.get_task_init_states.return_value = [np.zeros(1), np.zeros(1)]
    env.env = Mock()
    env.env.step.return_value = ([raw_obs], [0.0], [False], [{}])

    before = env._get_history_obs()
    obs, _ = env.reset(env_idx=[0], reset_state_ids=[0])

    assert obs["history_frame_mask"][0].tolist() == [False, False, True]
    assert obs["history_main_images"][0, :, 0, 0, 0].tolist() == [0, 0, 9]
    assert obs["history_wrist_images"][0, :, 0, 0, 0].tolist() == [0, 0, 19]
    torch.testing.assert_close(obs["history_time_offsets"][0], torch.zeros(3))
    for key in before:
        torch.testing.assert_close(obs[key][1], before[key][1])
