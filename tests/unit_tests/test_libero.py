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
def test_libero_camera_orientation(extract_image, is_libero_mem):
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    obs = {"agentview_image": image, "robot0_eye_in_hand_image": image}

    actual = extract_image(obs, is_libero_mem=is_libero_mem)

    expected = image[::-1] if is_libero_mem else image[::-1, ::-1]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(extract_image(obs), image[::-1, ::-1])


@pytest.fixture(params=[torch.uint8, torch.float32])
def libero_history_env(request):
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
                "main_images": torch.full((2, 2, 2, 3), step, dtype=request.param),
                "wrist_images": torch.full(
                    (2, 2, 2, 3), step + 10, dtype=request.param
                ),
            },
            range(2),
        )
    return env


@pytest.mark.parametrize("history_length", [1, 3])
def test_libero_history_sampling_after_rollover(libero_history_env, history_length):
    libero_history_env.history_length = history_length
    history = libero_history_env._get_history_obs()
    expected_steps = [6] if history_length == 1 else [2, 4, 6]
    expected_offsets = [0.0] if history_length == 1 else [-0.2, -0.1, 0.0]

    assert history["main_images"][0, :, 0, 0, 0].tolist() == expected_steps
    assert history["wrist_images"][0, :, 0, 0, 0].tolist() == [
        step + 10 for step in expected_steps
    ]
    assert history["frame_mask"].tolist() == [[True] * history_length] * 2
    torch.testing.assert_close(
        history["time_offsets"], torch.tensor([expected_offsets] * 2)
    )
    assert all(value.is_contiguous() for value in history.values())
    assert (
        history["main_images"].dtype
        == libero_history_env._observation_history[0][-1][0].dtype
    )
    history["main_images"].zero_()
    assert libero_history_env._observation_history[0][-1][0][0, 0, 0] == 6


def test_libero_partial_reset_preserves_other_histories(libero_history_env):
    env = libero_history_env
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
    image_dtype = env._observation_history[0][-1][0].numpy().dtype
    raw_obs = {
        "agentview_image": np.full((2, 2, 3), 9, dtype=image_dtype),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), 19, dtype=image_dtype),
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

    assert obs["history"]["frame_mask"][0].tolist() == [False, False, True]
    assert obs["history"]["main_images"][0, :, 0, 0, 0].tolist() == [0, 0, 9]
    assert obs["history"]["wrist_images"][0, :, 0, 0, 0].tolist() == [0, 0, 19]
    torch.testing.assert_close(obs["history"]["time_offsets"][0], torch.zeros(3))
    for key in before:
        torch.testing.assert_close(obs["history"][key][1], before[key][1])

    assert before["main_images"][0, :, 0, 0, 0].tolist() == [2, 4, 6]
    assert before["frame_mask"][0].tolist() == [True, True, True]


@pytest.mark.parametrize(
    "variant,available,packages",
    [
        ("standard", "libero.libero", ["libero.libero"]),
        (
            "standard",
            "liberopro.liberopro",
            ["libero.libero", "liberopro.liberopro"],
        ),
        (
            "standard",
            "liberoplus.liberoplus",
            ["libero.libero", "liberopro.liberopro", "liberoplus.liberoplus"],
        ),
        ("pro", "liberopro.liberopro", ["liberopro.liberopro"]),
        ("pro", "libero.libero", ["liberopro.liberopro", "libero.libero"]),
        ("plus", "liberoplus.liberoplus", ["liberoplus.liberoplus"]),
        ("plus", "libero.libero", ["liberoplus.liberoplus", "libero.libero"]),
    ],
)
def test_libero_import_uses_variant_order(monkeypatch, variant, available, packages):
    from types import ModuleType

    from rlinf.envs.sim.libero import utils

    selected = ModuleType(f"{available}.benchmark")
    attempted = []

    def import_module(name):
        attempted.append(name)
        if name != selected.__name__:
            raise ImportError(name)
        return selected

    monkeypatch.setattr(utils, "libero_type", variant)
    monkeypatch.setattr(utils, "import_module", import_module)

    assert utils.import_libero_module("benchmark") is selected
    assert attempted == [f"{package}.benchmark" for package in packages]


@pytest.mark.parametrize(
    "variant,packages",
    [
        ("standard", ["libero.libero", "liberopro.liberopro", "liberoplus.liberoplus"]),
        ("pro", ["liberopro.liberopro", "libero.libero"]),
        ("plus", ["liberoplus.liberoplus", "libero.libero"]),
    ],
)
def test_libero_import_limits_fallbacks(monkeypatch, variant, packages):
    from rlinf.envs.sim.libero import utils

    attempted = []

    def import_module(name):
        attempted.append(name)
        raise ImportError(name)

    monkeypatch.setattr(utils, "libero_type", variant)
    monkeypatch.setattr(utils, "import_module", import_module)

    with pytest.raises(ImportError, match="Cannot import LIBERO envs"):
        utils.import_libero_module("envs")
    assert attempted == [f"{package}.envs" for package in packages]
