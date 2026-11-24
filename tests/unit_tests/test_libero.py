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

"""LIBERO observations, episode history, grouped resets, and ART features."""

import importlib
import sys
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
    pytest.importorskip("gym")
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    env = LiberoEnv.__new__(LiberoEnv)
    env.num_envs = 2
    env.art_cluster_states = False
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


@pytest.fixture(scope="module")
def libero_reset_env(tmp_path_factory):
    # Only replace the external simulator and asset package. Physical modules
    # let spawned workers import the same fake; RLinf's vector path stays real.
    root = tmp_path_factory.mktemp("libero_vendor")
    package = root / "libero" / "libero"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def get_libero_path(name):\n"
        "    return str(Path(__file__).parent)\n"
    )
    (package / "benchmark.py").write_text(
        "from types import SimpleNamespace\n"
        "import numpy as np\n"
        "class Benchmark:\n"
        "    def get_task(self, task_id):\n"
        "        return SimpleNamespace(problem_folder='spatial',\n"
        "            bddl_file='task.bddl', language='move the object')\n"
        "    def get_task_init_states(self, task_id):\n"
        "        return np.arange(6).reshape(3, 2)\n"
    )
    (package / "envs.py").write_text(
        """
import gym
import numpy as np

class OffScreenRenderEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def __init__(self, **kwargs):
        self.fixture_pos = np.zeros(3)
        self.dynamic_state = np.zeros(2)

    def seed(self, seed):
        self.reset_seed = seed
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self.fixture_pos = self.rng.uniform(size=3)
        self.dynamic_state = np.zeros(2)
        return self.dynamic_state.copy()

    def set_init_state(self, state):
        # Like LIBERO, restoring dynamic state leaves fixture poses alone.
        self.dynamic_state = np.array(state, copy=True)
        return self.dynamic_state.copy()
"""
    )
    module_names = (
        "libero",
        "rlinf.envs.sim.libero.libero_env",
        "rlinf.envs.sim.libero.utils",
        "rlinf.envs.sim.libero.venv",
    )
    original_modules = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root))
        patch.setenv("LIBERO_TYPE", "standard")
        cls = importlib.import_module("rlinf.envs.sim.libero.libero_env").LiberoEnv
        benchmark = importlib.import_module("libero.libero.benchmark").Benchmark
        env = cls.__new__(cls)
        # Supply benchmark metadata without real assets or rendering. All reset
        # methods, seed expansion and subprocess IPC use the production classes.
        env.cfg = OmegaConf.create(
            {"task_suite_name": "libero_spatial", "init_params": {}}
        )
        env.num_envs = 4
        env.group_size = 2
        env.is_libero_mem = False
        env.short_memory = None
        env.seed = 42
        env.task_ids = np.zeros(env.num_envs, dtype=int)
        env.trial_ids = np.zeros(env.num_envs, dtype=int)
        env.cumsum_trial_id_bins = np.array([3])
        env.task_suite = benchmark()
        env._init_env()
        try:
            yield env
        finally:
            env.env.close()
            for name in set(sys.modules) - original_modules:
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in module_names
                ):
                    del sys.modules[name]


def test_art_grouped_reset_matches_static_fixtures(libero_reset_env):
    env = libero_reset_env
    env.cfg.art_grouped_reset = True
    env.is_eval = False
    env._reconfigure(np.zeros(4, dtype=int), np.arange(4))

    assert env.env.get_env_attr("reset_seed") == [168, 168, 170, 170]
    fixtures = env.env.get_env_attr("fixture_pos")
    np.testing.assert_array_equal(fixtures[0], fixtures[1])
    np.testing.assert_array_equal(fixtures[2], fixtures[3])
    assert not np.array_equal(fixtures[0], fixtures[2])
    for state in env.env.get_env_attr("dynamic_state"):
        np.testing.assert_array_equal(state, [0, 1])

    # Partial/reordered resets retain absolute environment group membership.
    env._reconfigure(np.array([1, 1]), np.array([3, 2]))
    assert env.env.get_env_attr("reset_seed") == [84, 84, 86, 86]
    fixtures = env.env.get_env_attr("fixture_pos")
    np.testing.assert_array_equal(fixtures[2], fixtures[3])
    for state in env.env.get_env_attr("dynamic_state", id=[2, 3]):
        np.testing.assert_array_equal(state, [2, 3])


@pytest.mark.parametrize("grouped,is_eval", [(False, False), (True, True)])
def test_other_libero_modes_keep_independent_reset_seeds(
    libero_reset_env, grouped, is_eval
):
    env = libero_reset_env
    env.cfg.art_grouped_reset = grouped
    env.is_eval = is_eval
    env._reconfigure(np.zeros(4, dtype=int), np.arange(4))

    assert env.env.get_env_attr("reset_seed") == [168, 169, 170, 171]
    fixtures = env.env.get_env_attr("fixture_pos")
    assert not np.array_equal(fixtures[0], fixtures[1])

    env._reconfigure(np.array([1, 1]), np.array([3, 2]))
    assert env.env.get_env_attr("reset_seed") == [84, 85, 86, 87]


def test_art_cluster_features_measure_full_eef_and_object_rotations():
    from rlinf.envs.sim.libero.art_state import ART_ANGLE_UNIT_RAD, cluster_features

    state = {
        "object_position": np.zeros(3),
        "goal_position": np.ones(3) * 0.01,
        "object_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "eef_position": np.ones(3) * 0.02,
        "eef_rotation": np.eye(3),
    }
    baseline = cluster_features(state, np.array([0.01, -0.01]), 20)
    assert baseline.shape == (26,)
    np.testing.assert_allclose(baseline[:6], [2, 2, 2, -1, -1, -1])
    np.testing.assert_allclose(baseline[-2:], [2, 1])
    for angle in np.radians([10, 45, 60, 90, 180]):
        c, s = np.cos(angle), np.sin(angle)
        for axis in range(3):
            rotation = np.eye(3)
            i, j = (axis + 1) % 3, (axis + 2) % 3
            rotation[i, i] = rotation[j, j] = c
            rotation[i, j], rotation[j, i] = -s, s
            quaternion = np.zeros(4)
            quaternion[0], quaternion[axis + 1] = np.cos(angle / 2), np.sin(angle / 2)
            expected = 2 * np.sin(angle / 2) / ART_ANGLE_UNIT_RAD
            for key, value in (
                ("eef_rotation", rotation),
                ("object_quaternion", quaternion),
            ):
                feature = cluster_features(
                    dict(state, **{key: value}), np.array([0.01, -0.01]), 20
                )
                np.testing.assert_allclose(np.linalg.norm(feature - baseline), expected)
                if key == "object_quaternion":
                    np.testing.assert_allclose(
                        feature,
                        cluster_features(
                            dict(state, object_quaternion=-quaternion),
                            np.array([0.01, -0.01]),
                            20,
                        ),
                    )


def test_art_cluster_observations_carry_policy_owned_inputs():
    from rlinf.envs.sim.libero.art_state import ART_PHASES
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    state = {
        "phase": "free",
        "object_position": np.zeros(3),
        "goal_position": np.ones(3) * 0.01,
        "object_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "eef_position": np.ones(3) * 0.02,
        "eef_rotation": np.eye(3),
    }
    env = LiberoEnv.__new__(LiberoEnv)
    env.env = Mock()
    env.env.object_states.return_value = [state, state]
    env.cfg = OmegaConf.create({"max_episode_steps": 240})
    env._elapsed_steps = np.array([0, 25])
    env.task_ids = np.array([6, 7])
    env.current_raw_obs = [
        {"robot0_gripper_qpos": np.array([0.01, -0.01])} for _ in range(2)
    ]
    obs = {"states": torch.zeros(2, 8)}
    env._add_art_cluster_states(obs)
    assert set(obs) == {"states", "policy_inputs"}
    features = obs["policy_inputs"]["art_cluster_features"]
    assert features.shape == (2, 26)
    torch.testing.assert_close(features[:, -1], torch.tensor([12.0, 10.75]))
    assert obs["policy_inputs"]["art_cluster_bucket"].tolist() == [
        6 * len(ART_PHASES),
        7 * len(ART_PHASES),
    ]
