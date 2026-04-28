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

import sys
import types

import gymnasium as gym
import numpy as np
import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from rlinf.envs.realworld.common.wrappers import (  # noqa: E402
    apply_dual_pose_action_wrappers,
)
from rlinf.envs.realworld.franka.utils import construct_adjoint_matrix  # noqa: E402


class DummyDualPoseEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(14,), dtype=np.float32
                        )
                    }
                )
            }
        )
        self.absolute_action_space = gym.spaces.Box(
            low=np.full((14,), -2.0, dtype=np.float32),
            high=np.full((14,), 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.last_mode = None
        self.last_action = None

    @property
    def config(self):
        class Config:
            is_dummy = True

        return Config()

    def get_absolute_pose_action_space(self):
        return self.absolute_action_space

    def get_relative_pose_action_space(self):
        return self.action_space

    def step_absolute_pose(self, action):
        self.last_mode = "absolute_pose"
        self.last_action = np.asarray(action)
        return self._obs(), 0.0, False, False, {}

    def step_relative_pose(self, action):
        self.last_mode = "relative_pose"
        self.last_action = np.asarray(action)
        return self._obs(), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def _obs(self):
        # Two identity quaternions, laid out as [xyz, quat] per arm.
        return {
            "state": {
                "tcp_pose": np.array(
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0] * 2,
                    dtype=np.float32,
                )
            }
        }


def test_dual_pose_builder_absolute_mode_routes_and_converts_euler_obs():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(env, {"action_mode": "absolute_pose"})

    action = np.full((14,), 1.5, dtype=np.float32)
    obs, reward, terminated, truncated, info = wrapped.step(action)

    assert wrapped.action_space.low[0] == -2.0
    assert env.last_mode == "absolute_pose"
    np.testing.assert_array_equal(env.last_action, action)
    assert obs["state"]["tcp_pose"].shape == (12,)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info == {}


def test_dual_pose_builder_relative_mode_applies_relative_frame_by_default():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(env, {"action_mode": "relative_pose"})

    # reset() initialises the adjoint matrices inside DualRelativeFrame.
    reset_obs, _ = wrapped.reset()
    np.testing.assert_allclose(
        reset_obs["state"]["tcp_pose"], np.zeros((12,), dtype=np.float32), atol=1e-6
    )

    action = np.full((14,), 0.5, dtype=np.float32)
    obs, *_ = wrapped.step(action)

    assert wrapped.action_space.low[0] == -1.0
    assert env.last_mode == "relative_pose"
    expected_action = action.copy()
    reset_pose = env._obs()["state"]["tcp_pose"]
    expected_action[:6] = construct_adjoint_matrix(reset_pose[:7]) @ action[:6]
    expected_action[7:13] = construct_adjoint_matrix(reset_pose[7:]) @ action[7:13]
    np.testing.assert_allclose(env.last_action, expected_action)
    assert not np.allclose(env.last_action, action)
    assert obs["state"]["tcp_pose"].shape == (12,)


def test_dual_pose_builder_defaults_to_relative_mode():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(env, {"use_relative_frame": False})

    action = np.full((14,), 0.5, dtype=np.float32)
    obs, *_ = wrapped.step(action)

    assert wrapped.action_space.low[0] == -1.0
    assert env.last_mode == "relative_pose"
    np.testing.assert_array_equal(env.last_action, action)
    assert obs["state"]["tcp_pose"].shape == (12,)


def test_dual_pose_builder_relative_mode_no_relative_frame():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(
        env, {"action_mode": "relative_pose", "use_relative_frame": False}
    )

    action = np.full((14,), 0.5, dtype=np.float32)
    obs, *_ = wrapped.step(action)

    assert env.last_mode == "relative_pose"
    np.testing.assert_array_equal(env.last_action, action)
    assert obs["state"]["tcp_pose"].shape == (12,)


def test_dual_pose_builder_rejects_unknown_action_mode():
    with pytest.raises(ValueError, match="Unsupported action_mode"):
        apply_dual_pose_action_wrappers(DummyDualPoseEnv(), {"action_mode": "joint"})


def test_dual_pose_builder_rejects_relative_master_takeover():
    with pytest.raises(ValueError, match="requires action_mode='absolute_pose'"):
        apply_dual_pose_action_wrappers(
            DummyDualPoseEnv(),
            {"action_mode": "relative_pose", "use_master_takeover": True},
        )
