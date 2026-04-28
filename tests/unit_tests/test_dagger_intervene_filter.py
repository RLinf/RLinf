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

import numpy as np
import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.envs.realworld.realworld_env import RealWorldEnv


def test_extract_intervene_traj_all_drops_partial_chunk():
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.actions.append(torch.zeros(1, 420))
    flags = torch.zeros(1, 420, dtype=torch.bool)
    flags[:, :14] = True
    rollout.intervene_flags.append(flags)
    rollout.rewards.append(torch.zeros(1))
    trajectory = rollout.to_trajectory()

    assert trajectory.extract_intervene_traj(mode="all") is None
    assert trajectory.extract_intervene_traj(mode="any") is not None


def test_realworld_explicit_false_intervene_flag_is_respected():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.num_envs = 1

    flag = env._extract_intervene_flag(
        {
            "intervene_flag": np.asarray([False], dtype=bool),
            "intervene_action": [np.ones(14, dtype=np.float32)],
        }
    )

    np.testing.assert_array_equal(flag, np.asarray([False], dtype=bool))
