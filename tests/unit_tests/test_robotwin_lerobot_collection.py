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

"""Regression tests for RoboTwin online LeRobot frame alignment."""

import numpy as np
import pytest
import torch

from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import PolicyOutput
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


def _obs(states):
    return {
        "states": torch.as_tensor(states, dtype=torch.float32).reshape(-1, 1),
        "task_descriptions": [f"task-{idx}" for idx in range(len(states))],
    }


def _builder(num_envs, num_action_chunks):
    return EmbodiedLerobotTrajectoryBuilder(
        max_episode_length=100,
        num_envs=num_envs,
        only_success=False,
        num_action_chunks=num_action_chunks,
        action_dim=1,
    )


def test_valid_action_mask_skips_padding_and_keeps_pre_action_observations():
    builder = _builder(num_envs=2, num_action_chunks=4)
    obs_list = [
        _obs([10, 20]),
        _obs([11, 21]),
        _obs([12, 22]),
        _obs([13, 23]),
    ]
    infos_list = [
        {},
        {},
        {},
        {
            "final_observation": _obs([999, 888]),
            "final_info": {
                "episode": {"success_once": torch.tensor([True, False])}
            },
            "_final_info": torch.tensor([True, True]),
            "_final_observation": torch.tensor([True, True]),
        },
    ]

    builder.append_chunk_episode_data(
        policy_output=PolicyOutput(
            intervene_flags=torch.tensor(
                [[False, True, True, True], [False, False, False, False]]
            ),
            forward_inputs={
                "action": torch.tensor(
                    [
                        [[1000], [1001], [1002], [1003]],
                        [[2000], [2001], [2002], [2003]],
                    ],
                    dtype=torch.float32,
                )
            },
        ),
        chunk_actions=np.array(
            [
                [[100], [101], [102], [103]],
                [[200], [201], [202], [203]],
            ],
            dtype=np.float32,
        ),
        obs_list=obs_list,
        terminations=torch.tensor(
            [[False, False, False, True], [False, False, False, False]]
        ),
        truncations=torch.tensor(
            [[False, False, False, False], [False, False, False, True]]
        ),
        infos_list=infos_list,
        valid_action_mask=torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
        observations_are_action_aligned=True,
    )

    # The next chunk starts from reset observations supplied as curr_obs. The
    # action-aligned path must not seed it with terminal 999/888 or old states.
    builder.append_chunk_episode_data(
        policy_output=PolicyOutput(),
        chunk_actions=np.array(
            [
                [[300], [301], [302], [303]],
                [[400], [401], [402], [403]],
            ],
            dtype=np.float32,
        ),
        obs_list=[
            _obs([30, 40]),
            _obs([31, 41]),
            _obs([32, 42]),
            _obs([33, 43]),
        ],
        terminations=torch.zeros((2, 4), dtype=torch.bool),
        truncations=torch.zeros((2, 4), dtype=torch.bool),
        infos_list=[{}, {}, {}, {}],
        valid_action_mask=torch.ones((2, 4), dtype=torch.bool),
        observations_are_action_aligned=True,
    )

    episodes = builder.drain_episodes()
    assert [len(episode) for episode in episodes] == [2, 4]
    assert [frame["state"].item() for frame in episodes[0]] == [10, 11]
    assert [frame["actions"].item() for frame in episodes[0]] == [100, 1001]
    assert [frame["state"].item() for frame in episodes[1]] == [20, 21, 22, 23]
    assert [frame["actions"].item() for frame in episodes[1]] == [
        200,
        201,
        202,
        203,
    ]
    assert episodes[0][-1]["done"].item()
    assert episodes[0][-1]["is_success"].item()
    assert episodes[1][-1]["done"].item()
    assert not episodes[1][-1]["is_success"].item()
    assert builder._env_buffers[0][0]["state"].item() == 30
    assert builder._env_buffers[1][0]["state"].item() == 40
    assert builder._pending_obs == [None, None]


def test_zero_valid_terminal_chunk_flushes_existing_episode_without_padding():
    builder = _builder(num_envs=1, num_action_chunks=2)
    builder.append_chunk_episode_data(
        policy_output=PolicyOutput(),
        chunk_actions=np.array([[[1], [2]]], dtype=np.float32),
        obs_list=[_obs([5]), _obs([6])],
        terminations=torch.zeros((1, 2), dtype=torch.bool),
        truncations=torch.zeros((1, 2), dtype=torch.bool),
        infos_list=[{}, {}],
        valid_action_mask=torch.ones((1, 2), dtype=torch.bool),
        observations_are_action_aligned=True,
    )
    builder.append_chunk_episode_data(
        policy_output=PolicyOutput(),
        chunk_actions=np.array([[[3], [4]]], dtype=np.float32),
        obs_list=[_obs([777]), _obs([777])],
        terminations=torch.tensor([[False, True]]),
        truncations=torch.zeros((1, 2), dtype=torch.bool),
        infos_list=[{}, {"episode": {"success_once": torch.tensor([True])}}],
        valid_action_mask=torch.zeros((1, 2), dtype=torch.bool),
        observations_are_action_aligned=True,
    )

    episodes = builder.drain_episodes()
    assert len(episodes) == 1
    assert [frame["state"].item() for frame in episodes[0]] == [5, 6]
    assert [frame["actions"].item() for frame in episodes[0]] == [1, 2]
    assert episodes[0][-1]["is_success"].item()


def test_valid_action_mask_rejects_non_prefix_values():
    builder = _builder(num_envs=1, num_action_chunks=3)

    with pytest.raises(ValueError, match="contiguous prefix"):
        builder.append_chunk_episode_data(
            policy_output=PolicyOutput(),
            chunk_actions=np.zeros((1, 3, 1), dtype=np.float32),
            obs_list=[_obs([0]), _obs([1]), _obs([2])],
            terminations=torch.zeros((1, 3), dtype=torch.bool),
            truncations=torch.zeros((1, 3), dtype=torch.bool),
            infos_list=[{}, {}, {}],
            valid_action_mask=torch.tensor([[True, False, True]]),
        )


def test_robotwin_builds_per_environment_valid_action_prefixes():
    mask = RoboTwinEnv._valid_action_mask_from_infos(
        [
            {"executed_action_count": 2},
            {"executed_action_count": np.array([4])},
            {},
        ],
        chunk_step=4,
    )

    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]


@pytest.mark.parametrize("executed_count", [-1, 5])
def test_robotwin_rejects_out_of_range_executed_action_count(executed_count):
    with pytest.raises(RuntimeError, match="must be in"):
        RoboTwinEnv._valid_action_mask_from_infos(
            [{"executed_action_count": executed_count}], chunk_step=4
        )
