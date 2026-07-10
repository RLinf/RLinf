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

import torch

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.workers.trajectory import (
    Actions,
    EnvBootstrap,
    Observations,
    RolloutBootstrap,
    TrajectoryStorage,
    assign_peer_ranks,
    assign_trajectory_ranks,
)


def test_trajectory_rank_mapping_is_symmetric_for_actor_receives():
    actor_world_size = 2
    trajectory_world_size = 2

    assert assign_peer_ranks(0, actor_world_size, trajectory_world_size) == [0]
    assert assign_peer_ranks(1, actor_world_size, trajectory_world_size) == [1]
    assert assign_trajectory_ranks(0, actor_world_size, trajectory_world_size) == [0]
    assert assign_trajectory_ranks(1, actor_world_size, trajectory_world_size) == [1]


def test_pending_intervention_is_applied_after_actions_arrive():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=False,
    )

    storage.apply_intervention(
        current_epoch=0,
        current_step=0,
        slot_indices=[0, 1],
        intervene_actions=torch.tensor([[[9.0, 9.0]], [[8.0, 8.0]]]),
        intervene_flags=torch.tensor([[True], [False]]),
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
            forward_inputs={
                "action": torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
                "model_action": torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
            },
        ),
        [0, 1],
    )

    assert torch.equal(
        storage.actions[0],
        torch.tensor([[[9.0, 9.0]], [[2.0, 2.0]]]),
    )
    assert torch.equal(
        storage.forward_inputs["action"][0],
        torch.tensor([[[9.0, 9.0]], [[2.0, 2.0]]]),
    )
    assert "model_action" not in storage.forward_inputs
    assert torch.equal(
        storage.intervene_flags[0],
        torch.tensor([[[True, True]], [[False, False]]]),
    )


def test_bootstrap_without_reward_marks_reward_ready():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            rewards=torch.ones(2, 1),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(global_step=0, rank=0, current_epoch=0),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )

    assert storage.complete()


def test_observation_without_reward_marks_reward_ready():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            rewards=None,
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(global_step=0, rank=0, current_epoch=0),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )

    assert storage.complete()


def test_terminal_bootstrap_reward_is_applied_once():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=torch.tensor([[1.0], [2.0]]),
            dones=torch.tensor([[False], [True]]),
            truncations=torch.tensor([[False], [True]]),
        ),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.tensor([[3.0], [4.0]]),
        ),
        [0, 1],
    )

    storage.apply_terminal_bootstrap_reward(
        current_epoch=0,
        gamma=0.5,
        auto_reset=True,
        bootstrap_type="standard",
    )
    storage.apply_terminal_bootstrap_reward(
        current_epoch=0,
        gamma=0.5,
        auto_reset=True,
        bootstrap_type="standard",
    )

    assert torch.equal(storage.rewards[0, :, 0], torch.tensor([1.0, 4.0]))


def test_exported_trajectory_uses_transition_length_for_rewards():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=True,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            rewards=None,
            dones=torch.zeros(2, 1, dtype=torch.bool),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            rewards=torch.full((2, 1), 1.0),
            dones=torch.zeros(2, 1, dtype=torch.bool),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=torch.full((2, 1), 3.0),
            dones=torch.ones(2, 1, dtype=torch.bool),
        ),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )

    trajectory = storage.to_trajectory()

    assert trajectory.actions.shape[:2] == (2, 2)
    assert trajectory.rewards.shape[:2] == (2, 2)
    assert trajectory.dones.shape[:2] == (3, 2)
    assert trajectory.prev_values.shape[:2] == (3, 2)
    assert torch.equal(
        trajectory.rewards[:, :, 0], torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    )
    assert torch.equal(
        trajectory.dones[:, :, 0],
        torch.tensor(
            [[False, False], [False, False], [True, True]],
            dtype=torch.bool,
        ),
    )


def test_storage_matches_embodied_rollout_result_time_alignment():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=True,
    )
    rollout_result = EmbodiedRolloutResult(max_episode_length=4)

    obs0 = {"states": torch.tensor([[0.0], [10.0]])}
    obs1 = {"states": torch.tensor([[1.0], [11.0]])}
    obs2 = {"states": torch.tensor([[2.0], [12.0]])}
    action0 = torch.tensor([[[101.0]], [[102.0]]])
    action1 = torch.tensor([[[201.0]], [[202.0]]])
    logprob0 = torch.tensor([[[0.1]], [[0.2]]])
    logprob1 = torch.tensor([[[0.3]], [[0.4]]])
    value0 = torch.tensor([[1.0], [2.0]])
    value1 = torch.tensor([[3.0], [4.0]])
    value2 = torch.tensor([[5.0], [6.0]])
    version0 = torch.tensor([[[7.0]], [[7.0]]])
    version1 = torch.tensor([[[8.0]], [[8.0]]])
    reward0 = torch.tensor([[0.5], [0.6]])
    reward1 = torch.tensor([[1.5], [1.6]])
    done0 = torch.zeros(2, 1, dtype=torch.bool)
    done1 = torch.tensor([[False], [True]])
    done2 = torch.tensor([[True], [True]])

    rollout_result.append_step_result(
        ChunkStepResult(
            actions=action0,
            prev_logprobs=logprob0,
            prev_values=value0,
            forward_inputs={"action": action0},
            versions=version0,
            dones=done0,
            terminations=done0,
            truncations=done0,
            rewards=None,
        )
    )
    rollout_result.append_transitions(obs0.copy(), obs1.copy())
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=action1,
            prev_logprobs=logprob1,
            prev_values=value1,
            forward_inputs={"action": action1},
            versions=version1,
            dones=done1,
            terminations=done1,
            truncations=torch.zeros_like(done1),
            rewards=reward0,
        )
    )
    rollout_result.append_transitions(obs1.copy(), obs2.copy())
    rollout_result.append_step_result(
        ChunkStepResult(
            prev_values=value2,
            dones=done2,
            terminations=done2,
            truncations=torch.zeros_like(done2),
            rewards=reward1,
        )
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            obs=obs0,
            next_obs=obs1,
            rewards=None,
            dones=done0,
            terminations=done0,
            truncations=done0,
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=action0,
            prev_logprobs=logprob0,
            prev_values=value0,
            versions=version0,
            forward_inputs={"action": action0},
        ),
        [0, 1],
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            obs=obs1,
            next_obs=obs2,
            rewards=reward0,
            dones=done1,
            terminations=done1,
            truncations=torch.zeros_like(done1),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            actions=action1,
            prev_logprobs=logprob1,
            prev_values=value1,
            versions=version1,
            forward_inputs={"action": action1},
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=reward1,
            dones=done2,
            terminations=done2,
            truncations=torch.zeros_like(done2),
        ),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=value2,
        ),
        [0, 1],
    )

    expected = rollout_result.to_trajectory()
    actual = storage.to_trajectory()

    assert storage.complete()
    for field_name in (
        "actions",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "prev_logprobs",
        "prev_values",
        "versions",
    ):
        assert torch.equal(getattr(actual, field_name), getattr(expected, field_name))
    assert torch.equal(actual.curr_obs["states"], expected.curr_obs["states"])
    assert torch.equal(actual.next_obs["states"], expected.next_obs["states"])
    assert torch.equal(
        actual.forward_inputs["action"], expected.forward_inputs["action"]
    )
