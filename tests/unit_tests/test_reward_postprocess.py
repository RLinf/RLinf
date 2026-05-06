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

from __future__ import annotations

import torch

from rlinf.workers.env.reward_postprocess import (
    apply_history_reward_assignment,
    apply_success_bonus,
    compute_delta_gae_step_rewards,
    compute_gae_targets,
    compute_reward_assign_lengths,
    normalize_total_reward,
)


def test_compute_gae_targets_matches_hand_calculation():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    next_values = torch.tensor([[0.25], [0.0]])
    dones = torch.tensor([[False], [True]])

    targets = compute_gae_targets(
        rewards,
        values,
        next_values,
        dones,
        gamma=0.8,
        gae_lambda=0.9,
    )

    expected = torch.tensor([[2.4600], [2.0000]])
    assert torch.allclose(targets, expected, atol=1e-4)


def test_compute_reward_assign_lengths_uses_min_history_and_current_length():
    history_lengths = {
        "history_window": [5, 2],
        "full_history": [4, 3],
    }

    lengths = compute_reward_assign_lengths(
        history_lengths,
        num_envs=2,
        current_rollout_length=3,
    )

    assert torch.equal(lengths, torch.tensor([3, 2], dtype=torch.long))


def test_delta_gae_step_rewards_respect_threshold_boundaries():
    gae_targets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.2, -0.3],
        ]
    )
    reward_assign_lengths = torch.tensor(
        [
            [1, 1, 1],
            [2, 2, 2],
        ],
        dtype=torch.long,
    )

    step_rewards, delta = compute_delta_gae_step_rewards(
        gae_targets,
        reward_assign_lengths,
        label_mode="threshold",
        delta_threshold=0.2,
        positive_reward=1.0,
        unchanged_reward=0.0,
        negative_reward=-0.5,
    )

    assert torch.allclose(delta[1], torch.tensor([0.25, 0.2, -0.3]))
    assert torch.allclose(step_rewards[1], torch.tensor([1.0, 0.0, -0.5]))


def test_delta_gae_step_rewards_batch_tercile_uses_rollout_quantiles():
    gae_targets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    reward_assign_lengths = torch.tensor(
        [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ],
        dtype=torch.long,
    )

    step_rewards, delta = compute_delta_gae_step_rewards(
        gae_targets,
        reward_assign_lengths,
        label_mode="batch_tercile",
        delta_threshold=0.2,
        positive_reward=1.0,
        unchanged_reward=0.0,
        negative_reward=-0.5,
    )

    assert torch.allclose(
        delta[1:],
        torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.3, 0.3]]),
    )
    assert torch.allclose(
        step_rewards,
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.5, -0.5, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_apply_history_reward_assignment_matches_history_buffer_semantics():
    step_rewards = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [0.0, 1.0],
        ]
    )
    reward_assign_lengths = torch.tensor(
        [
            [1, 1],
            [2, 2],
            [3, 1],
        ],
        dtype=torch.long,
    )

    assigned = apply_history_reward_assignment(step_rewards, reward_assign_lengths)

    expected = torch.tensor(
        [
            [1.0, -0.5],
            [1.0, -0.5],
            [0.0, 1.0],
        ]
    )
    assert torch.allclose(assigned, expected)


def test_apply_success_bonus_adds_bonus_only_on_success_steps():
    rewards = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    success_mask = torch.tensor([[False, True], [True, False]])

    mixed = apply_success_bonus(rewards, success_mask, success_bonus=10.0)

    assert torch.allclose(mixed, torch.tensor([[0.0, 11.0], [12.0, 3.0]]))


def test_normalize_total_reward_batch_zscore_has_zero_mean_and_unit_variance():
    rewards = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32)

    normalized = normalize_total_reward(rewards, mode="batch_zscore", eps=1.0e-6)

    assert abs(float(normalized.mean().item())) < 1.0e-6
    assert abs(float(normalized.std(unbiased=False).item()) - 1.0) < 1.0e-6


def test_normalize_total_reward_zero_variance_falls_back_to_zero_centered():
    rewards = torch.full((3, 2), 5.0, dtype=torch.float32)

    normalized = normalize_total_reward(rewards, mode="batch_zscore", eps=1.0e-6)

    assert torch.allclose(normalized, torch.zeros_like(rewards))
