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
    compute_reward_assign_lengths,
    normalize_total_reward,
)


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


def test_normalize_total_reward_batch_zscore_has_zero_mean_and_unit_variance():
    rewards = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32)

    normalized = normalize_total_reward(rewards, mode="batch_zscore", eps=1.0e-6)

    assert abs(float(normalized.mean().item())) < 1.0e-6
    assert abs(float(normalized.std(unbiased=False).item()) - 1.0) < 1.0e-6


def test_normalize_total_reward_zero_variance_falls_back_to_zero_centered():
    rewards = torch.full((3, 2), 5.0, dtype=torch.float32)

    normalized = normalize_total_reward(rewards, mode="batch_zscore", eps=1.0e-6)

    assert torch.allclose(normalized, torch.zeros_like(rewards))
