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

from rlinf.workers.env.reward_postprocess import compute_reward_assign_lengths


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
