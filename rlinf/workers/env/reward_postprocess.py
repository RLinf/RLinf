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


def compute_reward_assign_lengths(
    history_lengths: dict[str, list[int]] | None,
    *,
    num_envs: int,
    current_rollout_length: int,
) -> torch.Tensor:
    if history_lengths is None:
        return torch.ones((num_envs,), dtype=torch.long)

    assign_lengths = []
    for env_id in range(num_envs):
        assign_length = min(
            history_buffer_length[env_id]
            for history_buffer_length in history_lengths.values()
        )
        assign_lengths.append(min(assign_length, current_rollout_length))
    return torch.as_tensor(assign_lengths, dtype=torch.long)
