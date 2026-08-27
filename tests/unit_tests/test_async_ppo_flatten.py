# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from rlinf.workers.actor.async_ppo_fsdp_worker import (
    _get_rollout_training_shape,
    flatten_rollout_batch_for_train,
)


def test_flatten_trims_bootstrap_time_entry_before_shared_shuffle() -> None:
    batch_size = 4
    batch = {
        "advantages": torch.arange(batch_size, dtype=torch.float32).reshape(1, 4, 1),
        "rewards": torch.zeros(1, batch_size, 1),
        "prev_logprobs": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        "dones": torch.zeros(2, batch_size, 1, dtype=torch.bool),
        "forward_inputs": {
            "action": torch.arange(2 * batch_size * 6, dtype=torch.float32).reshape(
                2, batch_size, 6
            )
        },
    }
    shuffle = torch.tensor([3, 1, 0, 2])

    flattened = flatten_rollout_batch_for_train(batch, shuffle)

    assert flattened["advantages"].shape == (batch_size, 1)
    assert flattened["prev_logprobs"].shape == (batch_size, 1)
    assert flattened["dones"].shape == (batch_size, 1)
    assert flattened["forward_inputs"]["action"].shape == (batch_size, 6)
    torch.testing.assert_close(
        flattened["prev_logprobs"].squeeze(-1),
        torch.tensor([3.0, 1.0, 0.0, 2.0]),
    )


def test_flatten_rejects_inconsistent_time_dimension() -> None:
    batch = {
        "advantages": torch.zeros(1, 2, 1),
        "versions": torch.zeros(3, 2, 1),
    }

    with pytest.raises(ValueError, match="expected 1 or 2"):
        flatten_rollout_batch_for_train(batch, shuffle_id=None)


def test_training_shape_uses_transitions_instead_of_bootstrap_entries() -> None:
    batch = {
        "advantages": torch.zeros(1, 4, 50),
        "prev_logprobs": torch.zeros(2, 4, 50, 6),
        "forward_inputs": {"action": torch.zeros(2, 4, 300)},
    }

    assert _get_rollout_training_shape(batch) == (1, 4)
