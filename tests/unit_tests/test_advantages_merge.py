# Copyright 2025 The RLinf Authors.
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

"""Tests for advantage functions and ``RolloutResult.merge_batches``.

These cover the ``returns is None`` guard in
``megatron_worker.compute_advantages_and_returns``: GRPO (and other critic-less
adv_types) return ``None`` for ``returns``, and ``merge_batches`` has no branch
for ``None``-typed values, so the guard prevents a guaranteed
``ValueError: Unsupported batch key type`` when ``batch_size % global_batch_size != 0``
triggers batch merging.
"""

import pytest
import torch

from rlinf.algorithms.advantages import compute_grpo_advantages
from rlinf.data.schema.reasoning_results import RolloutResult


def test_grpo_advantages_returns_none():
    """GRPO has no value baseline, so ``returns`` is intentionally None."""
    rewards = torch.rand(4)
    loss_mask = torch.ones(4, 4)
    _, returns = compute_grpo_advantages(rewards, loss_mask, group_size=2)
    assert returns is None


def test_merge_batches_rejects_none_value():
    """Characterizes the bug the ``if returns is not None`` guard avoids:
    ``merge_batches`` has no branch for None-typed values."""
    batches = [
        {"response_lengths": torch.tensor([1, 2]), "returns": None},
        {"response_lengths": torch.tensor([3, 4]), "returns": None},
    ]
    with pytest.raises(ValueError, match="Unsupported batch key type"):
        RolloutResult.merge_batches(batches)


def test_merge_batches_grpo_batch_without_returns_key():
    """Mirrors what ``compute_advantages_and_returns`` produces for grpo:
    no ``returns`` key at all, so merge succeeds."""
    batches = [
        {
            "response_lengths": torch.tensor([1, 2]),
            "advantages": torch.rand(2, 5),
        },
        {
            "response_lengths": torch.tensor([3, 4]),
            "advantages": torch.rand(2, 5),
        },
    ]
    merged = RolloutResult.merge_batches(batches)
    assert "returns" not in merged
    assert merged["response_lengths"].shape[0] == 4
