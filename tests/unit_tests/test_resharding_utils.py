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

"""Tests for pure-function reshard utilities.

These cover the sharding arithmetic used by actor→rollout weight reshard,
without requiring GPU/distributed/mock worker:

- ``reshard_tensor_by_rank``: the slice that ``gather_and_reshard_tensor``
  narrows to after all-gather. Covers item 4b (``rank % inference_tp``
  must stay in range) and the row-parallel contiguity contract.
- ``ep_reshard_fn_deepseek_v3``: expert selection by EP rank. Covers item 9
  (``rollout_ep_size`` mismatch silently sends wrong experts).
"""

import torch

from rlinf.utils.resharding.utils import (
    ep_reshard_fn_deepseek_v3,
    reshard_tensor_by_rank,
)


def test_reshard_tensor_by_rank_slices_expected_shard():
    """dim=0 (column-parallel): rank i gets rows [i*shard : (i+1)*shard)."""
    full = torch.arange(8).reshape(8, 1).float()  # 8 rows, world_size=4 -> 2/rank
    shard1 = reshard_tensor_by_rank(full, dim=0, rank=1, world_size=4)
    assert torch.equal(shard1, full[2:4])
    assert shard1.is_contiguous()


def test_reshard_tensor_by_rank_row_parallel_is_contiguous():
    """dim=1 (row-parallel) narrow() returns a non-contiguous view; the function
    must .contiguous() (P2P send rejects non-contiguous tensors)."""
    full = torch.arange(16).reshape(2, 8).float()
    shard = reshard_tensor_by_rank(full, dim=1, rank=2, world_size=4)
    assert torch.equal(shard, full[:, 4:6])
    assert shard.is_contiguous()


def test_ep_reshard_selects_correct_expert_range():
    """8 global experts, rollout_ep_size=4 -> 2 experts/rank; rank 1 gets [2,3]."""
    expert_params = {
        f"mlp.experts.local_experts.{g}.linear_fc1.weight": torch.tensor([g])
        for g in range(8)
    }
    out = ep_reshard_fn_deepseek_v3(
        expert_params, rollout_ep_size=4, dst_ep_rank=1, num_moe_experts=8
    )
    selected_experts = {int(k.split("local_experts.")[1].split(".")[0]) for k in out}
    assert selected_experts == {2, 3}


def test_ep_reshard_wrong_ep_size_returns_wrong_experts():
    """固化第 9 条的 bug 场景：如果调用方传错 rollout_ep_size（比如 actor 本地
    推导算出 8，而 rollout engine 实际配的 ep_size=4），dst_ep_rank=1（在真实
    ep_size=4 下合法）在错误的 ep_size=8 下会静默拿到一份不同、更小的专家子集
    —— 不报错，只是权重发错了。"""
    expert_params = {
        f"mlp.experts.local_experts.{g}.linear_fc1.weight": torch.tensor([g])
        for g in range(8)
    }
    correct = ep_reshard_fn_deepseek_v3(
        expert_params, rollout_ep_size=4, dst_ep_rank=1, num_moe_experts=8
    )
    wrong = ep_reshard_fn_deepseek_v3(
        expert_params, rollout_ep_size=8, dst_ep_rank=1, num_moe_experts=8
    )
    assert correct != wrong  # 前 2 个专家 {2,3}，后者只有 1 个专家 {1} —— 静默传错
