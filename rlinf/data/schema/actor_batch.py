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

"""The multi-turn actor batch shared by Megatron and FSDP."""

from typing import TypeAlias, TypedDict

import torch

TokenRows: TypeAlias = list[torch.Tensor]


class DynamicActorBatch(TypedDict, total=False):
    """Aligned sequence rows with token columns indexed by absolute position.

    Token rows have length prompt_lengths[i] + response_lengths[i]. Sequence
    columns and extra:* metadata contain one entry per row. Storage and exchange
    use CPU tensors; only the current loss microbatch is padded to a dense width.
    """

    input_ids: TokenRows
    attention_mask: TokenRows
    position_ids: TokenRows
    response_mask: TokenRows
    rollout_logprobs: TokenRows
    recomputed_logprobs: TokenRows
    ref_logprobs: TokenRows
    advantages: TokenRows
    loss_scales: TokenRows
    prompt_lengths: torch.Tensor
    response_lengths: torch.Tensor
    rewards: torch.Tensor
    is_end: torch.Tensor
    idx_to_traj: list[int]


def pad_actor_batch_rows(
    batch: DynamicActorBatch,
    divisor: int,
    dp_group: torch.distributed.ProcessGroup,
    device: torch.device | str,
) -> DynamicActorBatch:
    """Align DP row counts to a multiple of the minibatch/microbatch schedule.

    Every column shares the divisible row count before splitting. Added rows
    have one attention token and no loss tokens. Input columns are not changed.
    """
    batch = {
        key: value.copy() if isinstance(value, list) else value
        for key, value in batch.items()
    }
    count = torch.tensor(len(batch["input_ids"]), device=device)
    torch.distributed.all_reduce(
        count, op=torch.distributed.ReduceOp.MAX, group=dp_group
    )
    target = (count.item() + divisor - 1) // divisor * divisor
    missing = target - len(batch["input_ids"])
    for key, value in batch.items():
        if key == "idx_to_traj":
            value.extend([0] * missing)
        elif isinstance(value, list):
            for _ in range(missing):
                row = value[0].new_zeros((1, *value[0].shape[1:]))
                if key == "attention_mask":
                    row.fill_(True)
                value.append(row)
        else:
            batch[key] = torch.cat(
                (value, value.new_zeros((missing, *value.shape[1:])))
            )
    return batch
