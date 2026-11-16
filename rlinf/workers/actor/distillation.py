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

"""Actor-side token alignment for distillation forwards."""

import torch

from rlinf.algorithms.distillation.config import DISTRIBUTION_MODES
from rlinf.algorithms.distillation.fsdp import compute_distribution_loss
from rlinf.algorithms.distillation.losses import DistillationForwardOutput


def has_empty_distillation_rank(batch: dict, device: str) -> bool:
    """Coordinate empty batches across all Actor ranks before model collectives."""
    empty = torch.tensor(batch["input_ids"].numel() == 0, device=device)
    torch.distributed.all_reduce(empty, op=torch.distributed.ReduceOp.MAX)
    return bool(empty.item())


def fsdp_distillation_forward(
    actor,
    logits: torch.Tensor,
    batch: dict,
    *,
    packed: bool,
    starts,
    ends,
    response_only: bool,
):
    """Consume logits using the same sequence spans as the model's packing.

    Targets are at token positions; each sequence's first token has no logits.
    Only one token chunk's Teacher payload is copied to the accelerator at once.
    """
    plan = actor.distillation_plan
    config = actor.distillation_loss_config
    full_length = batch["input_ids"].shape[1]
    width = actor.response_len if response_only else full_length
    offset = full_length - width
    shape = (len(starts), width)
    vocab = logits.shape[-1]
    if plan.canonical_loss_mode in DISTRIBUTION_MODES and config.topk > vocab:
        raise ValueError(f"Expected topk <= vocabulary size {vocab}, got {config.topk}")
    fields = {
        name: logits.new_zeros(shape, dtype=torch.float32)
        for name in (
            "logprobs",
            "entropy",
            "distribution_losses",
            "student_mass",
            "teacher_mass",
            "tail_loss",
        )
    }
    cursor = 0
    for row, (start, end) in enumerate(zip(starts, ends)):
        start, end = int(start), int(end)
        target_start = max(start + 1, offset)
        for left in range(target_start, end, 256):
            right = min(left + 256, end)
            prediction_left = cursor + left - start - 1 if packed else left - 1
            z = logits[
                0 if packed else row,
                prediction_left : prediction_left + right - left,
                :,
            ]
            destination = slice(left - offset, right - offset)
            mask = batch["response_mask"][row, left:right].bool().to(z.device)
            mask = mask & batch["teacher_valid_mask"][row, destination].to(z.device)
            if plan.need_current_logprobs:
                target_ids = batch["input_ids"][row, left:right].to(z.device)
                fields["logprobs"][row, destination] = actor.compute_logprobs(
                    z, target_ids
                )
            if plan.need_entropy:
                logq = z.float().log_softmax(-1)
                fields["entropy"][row, destination] = -(logq.exp() * logq).sum(-1)
            if plan.canonical_loss_mode in DISTRIBUTION_MODES:
                ids = batch.get("teacher_token_ids")
                ids = None if ids is None else ids[row, destination].to(z.device)
                output = compute_distribution_loss(
                    z,
                    ids,
                    batch["teacher_logprobs"][row, destination].to(z.device),
                    config,
                    valid_mask=mask,
                )
                for name in (
                    "distribution_losses",
                    "student_mass",
                    "teacher_mass",
                    "tail_loss",
                ):
                    fields[name][row, destination] = getattr(output, name)
        cursor += end - start
    # Preserve a backward path even when every token in this microbatch is masked.
    zero = logits[..., :1].float().sum() * 0
    fields = {key: value + zero for key, value in fields.items()}
    if not plan.need_current_logprobs:
        fields["logprobs"] = None
    if not plan.need_entropy:
        fields["entropy"] = None
    return DistillationForwardOutput(**fields)
