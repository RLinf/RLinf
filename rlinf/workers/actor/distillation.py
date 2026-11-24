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

from collections.abc import Sequence
from contextvars import ContextVar
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from rlinf.algorithms.distillation.config import DISTRIBUTION_MODES
from rlinf.algorithms.distillation.fsdp import compute_distribution_loss
from rlinf.algorithms.distillation.losses import DistillationForwardOutput

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor


def has_empty_distillation_rank(batch: dict, device: str) -> bool:
    """Coordinate empty batches across all Actor ranks before model collectives."""
    empty = torch.tensor(batch["input_ids"].numel() == 0, device=device)
    torch.distributed.all_reduce(empty, op=torch.distributed.ReduceOp.MAX)
    return bool(empty.item())


# A head is entered once per microbatch, including on independently sharded ranks.
# Only the linear algebra inside that FSDP forward is checkpointed.
_active_head = ContextVar("distillation_head", default=None)


class DistillationHead(nn.Linear):
    """Linear LM head with a loss-only forward inside the usual model call."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        active = _active_head.get()
        if active is not None and active[0] is self:
            return active[1](hidden, self.weight, self.bias)
        return super().forward(hidden)


def install_distillation_head(model: nn.Module) -> DistillationHead | None:
    """Preserve the original parameters of models that return linear logits directly."""
    if not type(model).__module__.startswith(
        "transformers.models."
    ) or model.config.model_type not in {
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }:
        return None
    head = model.get_output_embeddings()
    if (
        type(head) is not nn.Linear
        or "forward" in vars(head)
        or head._forward_hooks
        or head._forward_pre_hooks
    ):
        return None
    replacement = DistillationHead(
        head.in_features, head.out_features, bias=head.bias is not None, device="meta"
    )
    replacement.weight = head.weight
    replacement.bias = head.bias
    replacement.train(head.training)
    model.set_output_embeddings(replacement)
    return replacement


def _statistics(
    actor,
    values,
    weight,
    bias,
    *,
    batch,
    packed,
    starts,
    ends,
    response_only,
    temperature,
    chunk_size,
):
    """Map each target to its preceding hidden state in the same packed sequence."""
    plan = actor.distillation_plan
    config = actor.distillation_loss_config
    distribution = plan.canonical_loss_mode in DISTRIBUTION_MODES
    vocab = weight.size(0) if weight is not None else values.size(-1)
    if distribution and config.topk > vocab:
        raise ValueError(f"Expected topk <= vocabulary size {vocab}, got {config.topk}")
    full_length = batch["input_ids"].size(1)
    width = actor.response_len if response_only else full_length
    offset = full_length - width
    device = values.device
    starts = torch.as_tensor(starts, device=device)
    ends = torch.as_tensor(ends, device=device)
    pos = torch.arange(offset, full_length, device=device)
    valid = (
        batch["response_mask"][:, offset:].to(device).bool()
        & (pos > starts[:, None])
        & (pos < ends[:, None])
    )
    teacher_mask = batch["teacher_valid_mask"].to(device)
    # Task loss and regularizers can use response tokens without a Teacher target.
    if not (
        plan.need_task_rewards or plan.need_entropy or getattr(actor, "kl_beta", 0)
    ):
        valid &= teacher_mask
    rows, cols = valid.nonzero(as_tuple=True)
    prediction = cols + offset - 1
    if packed:
        lengths = ends - starts
        prediction += (lengths.cumsum(0) - lengths - starts)[rows]
    source_rows = torch.zeros_like(rows) if packed else rows
    result = values.new_zeros((*valid.shape, 6), dtype=torch.float32)
    if not rows.numel():
        # An empty rank must still participate in head and backbone gradient reduction.
        empty = values[source_rows, prediction]
        if weight is not None:
            empty = F.linear(empty, weight, bias)
        return result + empty.sum().float()

    def reduce_chunk(inputs, head_weight, head_bias, row, col):
        z = inputs
        if head_weight is not None:
            z = F.linear(inputs, head_weight, head_bias)
            if temperature != 1.0:
                z = z / temperature
        zero = z.new_zeros((z.size(0),), dtype=torch.float32)
        logprobs = entropy = loss = student_mass = teacher_mass = tail = zero
        if plan.need_current_logprobs:
            logprobs = actor.compute_logprobs(z, batch["input_ids"][row, col + offset])
        if plan.need_entropy:
            logq = z.float().log_softmax(-1)
            entropy = -(logq.exp() * logq).sum(-1)
        if distribution:
            ids = batch.get("teacher_token_ids")
            teacher = batch["teacher_logprobs"]
            output = compute_distribution_loss(
                z,
                None
                if ids is None
                else ids[row.to(ids.device), col.to(ids.device)].to(device),
                teacher[row.to(teacher.device), col.to(teacher.device)].to(device),
                config,
                valid_mask=teacher_mask[row, col],
            )
            loss, student_mass, teacher_mass, tail = (
                output.distribution_losses,
                output.student_mass,
                output.teacher_mass,
                output.tail_loss,
            )
        return torch.stack(
            (logprobs, entropy, loss, student_mass, teacher_mass, tail), dim=-1
        ).float()

    # Gather hidden rows once so their gradients scatter back into the backbone once.
    hidden = values[source_rows, prediction] if weight is not None else None
    chunks = []
    for start in range(0, rows.numel(), chunk_size):
        sl = slice(start, start + chunk_size)
        selected = (
            hidden[sl]
            if hidden is not None
            else values[source_rows[sl], prediction[sl]]
        )
        args = (selected, weight, bias, rows[sl], cols[sl])
        if weight is not None and torch.is_grad_enabled():
            chunks.append(
                checkpoint(
                    reduce_chunk, *args, use_reentrant=False, preserve_rng_state=False
                )
            )
        else:
            chunks.append(reduce_chunk(*args))
    return result.index_put((rows, cols), torch.cat(chunks))


def _as_output(actor, stats):
    logprobs, entropy, losses, student_mass, teacher_mass, tail = stats.unbind(-1)
    return DistillationForwardOutput(
        logprobs=logprobs if actor.distillation_plan.need_current_logprobs else None,
        entropy=entropy if actor.distillation_plan.need_entropy else None,
        distribution_losses=losses,
        student_mass=student_mass,
        teacher_mass=teacher_mass,
        tail_loss=tail,
    )


def fsdp_distillation_forward(
    actor: "FSDPActor",
    logits: torch.Tensor,
    batch: dict,
    *,
    packed: bool,
    starts: Sequence[int] | torch.Tensor,
    ends: Sequence[int] | torch.Tensor,
    response_only: bool,
) -> DistillationForwardOutput:
    """Reduce model-provided logits for architectures with custom output transformations."""
    stats = _statistics(
        actor,
        logits,
        None,
        None,
        batch=batch,
        packed=packed,
        starts=starts,
        ends=ends,
        response_only=response_only,
        temperature=1.0,
        chunk_size=256,
    )
    return _as_output(actor, stats)


def distillation_model_forward(
    actor: "FSDPActor",
    model_inputs: dict,
    batch: dict,
    *,
    packed: bool,
    starts: Sequence[int] | torch.Tensor,
    ends: Sequence[int] | torch.Tensor,
    response_only: bool,
) -> DistillationForwardOutput:
    """Project only trainable predictions and return aligned per-token statistics."""
    spans = {
        "packed": packed,
        "starts": starts,
        "ends": ends,
        "response_only": response_only,
    }
    temperature = actor.cfg.algorithm.sampling_params.temperature
    head = actor.distillation_head
    if head is None:
        with actor.amp_context:
            logits = actor.model(**model_inputs).logits
        return fsdp_distillation_forward(
            actor, logits.div_(temperature), batch, **spans
        )

    reduce = partial(
        _statistics,
        actor,
        batch=batch,
        temperature=temperature,
        chunk_size=actor.distillation_chunk_size,
        **spans,
    )
    token = _active_head.set((head, reduce))
    try:
        with actor.amp_context:
            stats = actor.model(**model_inputs).logits
    finally:
        _active_head.reset(token)
    return _as_output(actor, stats)
