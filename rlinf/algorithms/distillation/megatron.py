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

"""Vocabulary-parallel distillation without gathering full Student logits."""

import torch
import torch.distributed as dist
from torch import Tensor

from .config import DistillationLossConfig
from .fsdp import DistillationDistributionOutput
from .losses import compute_mc_pg_tokens


@torch.no_grad()
def select_vocab_parallel_topk(
    logits: Tensor, k: int, vocab_size: int, tp_group
) -> Tensor:
    """Select global top-k from bounded candidates on every vocabulary shard."""
    if type(k) is not int or not 1 <= k <= vocab_size:
        raise ValueError(f"Expected K in [1, {vocab_size}], got {k!r}")
    group = tp_group
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    width = logits.shape[-1]
    global_ids = torch.arange(width, device=logits.device) + rank * width
    scores = logits.detach().float().masked_fill(global_ids >= vocab_size, -torch.inf)
    local_k = min(k, width)
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :local_k]
    values = scores.gather(-1, ids)
    ids = ids + rank * width
    gathered_scores = [torch.empty_like(values) for _ in range(size)]
    gathered_ids = [torch.empty_like(ids) for _ in range(size)]
    dist.all_gather(gathered_scores, values, group=group)
    dist.all_gather(gathered_ids, ids, group=group)
    candidates = torch.cat(gathered_ids, -1)
    values = torch.cat(gathered_scores, -1)
    # Candidate order is global-ID order before stable probability ordering.
    order = candidates.argsort(dim=-1, stable=True)
    candidates, values = candidates.gather(-1, order), values.gather(-1, order)
    return candidates.gather(
        -1, values.argsort(dim=-1, descending=True, stable=True)[..., :k]
    )


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, reduce_backward):
        ctx.group = group
        ctx.reduce_backward = reduce_backward
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        if ctx.reduce_backward:
            grad = grad.clone()
            dist.all_reduce(grad, group=ctx.group)
        return grad, None, None


def compute_vocab_parallel_distribution_loss(
    local_logits: Tensor,
    teacher_ids: Tensor | None,
    teacher_logprobs: Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    vocab_size: int,
    tp_group,
    config: DistillationLossConfig,
    valid_mask: Tensor,
    behavior_logprobs: Tensor | None = None,
) -> DistillationDistributionOutput:
    """Evaluate full local shards or replicated sparse targets.

    Args:
        local_logits: Logits for this TP rank, including optional padding IDs.
        teacher_ids: Global sparse support IDs, or None for full vocabulary.
        teacher_logprobs: Sparse replicated scores or this rank's full shard.
        vocab_start: First global ID on this rank.
        vocab_end: Exclusive end of this rank's shard.
        vocab_size: Model output vocabulary size, excluding TP alignment padding.
        tp_group: Tensor-parallel process group.
        config: Distillation objective settings.
        valid_mask: Target-position mask shared across TP ranks.

    Returns:
        Replicated per-token losses with gradients for this rank's logits.
    """
    full = teacher_ids is None
    mask = valid_mask.bool()
    z = torch.where(mask.unsqueeze(-1), local_logits.float(), 0.0)
    real = torch.arange(vocab_start, vocab_end, device=z.device) < vocab_size
    z = z.masked_fill(~real, -torch.inf)
    maximum = z.detach().amax(-1, keepdim=True)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
    local_sum = (z - maximum).exp().sum(-1, keepdim=True)
    # Full shards contribute distinct losses; sparse losses are replicated.
    log_z = _Sum.apply(local_sum, tp_group, full).log() + maximum
    t = torch.where(mask.unsqueeze(-1), teacher_logprobs.detach().float(), 0.0)
    mode = config.canonical_loss_mode
    reverse = mode.startswith("reverse")
    if full:
        if t.shape != z.shape or config.topk != -1:
            raise ValueError(
                "Full TP targets must match the local vocabulary shard and topk=-1"
            )
        qlog = torch.where(real, z - log_z, 0.0)
        p = torch.where(real, t.exp(), 0.0)
        q = torch.where(real, qlog.exp(), 0.0)
        token_loss = (q * (qlog - t) if reverse else p * (t - qlog)).sum(-1)
        token_loss = _Sum.apply(token_loss, tp_group, False)
        pmass = _Sum.apply(p.sum(-1), tp_group, False)
        qmass = _Sum.apply(q.sum(-1), tp_group, False)
        tail = token_loss * 0
    else:
        ids = torch.where(mask.unsqueeze(-1), teacher_ids, 0)
        owned = (ids >= vocab_start) & (ids < min(vocab_end, vocab_size))
        local_ids = (ids - vocab_start).clamp(0, z.shape[-1] - 1)
        selected = torch.where(owned, z.gather(-1, local_ids), 0.0)
        qlog = _Sum.apply(selected, tp_group, False) - log_z
        if config.mc_k > 1:
            if behavior_logprobs is None:
                raise ValueError("MC-K requires behavior logprobs for the selected IDs")
            loss = compute_mc_pg_tokens(qlog, behavior_logprobs, t, mask, config)
            zero = torch.zeros_like(loss)
            return DistillationDistributionOutput(loss, zero, zero, zero)
        p, q = t.exp(), qlog.exp()
        pmass, qmass = p.sum(-1), q.sum(-1)
        if config.log_prob_min_clamp is not None:
            t = t.clamp_min(config.log_prob_min_clamp)
            qlog = qlog.clamp_min(config.log_prob_min_clamp)
            p = t.exp()
        token_loss = (q * (qlog - t) if reverse else p * (t - qlog)).sum(-1)
        tail = token_loss * 0
        if mode.endswith("_tail") and config.topk < vocab_size:
            # scatter_add handles dummy index collisions with actual owned IDs.
            occupied = torch.zeros_like(z, dtype=torch.long).scatter_add(
                -1, local_ids, owned.long()
            )
            outside = real & (occupied == 0)
            outside_logits = z.masked_fill(~outside, -torch.inf)
            outside_max = outside_logits.detach().amax(-1, keepdim=True)
            dist.all_reduce(outside_max, op=dist.ReduceOp.MAX, group=tp_group)
            outside_sum = _Sum.apply(
                (outside_logits - outside_max).exp().sum(-1, keepdim=True),
                tp_group,
                False,
            )
            qtail_log = (outside_sum.log() + outside_max - log_z).squeeze(-1)
            qtail = qtail_log.exp()
            ptail = (1 - pmass).clamp_min(0)
            if reverse:
                tail = qtail * (qtail_log - ptail.clamp_min(config.tail_mass_eps).log())
            else:
                safe_p = torch.where(ptail > 0, ptail, 1.0)
                tail = ptail * (
                    safe_p.log() - qtail.clamp_min(config.tail_mass_eps).log()
                )
            token_loss = token_loss + tail
    if mode == "forward_kl_topk":
        token_loss = token_loss.clamp_min(0)
    return DistillationDistributionOutput(
        *(torch.where(mask, value, 0.0) for value in (token_loss, qmass, pmass, tail))
    )
