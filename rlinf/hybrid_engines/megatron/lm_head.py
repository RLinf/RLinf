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

"""Vocabulary-parallel LM projection with chunk rematerialization."""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _chunk_distribution(hidden, weight, bias, targets, temperature, group):
    logits = F.linear(hidden, weight, bias).float() / temperature
    maximum = logits.amax(dim=-1)
    if group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    logits.sub_(maximum.unsqueeze(-1))
    rank = dist.get_rank(group) if group is not None else 0
    local_targets = targets - rank * weight.shape[0]
    owned = (local_targets >= 0) & (local_targets < weight.shape[0])
    local_targets = local_targets.clamp(0, weight.shape[0] - 1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    selected = logits[rows, local_targets].masked_fill(~owned, 0.0)
    probabilities = logits.exp()
    denominator = probabilities.sum(dim=-1)
    if group is not None:
        dist.all_reduce(denominator, group=group)
        dist.all_reduce(selected, group=group)
    probabilities.div_(denominator.unsqueeze(-1))
    log_probs = selected - denominator.log()
    return probabilities, logits, maximum, denominator, log_probs


def _accumulate_weight_gradient(hidden, dlogits, grad_weight, use_fusion):
    if use_fusion and hidden.is_cuda:
        from megatron.core.tensor_parallel.layers import fused_weight_gradient_mlp_cuda

        if grad_weight.dtype == torch.float32:
            accumulate = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32
        elif grad_weight.dtype in (torch.bfloat16, torch.float16):
            accumulate = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16
        else:
            raise ValueError(f"Unsupported main_grad dtype: {grad_weight.dtype}")
        accumulate(hidden.contiguous(), dlogits.contiguous(), grad_weight)
    else:
        grad_weight.addmm_(
            dlogits.t().to(grad_weight.dtype), hidden.to(grad_weight.dtype)
        )


# Limit the last FP32-to-parameter-dtype cast to a quarter of the weight shard.
_WEIGHT_GRAD_TILES = 4


def _accumulate_weight_tiles(tiles, hidden, dlogits):
    hidden = hidden.float()
    for tile, update in zip(
        tiles, dlogits.chunk(_WEIGHT_GRAD_TILES, dim=1), strict=True
    ):
        tile.addmm_(update.t().float(), hidden)


class _ChunkedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        bias,
        targets,
        chunk_size,
        temperature,
        group,
        calculate_entropy,
        gradient_accumulation_fusion,
    ):
        log_probs = torch.empty(
            targets.shape, dtype=torch.float32, device=hidden.device
        )
        entropy = torch.empty_like(log_probs) if calculate_entropy else None
        maxima = torch.empty_like(log_probs)
        denominators = torch.empty_like(log_probs)
        for start in range(0, targets.numel(), chunk_size):
            end = start + chunk_size
            probabilities, logits, maximum, denominator, values = _chunk_distribution(
                hidden[start:end], weight, bias, targets[start:end], temperature, group
            )
            log_probs[start:end] = values
            maxima[start:end] = maximum
            denominators[start:end] = denominator
            if calculate_entropy:
                weighted_logits = (probabilities * logits).sum(dim=-1)
                if group is not None:
                    dist.all_reduce(weighted_logits, group=group)
                entropy[start:end] = denominator.log() - weighted_logits
            del probabilities, logits
        ctx.save_for_backward(hidden, weight, bias, targets, maxima, denominators)
        ctx.chunk_size = chunk_size
        ctx.temperature = temperature
        ctx.vocab_start = (
            dist.get_rank(group) if group is not None else 0
        ) * weight.shape[0]
        ctx.main_grad = (
            getattr(weight, "main_grad", None)
            if gradient_accumulation_fusion and weight.requires_grad
            else None
        )
        if entropy is not None:
            ctx.mark_non_differentiable(entropy)
        ctx.set_materialize_grads(False)
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        hidden, weight, bias, targets, maxima, denominators = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden)
        accumulation_dtype = (
            torch.float32 if weight.dtype != torch.float64 else weight.dtype
        )
        grad_weight = ctx.main_grad
        weight_tiles = []
        if grad_weight is None and ctx.needs_input_grad[1]:
            if weight.dtype == accumulation_dtype:
                grad_weight = torch.zeros_like(weight)
            else:
                weight_tiles = [
                    torch.zeros_like(tile, dtype=accumulation_dtype)
                    for tile in weight.chunk(_WEIGHT_GRAD_TILES)
                ]
        grad_bias = (
            torch.zeros_like(bias)
            if bias is not None and ctx.needs_input_grad[2]
            else None
        )
        for start in range(0, targets.numel(), ctx.chunk_size):
            end = start + ctx.chunk_size
            # Saved row statistics avoid repeating TP collectives in backward.
            probabilities = F.linear(hidden[start:end], weight, bias).float()
            probabilities.div_(ctx.temperature).sub_(maxima[start:end, None])
            probabilities.exp_().div_(denominators[start:end, None])
            local_targets = targets[start:end] - ctx.vocab_start
            owned = (local_targets >= 0) & (local_targets < weight.shape[0])
            local_targets = local_targets.clamp(0, weight.shape[0] - 1)
            probabilities.neg_()
            rows = torch.arange(probabilities.shape[0], device=hidden.device)
            probabilities[rows, local_targets] += owned
            if grad_log_probs is None:
                probabilities.zero_()
            else:
                probabilities.mul_(grad_log_probs[start:end, None] / ctx.temperature)
            dlogits = probabilities.to(weight.dtype)
            grad_hidden[start:end] = dlogits @ weight
            if grad_weight is not None:
                _accumulate_weight_gradient(
                    hidden[start:end], dlogits, grad_weight, ctx.main_grad is not None
                )
            elif weight_tiles:
                _accumulate_weight_tiles(weight_tiles, hidden[start:end], dlogits)
            if grad_bias is not None:
                grad_bias.add_(dlogits.sum(dim=0))
            del probabilities, dlogits
        # Megatron DDP must run its ready hook without adding the gradient twice.
        if ctx.main_grad is not None:
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = None
        elif weight_tiles:
            # Release each FP32 tile before converting the next one.
            for i in range(len(weight_tiles)):
                weight_tiles[i] = weight_tiles[i].to(weight.dtype)
            grad_weight = torch.cat(weight_tiles)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None


def chunked_vocab_parallel_log_probs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    *,
    chunk_size: int,
    temperature: float = 1.0,
    bias: torch.Tensor | None = None,
    tp_group: dist.ProcessGroup | None = None,
    calculate_entropy: bool = False,
    gradient_accumulation_fusion: bool = False,
) -> dict[str, torch.Tensor]:
    """Score selected positions without retaining sequence-by-vocabulary logits.

    Args:
        hidden: Replicated hidden states, with labels-shaped leading dimensions.
            The caller owns TP input-gradient reduction (or SP reduce-scatter).
        weight: Contiguous vocabulary shard, ordered by rank in ``tp_group``.
        labels: Global vocabulary IDs for the next tokens.
        label_mask: Positions whose log probabilities are consumed by the loss.
        chunk_size: Maximum number of positions in each vocabulary projection.
        temperature: Positive sampling temperature, applied before softmax.
        bias: Optional vocabulary-sharded output bias.
        tp_group: Tensor-parallel group; None means a single vocabulary shard.
        calculate_entropy: Return detached entropy at every input position.
        gradient_accumulation_fusion: Accumulate into Megatron's main_grad buffer.

    Returns:
        Labels-shaped log_probs, zero outside label_mask, and optional entropy.
        Only log_probs is differentiable; backward rematerializes each chunk.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if hidden.shape[:-1] != labels.shape or labels.shape != label_mask.shape:
        raise ValueError("Hidden leading dimensions, labels and label_mask must match")
    if hidden.dtype != weight.dtype:
        raise ValueError(
            f"Hidden and weight dtypes must match, got {hidden.dtype}, {weight.dtype}"
        )
    flat_mask = label_mask.reshape(-1).bool()
    positions = (
        torch.arange(labels.numel(), device=labels.device)
        if calculate_entropy
        else flat_mask.nonzero(as_tuple=True)[0]
    )
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    selected, entropy = _ChunkedLogProbs.apply(
        flat_hidden[positions],
        weight,
        bias,
        labels.reshape(-1).masked_fill(~flat_mask, 0)[positions],
        chunk_size,
        temperature,
        tp_group,
        calculate_entropy,
        gradient_accumulation_fusion,
    )
    log_probs = selected.new_zeros(labels.numel()).scatter(0, positions, selected)
    result = {"log_probs": log_probs.masked_fill(~flat_mask, 0).view_as(labels)}
    if entropy is not None:
        result["entropy"] = entropy.view_as(labels)
    return result
