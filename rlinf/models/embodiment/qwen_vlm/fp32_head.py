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

"""Chunked fp32 log-probabilities of the language-model head.

Training never stores the ``[token, vocab]`` logits. Each chunk is a GEMM of the
bf16 hidden state and the head weight accumulated in fp32. The chunk keeps
only the log-probability of the target ids and the entropy, then it is freed.
The backward recomputes that same chunk from the saved hidden state.
"""

from __future__ import annotations

import torch


def _fp32_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """``left @ right`` with the products summed in fp32."""
    if left.is_cuda:
        return torch.mm(left, right, out_dtype=torch.float32)
    return torch.mm(left.float(), right.float())


def _fp32_addmm(total: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> None:
    """Add ``left @ right`` into ``total``, summing the products in fp32."""
    if left.is_cuda:
        torch.addmm(total, left, right, out_dtype=torch.float32, out=total)
        return
    total.addmm_(left.float(), right.float())


class Fp32HeadLogProbs(torch.autograd.Function):
    """Log-probs at given ids and the entropy of the head logits, one chunk of rows at a time."""

    # 4096 rows is about 3.8 GiB of fp32 logits at Qwen3.8's vocabulary.
    CHUNK_ROWS = 4096

    @staticmethod
    def forward(ctx, hidden_states, weight, ids, temperature):
        hidden_states = hidden_states.contiguous()
        ids = ids.contiguous()
        temperature = float(temperature)
        count = hidden_states.shape[0]
        log_probs = hidden_states.new_empty(ids.shape, dtype=torch.float32)
        log_normalizers = hidden_states.new_empty(count, dtype=torch.float32)
        entropy = hidden_states.new_empty(count, dtype=torch.float32)

        for rows in Fp32HeadLogProbs.chunks(count):
            logits = _fp32_mm(hidden_states[rows], weight.t()).div_(temperature)
            log_normalizers[rows] = logits.logsumexp(-1)
            log_probs[rows] = logits.gather(1, ids[rows]) - log_normalizers[rows, None]
            log_p = logits.sub_(log_normalizers[rows, None])
            entropy[rows] = -log_p.exp().mul_(log_p).sum(-1)
            del logits

        ctx.save_for_backward(hidden_states, weight, ids, log_normalizers, entropy)
        ctx.temperature = temperature
        ctx.set_materialize_grads(False)
        return log_probs, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_probs, grad_entropy):
        hidden_states, weight, ids, log_normalizers, entropy = ctx.saved_tensors
        grad_hidden_states = torch.empty_like(hidden_states)
        grad_weight = torch.zeros(
            weight.shape, dtype=torch.float32, device=weight.device
        )
        grad_log_probs = grad_log_probs.float()

        for rows in Fp32HeadLogProbs.chunks(hidden_states.shape[0]):
            log_p = _fp32_mm(hidden_states[rows], weight.t()).div_(ctx.temperature)
            log_p.sub_(log_normalizers[rows, None])
            probs = log_p.exp()
            grad_logits = probs * -grad_log_probs[rows].sum(-1, keepdim=True)
            if grad_entropy is not None:
                grad_logits = grad_logits - log_p.add_(entropy[rows, None]).mul_(
                    probs
                ).mul_(grad_entropy[rows, None])
            for column in range(ids.shape[1]):
                grad_logits.scatter_add_(
                    1, ids[rows, column, None], grad_log_probs[rows, column, None]
                )
            grad_logits = grad_logits.div_(ctx.temperature).to(hidden_states.dtype)
            grad_hidden_states[rows] = grad_logits @ weight
            _fp32_addmm(grad_weight, grad_logits.t(), hidden_states[rows])
            del log_p, probs, grad_logits

        return grad_hidden_states, grad_weight.to(dtype=weight.dtype), None, None

    @staticmethod
    def chunks(count: int):
        for start in range(0, count, Fp32HeadLogProbs.CHUNK_ROWS):
            yield slice(start, min(start + Fp32HeadLogProbs.CHUNK_ROWS, count))


def response_logprobs_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    response_index: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob and entropy of the response tokens, without storing the logits.

    ``response_index`` addresses ``input_ids``. The token at that position is
    predicted by the hidden state one step earlier. A nonpositive index is
    padding and both returned values there are zero.

    Args:
        hidden: Backbone states, ``[batch, sequence, hidden]``.
        weight: ``lm_head`` weight, ``[vocab, hidden]``. Read inside the FSDP
            forward so the gathered parameter, not a shard, is what this sees.
        input_ids: Token ids, ``[batch, sequence]``.
        response_index: Positions in ``input_ids``, ``[batch, responses]``.
        temperature: Divides the fp32 logits. Training uses 1.

    Returns:
        Log-probs and entropy, each ``[batch, responses]`` in fp32.
    """
    valid = response_index > 0
    log_probs = hidden.new_zeros(response_index.shape, dtype=torch.float32)
    entropy = hidden.new_zeros(response_index.shape, dtype=torch.float32)
    if not bool(valid.any()):
        return log_probs, entropy

    batch_index = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    predict_from = response_index.clamp(min=1) - 1
    token_ids = input_ids.gather(1, response_index.clamp(min=1))
    selected_hidden = hidden[batch_index, predict_from][valid]
    selected_ids = token_ids[valid].unsqueeze(-1)
    selected_log_probs, selected_entropy = Fp32HeadLogProbs.apply(
        selected_hidden.to(dtype=weight.dtype),
        weight,
        selected_ids,
        float(temperature),
    )
    log_probs[valid] = selected_log_probs[:, 0]
    entropy[valid] = selected_entropy
    return log_probs, entropy


def response_logprobs_and_heads_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    response_index: torch.Tensor,
    head_ids: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute response log-probs, entropy, and top-K head log-probs in one chunked fp32 GEMM.

    Args:
        hidden: Backbone states, ``[batch, sequence, hidden]``.
        weight: ``lm_head`` weight, ``[vocab, hidden]``.
        input_ids: Token ids, ``[batch, sequence]``.
        response_index: Positions in ``input_ids``, ``[batch, responses]``.
        head_ids: Sampler top-K head token ids, ``[batch, responses, K]``.
        temperature: Divides the fp32 logits. Training uses 1.

    Returns:
        ``(log_probs, entropy, train_head_log_probs)`` in fp32 with shapes
        ``[batch, responses]``, ``[batch, responses]``, and
        ``[batch, responses, K]``.
    """
    valid = response_index > 0
    k = int(head_ids.shape[-1])
    log_probs = hidden.new_zeros(response_index.shape, dtype=torch.float32)
    entropy = hidden.new_zeros(response_index.shape, dtype=torch.float32)
    train_head_log_probs = hidden.new_zeros(
        (*response_index.shape, k), dtype=torch.float32
    )
    if not bool(valid.any()):
        return log_probs, entropy, train_head_log_probs

    batch_index = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    predict_from = response_index.clamp(min=1) - 1
    token_ids = input_ids.gather(1, response_index.clamp(min=1))
    selected_hidden = hidden[batch_index, predict_from][valid]
    selected_target = token_ids[valid].unsqueeze(-1)
    selected_heads = head_ids[valid].to(device=selected_target.device, dtype=torch.long)
    combined_ids = torch.cat([selected_target, selected_heads], dim=-1)
    combined_log_probs, selected_entropy = Fp32HeadLogProbs.apply(
        selected_hidden.to(dtype=weight.dtype),
        weight,
        combined_ids,
        float(temperature),
    )
    log_probs[valid] = combined_log_probs[:, 0]
    entropy[valid] = selected_entropy
    train_head_log_probs[valid] = combined_log_probs[:, 1:]
    return log_probs, entropy, train_head_log_probs
