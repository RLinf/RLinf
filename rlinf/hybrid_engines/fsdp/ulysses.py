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

"""Ulysses sequence-parallel primitives for long-context VLM RL training.

Supports splitting 160k-256k multi-turn VLM sequences across an ``sp`` process
group so each GPU holds ``seq_len / sp_size`` tokens during backbone forward,
attention all-to-all, and chunked FP32 lm_head log-prob evaluation.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as functional


def get_sp_size(sp_group: dist.ProcessGroup | None) -> int:
    if sp_group is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(sp_group)


def get_sp_rank(sp_group: dist.ProcessGroup | None) -> int:
    if sp_group is None or not dist.is_initialized():
        return 0
    return dist.get_rank(sp_group)


def pad_sequence_for_sp(
    tensor: torch.Tensor, sp_size: int, *, dim: int = 1, pad_value: int | float = 0
) -> tuple[torch.Tensor, int]:
    """Right-pad ``tensor`` along ``dim`` so its length is divisible by ``sp_size``."""
    if sp_size <= 1:
        return tensor, 0
    length = tensor.shape[dim]
    remainder = length % sp_size
    if remainder == 0:
        return tensor, 0
    pad_len = sp_size - remainder
    pad_spec = [0] * (2 * tensor.ndim)
    pad_index = 2 * (tensor.ndim - 1 - (dim % tensor.ndim)) + 1
    pad_spec[pad_index] = pad_len
    return functional.pad(tensor, pad_spec, value=pad_value), pad_len


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, sp_group: dist.ProcessGroup, dim: int):
        ctx.sp_group = sp_group
        ctx.dim = dim
        ctx.sp_rank = dist.get_rank(sp_group)
        world_size = dist.get_world_size(sp_group)
        chunks = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(chunks, local_tensor.contiguous(), group=sp_group)
        return torch.cat(chunks, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        world_size = dist.get_world_size(ctx.sp_group)
        grad_chunks = grad_output.chunk(world_size, dim=ctx.dim)
        return grad_chunks[ctx.sp_rank].contiguous(), None, None


class _SliceSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, full_tensor: torch.Tensor, sp_group: dist.ProcessGroup, dim: int):
        ctx.sp_group = sp_group
        ctx.dim = dim
        world_size = dist.get_world_size(sp_group)
        rank = dist.get_rank(sp_group)
        return full_tensor.chunk(world_size, dim=dim)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        world_size = dist.get_world_size(ctx.sp_group)
        chunks = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(chunks, grad_output.contiguous(), group=ctx.sp_group)
        return torch.cat(chunks, dim=ctx.dim), None, None


class _AllToAllSeq(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sp_group: dist.ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
    ):
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all_single_dim(tensor, sp_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_single_dim(
            grad_output, ctx.sp_group, ctx.gather_dim, ctx.scatter_dim
        )
        return grad_input, None, None, None


def _all_to_all_single_dim(
    tensor: torch.Tensor,
    sp_group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    world_size = dist.get_world_size(sp_group)
    input_list = [
        chunk.contiguous() for chunk in torch.chunk(tensor, world_size, dim=scatter_dim)
    ]
    output_list = [torch.empty_like(chunk) for chunk in input_list]
    dist.all_to_all(output_list, input_list, group=sp_group)
    return torch.cat(output_list, dim=gather_dim)


def slice_sequence_for_ulysses(
    tensor: torch.Tensor,
    sp_group: dist.ProcessGroup | None,
    *,
    dim: int = 1,
) -> torch.Tensor:
    """Slice ``tensor`` along ``dim`` for the local Ulysses SP rank."""
    if get_sp_size(sp_group) <= 1:
        return tensor
    return _SliceSequence.apply(tensor, sp_group, dim)


def gather_sequence_from_ulysses(
    tensor: torch.Tensor,
    sp_group: dist.ProcessGroup | None,
    *,
    dim: int = 1,
    unpad_len: int = 0,
) -> torch.Tensor:
    """All-gather sequence shards along ``dim`` and optionally strip right padding."""
    if get_sp_size(sp_group) <= 1:
        gathered = tensor
    else:
        gathered = _GatherSequence.apply(tensor, sp_group, dim)
    if unpad_len > 0:
        keep = gathered.shape[dim] - unpad_len
        return gathered.narrow(dim, 0, keep)
    return gathered


def ulysses_all_to_all(
    tensor: torch.Tensor,
    sp_group: dist.ProcessGroup | None,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Differentiable Ulysses all-to-all swapping sequence and head dimensions."""
    if get_sp_size(sp_group) <= 1:
        return tensor
    return _AllToAllSeq.apply(tensor, sp_group, scatter_dim, gather_dim)


def local_response_window(
    response_index: torch.Tensor,
    *,
    local_seq_len: int,
    sp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global ``response_index`` positions to a local Ulysses hidden shard.

    A response token at global position ``pos`` is predicted by hidden state
    ``pos - 1``. If ``pos - 1`` falls in ``[sp_rank * local_seq_len, (sp_rank + 1)
    * local_seq_len)``, this rank owns that token's prediction.

    Returns:
        ``(local_predict_pos, owns_mask)`` both matching ``response_index`` shape.
    """
    valid = response_index > 0
    predict_from = response_index - 1
    start = int(sp_rank) * int(local_seq_len)
    end = start + int(local_seq_len)
    owns = valid & (predict_from >= start) & (predict_from < end)
    local_predict_pos = (predict_from - start).clamp(
        min=0, max=max(local_seq_len - 1, 0)
    )
    return local_predict_pos, owns


def build_ulysses_sp_groups(
    world_size: int, sp_size: int
) -> tuple[dist.ProcessGroup | None, int]:
    """Create contiguous Ulysses SP process groups of size ``sp_size``."""
    if sp_size <= 1 or not dist.is_initialized():
        return None, 1
    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size {world_size} must be divisible by ulysses_sp_size {sp_size}"
        )
    rank = dist.get_rank()
    my_group = None
    for start in range(0, world_size, sp_size):
        ranks = list(range(start, start + sp_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = group
    return my_group, sp_size
