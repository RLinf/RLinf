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

"""FlashAttention packing for Pi0.5 prefix/suffix attention."""

from __future__ import annotations

import dataclasses

import torch

from rlinf.utils.flash_attention import flash_attention_varlen


@dataclasses.dataclass(frozen=True)
class _Block:
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    query_cu_seqlens: torch.Tensor
    key_cu_seqlens: torch.Tensor
    max_query_length: int
    max_key_length: int


@dataclasses.dataclass(frozen=True)
class FlashAttentionMetadata:
    """Per-forward packing shared by all Gemma layers."""

    blocks: tuple[_Block, ...]
    valid_queries: torch.Tensor
    has_invalid_queries: bool
    # One block covers every query in its original order.
    full_query: bool
    backend: str = "flash_attention_2"


def _cu_seqlens(valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=-1, dtype=torch.int32)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)])


def prepare_flash_attention(
    prefix_mask: torch.Tensor,
    suffix_length: int,
    *,
    cached: bool = False,
    backend: str = "flash_attention_2",
) -> FlashAttentionMetadata:
    """Pack Pi0.5 attention directly from token validity and suffix length.

    Args:
        prefix_mask: Boolean prefix token validity with shape ``[batch, prefix]``.
        suffix_length: Number of suffix tokens, all of which are valid.
        cached: Query only the suffix against a previously computed prefix cache.
        backend: Explicit FlashAttention-2 or FlashAttention-3 implementation.

    Returns:
        Packing indices and cumulative lengths reused across layers.
    """
    if prefix_mask.dtype != torch.bool or prefix_mask.ndim != 2:
        raise ValueError("FlashAttention expects a boolean [batch, prefix] mask")
    batch, prefix_length = prefix_mask.shape
    key_length = prefix_length + suffix_length
    if suffix_length < 0 or key_length == 0 or (cached and suffix_length == 0):
        raise ValueError("FlashAttention requires nonempty query and key sequences")

    suffix_mask = prefix_mask.new_ones((batch, suffix_length))
    valid_keys = torch.cat([prefix_mask, suffix_mask], dim=1)
    prefix_cu_seqlens = _cu_seqlens(prefix_mask)
    suffix_cu_seqlens = (
        torch.arange(batch + 1, dtype=torch.int32, device=prefix_mask.device)
        * suffix_length
    )
    blocks = []
    if not cached and prefix_length:
        rows, columns = prefix_mask.nonzero(as_tuple=True)
        prefix_indices = rows * key_length + columns
        blocks.append(
            _Block(
                query_indices=prefix_indices,
                key_indices=prefix_indices,
                query_cu_seqlens=prefix_cu_seqlens,
                key_cu_seqlens=prefix_cu_seqlens,
                max_query_length=prefix_length,
                max_key_length=prefix_length,
            )
        )
    if suffix_length:
        if cached:
            suffix_indices = torch.arange(
                batch * suffix_length, device=prefix_mask.device
            )
        else:
            suffix_indices = (
                torch.arange(batch, device=prefix_mask.device)[:, None] * key_length
                + prefix_length
                + torch.arange(suffix_length, device=prefix_mask.device)
            ).flatten()
        blocks.append(
            _Block(
                query_indices=suffix_indices,
                key_indices=valid_keys.flatten().nonzero(as_tuple=True)[0],
                query_cu_seqlens=suffix_cu_seqlens,
                key_cu_seqlens=prefix_cu_seqlens + suffix_cu_seqlens,
                max_query_length=suffix_length,
                max_key_length=key_length,
            )
        )
    valid_queries = suffix_mask if cached else valid_keys
    query_count = sum(block.query_indices.numel() for block in blocks)
    return FlashAttentionMetadata(
        tuple(blocks),
        valid_queries,
        query_count != valid_queries.numel(),
        len(blocks) == 1 and query_count == valid_queries.numel(),
        backend,
    )


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: FlashAttentionMetadata,
) -> torch.Tensor:
    """Attend with pre-scaled queries while preserving masked-query semantics.

    Q/K/V use ``[batch, tokens, heads, head_dim]`` layout. Queries have already
    been scaled in their original dtype, matching the eager reference.
    """
    if query.device.type != "cuda" or query.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("FlashAttention requires CUDA FP16 or BF16 Q/K/V tensors")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("FlashAttention requires matching Q/K/V dtypes")

    batch, query_length, heads, head_dim = query.shape
    key_length, kv_heads = key.shape[1:3]
    flat_query = query.reshape(-1, heads, head_dim)
    flat_key = key.reshape(-1, kv_heads, head_dim)
    flat_value = value.reshape(-1, kv_heads, head_dim)
    output = None if metadata.full_query else torch.zeros_like(flat_query)
    if not any(block.query_indices.numel() for block in metadata.blocks):
        # Eager still materializes zero Q/K gradients for a fully masked batch.
        output = flat_query * 0 + flat_key.sum(dtype=torch.float32).to(query.dtype) * 0
    for block in metadata.blocks:
        if block.query_indices.numel() == 0:
            continue
        packed = flash_attention_varlen(
            flat_query if metadata.full_query else flat_query[block.query_indices],
            flat_key[block.key_indices],
            flat_value[block.key_indices],
            block.query_cu_seqlens,
            block.key_cu_seqlens,
            block.max_query_length,
            block.max_key_length,
            backend=metadata.backend,
            softmax_scale=1.0,
        )
        if metadata.full_query:
            return packed.reshape(batch, query_length, heads, head_dim)
        output.index_copy_(0, block.query_indices, packed)
    output = output.reshape(batch, query_length, heads, head_dim)
    if not metadata.has_invalid_queries:
        return output

    # Finite masked logits in eager attention produce uniform values, not zeros.
    weights = value.new_full((batch, key_length), 1.0 / key_length)
    uniform = torch.einsum("bs,bskd->bkd", weights, value)
    uniform = uniform.repeat_interleave(heads // kv_heads, dim=1)
    return torch.where(
        metadata.valid_queries[:, :, None, None], output, uniform[:, None]
    )
