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
from collections.abc import Callable
from functools import partial

import torch


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
    backend: str = "flash_attention_2"


def _cu_seqlens(valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=-1, dtype=torch.int32)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)])


def _flash_attention_kernel(backend: str) -> Callable[..., torch.Tensor]:
    if backend == "flash_attention_2":
        from flash_attn import flash_attn_varlen_func

        return partial(
            flash_attn_varlen_func, dropout_p=0.0, softmax_scale=1.0, causal=False
        )
    if backend == "flash_attention_3":
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

        return partial(
            flash_attn_varlen_func,
            softmax_scale=1.0,
            causal=False,
            return_attn_probs=False,
        )
    raise ValueError(f"Unsupported FlashAttention backend {backend!r}")


def prepare_flash_attention(
    mask: torch.Tensor,
    prefix_length: int,
    suffix_length: int,
    backend: str = "flash_attention_2",
) -> FlashAttentionMetadata:
    """Pack a Pi0.5 joint, prefix-only, or cached-suffix visibility mask.

    Args:
        mask: Boolean visibility mask with shape ``[batch, queries, keys]``.
        prefix_length: Current prefix query count, zero for a cached suffix.
        suffix_length: Current suffix query count, zero for prefix prefill.
        backend: Explicit FlashAttention-2 or FlashAttention-3 implementation.

    Returns:
        Packing indices and cumulative lengths reused across layers.
    """
    query_length, key_length = mask.shape[-2:]
    if mask.dtype != torch.bool or query_length != prefix_length + suffix_length:
        raise ValueError("FlashAttention expects a boolean Pi0.5 prefix/suffix mask")
    if prefix_length and key_length != query_length:
        raise ValueError("Pi0.5 prefix queries cannot append to an existing KV cache")

    valid_queries = mask.any(dim=-1)
    valid_keys = mask.any(dim=-2)
    visibility = torch.ones(
        query_length, key_length, dtype=torch.bool, device=mask.device
    )
    if prefix_length and suffix_length:
        visibility[:prefix_length, prefix_length:] = False
    expected = valid_queries[:, :, None] & valid_keys[:, None, :] & visibility[None]
    if not torch.equal(mask, expected):
        raise ValueError(
            "FlashAttention supports Pi0.5 bidirectional prefix/suffix blocks; "
            "the supplied attention mask has different visibility"
        )

    spans = (
        [(0, prefix_length, prefix_length), (prefix_length, query_length, key_length)]
        if prefix_length and suffix_length
        else [(0, query_length, key_length)]
    )
    blocks = []
    for start, end, key_end in spans:
        queries = valid_queries.clone()
        queries[:, :start] = False
        queries[:, end:] = False
        keys = valid_keys.clone()
        keys[:, key_end:] = False
        blocks.append(
            _Block(
                query_indices=queries.flatten().nonzero(as_tuple=True)[0],
                key_indices=keys.flatten().nonzero(as_tuple=True)[0],
                query_cu_seqlens=_cu_seqlens(queries),
                key_cu_seqlens=_cu_seqlens(keys),
                max_query_length=end - start,
                max_key_length=key_end,
            )
        )
    query_count = sum(block.query_indices.numel() for block in blocks)
    return FlashAttentionMetadata(
        tuple(blocks),
        valid_queries,
        query_count != valid_queries.numel(),
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

    kernel = _flash_attention_kernel(metadata.backend)

    batch, query_length, heads, head_dim = query.shape
    key_length, kv_heads = key.shape[1:3]
    flat_query = query.reshape(-1, heads, head_dim)
    flat_key = key.reshape(-1, kv_heads, head_dim)
    flat_value = value.reshape(-1, kv_heads, head_dim)
    output = torch.zeros_like(flat_query)
    if not any(block.query_indices.numel() for block in metadata.blocks):
        # Eager still materializes zero Q/K gradients for a fully masked batch.
        output = flat_query * 0 + flat_key.sum(dtype=torch.float32).to(query.dtype) * 0
    for block in metadata.blocks:
        if block.query_indices.numel() == 0:
            continue
        packed = kernel(
            flat_query[block.query_indices],
            flat_key[block.key_indices],
            flat_value[block.key_indices],
            block.query_cu_seqlens,
            block.key_cu_seqlens,
            block.max_query_length,
            block.max_key_length,
        )
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
