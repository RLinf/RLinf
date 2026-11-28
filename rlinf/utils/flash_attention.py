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

"""Direct FlashAttention kernel calls for packed variable-length sequences."""

from __future__ import annotations

import torch


def flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_cu_seqlens: torch.Tensor,
    key_cu_seqlens: torch.Tensor,
    max_query_length: int,
    max_key_length: int,
    *,
    backend: str,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Apply dropout-free FA2 or FA3 to already packed CUDA FP16/BF16 inputs.

    Args:
        query: Packed queries with shape ``[tokens, heads, head_dim]``.
        key: Packed keys with shape ``[tokens, kv_heads, head_dim]``.
        value: Packed values with the same shape and dtype as ``key``.
        query_cu_seqlens: Int32 cumulative query lengths, starting at zero.
        key_cu_seqlens: Int32 cumulative key lengths, starting at zero.
        max_query_length: Maximum query sequence length in the batch.
        max_key_length: Maximum key sequence length in the batch.
        backend: ``flash_attention_2`` or ``flash_attention_3``.
        softmax_scale: Score scale; ``None`` uses the kernel's head-dimension scale.
        causal: Use the kernel's bottom-right-aligned causal mask.

    Returns:
        Attention outputs with the same shape as ``query``.
    """
    if backend == "flash_attention_2":
        from flash_attn import flash_attn_varlen_func

        backend_kwargs = {"dropout_p": 0.0}
    elif backend == "flash_attention_3":
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

        backend_kwargs = {"return_attn_probs": False}
    else:
        raise ValueError(f"Unsupported FlashAttention backend {backend!r}")

    return flash_attn_varlen_func(
        query,
        key,
        value,
        query_cu_seqlens,
        key_cu_seqlens,
        max_query_length,
        max_key_length,
        softmax_scale=softmax_scale,
        causal=causal,
        **backend_kwargs,
    )
