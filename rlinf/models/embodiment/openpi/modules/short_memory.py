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

"""Short visual memory through factorized SigLIP space-time attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_relative_time_encoding(
    time_offsets: torch.Tensor, width: int
) -> torch.Tensor:
    """Encode seconds relative to the current frame as ``PE(t) - PE(0)``."""
    frequencies = torch.exp(
        torch.linspace(
            0.0,
            -math.log(10_000.0),
            width // 2,
            device=time_offsets.device,
            dtype=torch.float32,
        )
    )
    angles = time_offsets.float().unsqueeze(-1) * frequencies
    return torch.cat([angles.sin(), angles.cos() - 1.0], dim=-1).to(time_offsets.dtype)


def factorized_space_time_attention(
    attention: nn.MultiheadAttention,
    tokens: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Mix values causally over time, then spatially, sharing SigLIP projections."""
    batch_size, num_frames, num_patches, width = tokens.shape
    num_heads = attention.num_heads
    head_dim = width // num_heads
    qkv = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias)
    queries, keys, values = (
        projection.reshape(batch_size, num_frames, num_patches, num_heads, head_dim)
        for projection in qkv.chunk(3, dim=-1)
    )
    temporal_queries, temporal_keys, temporal_values = (
        projection.permute(0, 2, 3, 1, 4).reshape(
            batch_size * num_patches, num_heads, num_frames, head_dim
        )
        for projection in (queries, keys, values)
    )
    causal_mask = torch.ones(
        num_frames, num_frames, dtype=torch.bool, device=tokens.device
    ).tril()
    # Padded queries attend to themselves to avoid empty rows in cuDNN attention.
    temporal_mask = (causal_mask[None] & frame_mask[:, None, :]) | torch.eye(
        num_frames, dtype=torch.bool, device=tokens.device
    )
    temporal_mask = (
        temporal_mask[:, None, None]
        .expand(-1, num_patches, 1, -1, -1)
        .reshape(batch_size * num_patches, 1, num_frames, num_frames)
    )
    dropout = attention.dropout if attention.training else 0.0
    temporal_values = F.scaled_dot_product_attention(
        temporal_queries,
        temporal_keys,
        temporal_values,
        attn_mask=temporal_mask,
        dropout_p=dropout,
    )
    temporal_values = temporal_values.reshape(
        batch_size, num_patches, num_heads, num_frames, head_dim
    ).permute(0, 3, 1, 2, 4)
    spatial_queries, spatial_keys, spatial_values = (
        projection.permute(0, 1, 3, 2, 4).reshape(
            batch_size * num_frames, num_heads, num_patches, head_dim
        )
        for projection in (queries, keys, temporal_values)
    )
    output = F.scaled_dot_product_attention(
        spatial_queries, spatial_keys, spatial_values, dropout_p=dropout
    )
    output = (
        output.reshape(batch_size, num_frames, num_heads, num_patches, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size, num_frames, num_patches, width)
    )
    return F.linear(output, attention.out_proj.weight, attention.out_proj.bias)
