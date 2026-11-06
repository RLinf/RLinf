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
from collections.abc import Sequence

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
    batch_size, history_length, patch_count, width = tokens.shape
    head_count = attention.num_heads
    head_width = width // head_count
    qkv = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias)
    queries, keys, values = (
        projection.reshape(
            batch_size, history_length, patch_count, head_count, head_width
        )
        for projection in qkv.chunk(3, dim=-1)
    )
    temporal_queries, temporal_keys, temporal_values = (
        projection.permute(0, 2, 3, 1, 4).reshape(
            batch_size * patch_count, head_count, history_length, head_width
        )
        for projection in (queries, keys, values)
    )
    causal_mask = torch.ones(
        history_length, history_length, dtype=torch.bool, device=tokens.device
    ).tril()
    temporal_mask = causal_mask[None] & frame_mask[:, None, :]
    temporal_mask = (
        temporal_mask[:, None, None]
        .expand(-1, patch_count, 1, -1, -1)
        .reshape(batch_size * patch_count, 1, history_length, history_length)
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
        batch_size, patch_count, head_count, history_length, head_width
    ).permute(0, 3, 1, 2, 4)
    spatial_queries, spatial_keys, spatial_values = (
        projection.permute(0, 1, 3, 2, 4).reshape(
            batch_size * history_length, head_count, patch_count, head_width
        )
        for projection in (queries, keys, temporal_values)
    )
    output = F.scaled_dot_product_attention(
        spatial_queries, spatial_keys, spatial_values, dropout_p=dropout
    )
    output = (
        output.reshape(batch_size, history_length, head_count, patch_count, head_width)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size, history_length, patch_count, width)
    )
    return F.linear(output, attention.out_proj.weight, attention.out_proj.bias)


class ShortMemoryVisionEncoder(nn.Module):
    """Configure temporal SigLIP attention without owning backbone parameters."""

    def __init__(
        self,
        temporal_layers: Sequence[int] = (3, 7, 11, 15),
        drop_history_layer: int = 15,
    ) -> None:
        super().__init__()
        self.temporal_layers = tuple(temporal_layers)
        self.drop_history_layer = drop_history_layer
        if tuple(sorted(set(self.temporal_layers))) != self.temporal_layers:
            raise ValueError("Temporal layers must be unique and sorted.")
        if any(index < 0 or index > drop_history_layer for index in temporal_layers):
            raise ValueError(
                "Temporal layers must precede or equal the history-drop layer."
            )

    def forward(
        self,
        vision_model: nn.Module,
        images: torch.Tensor,
        *,
        frame_mask: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Encode oldest-to-current [B, K, H, W, C] into [B, P, D] tokens.

        Masks and relative time in seconds have shape [B, K]. Every rank calls
        the vision backbone once, independently of its valid-frame counts.
        """
        if images.shape[1] == 1 or torch.all(frame_mask.sum(dim=1) <= 1):
            return vision_model(images[:, -1])[0]
        if vision_model.pool_type != "none":
            raise ValueError("Short memory requires SigLIP pool_type='none'.")
        return vision_model(
            images,
            frame_mask=frame_mask,
            time_offsets=time_offsets,
            temporal_layers=self.temporal_layers,
            drop_history_layer=self.drop_history_layer,
        )[0]
