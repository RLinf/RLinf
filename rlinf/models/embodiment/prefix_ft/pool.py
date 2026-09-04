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

from typing import Literal

import torch

PrefixPoolMode = Literal["masked_mean", "mean", "last"]

PREFIX_POOL_MODES = ("masked_mean", "mean", "last", "rlt_token")


def pool_prefix(
    hidden: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    mode: PrefixPoolMode = "masked_mean",
) -> torch.Tensor:
    """Pool prefix hidden states ``[B, T, D]`` to a flat feature ``[B, D]``.

    ``mode="rlt_token"`` is not handled here; adapters call the RLT encoder
    instead.
    """
    if hidden.ndim != 3:
        raise ValueError(
            f"pool_prefix expects hidden of shape [B, T, D], got {tuple(hidden.shape)}."
        )
    hidden = hidden.to(dtype=torch.float32)
    batch_size, seq_len, _ = hidden.shape

    if mode == "mean":
        return hidden.mean(dim=1)

    if mode == "last":
        if mask is None:
            return hidden[:, -1]
        keep = mask.to(device=hidden.device, dtype=torch.bool)
        if keep.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                f"pool_prefix mask shape {tuple(keep.shape)} does not match "
                f"hidden [B={batch_size}, T={seq_len}]."
            )
        lengths = keep.long().sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(batch_size, device=hidden.device)
        return hidden[batch_index, lengths]

    if mode != "masked_mean":
        raise ValueError(
            f"pool_prefix mode must be 'masked_mean', 'mean', or 'last', got {mode!r}."
        )

    if mask is None:
        return hidden.mean(dim=1)

    keep = mask.to(device=hidden.device, dtype=hidden.dtype)
    if keep.ndim == 2:
        keep = keep[..., None]
    if keep.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            f"pool_prefix mask shape {tuple(mask.shape)} does not match "
            f"hidden [B={batch_size}, T={seq_len}]."
        )
    denom = keep.sum(dim=1).clamp(min=1.0)
    return (hidden * keep).sum(dim=1) / denom
