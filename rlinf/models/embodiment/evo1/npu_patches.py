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

"""Bound temporary attention matrices in Evo's frozen InternVL vision tower."""

from types import MethodType
from typing import Any

import torch


def _chunked_naive_attention(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size = self._rlinf_attention_batch_size
    if (
        hidden_states.device.type != "npu"
        or hidden_states.dtype not in (torch.bfloat16, torch.float16)
        or hidden_states.shape[0] <= batch_size
        or (self.training and self.attn_drop.p != 0)
        or (
            torch.is_grad_enabled()
            and (
                hidden_states.requires_grad
                or any(parameter.requires_grad for parameter in self.parameters())
            )
        )
    ):
        return self._rlinf_naive_attention(hidden_states)

    batch, length, channels = hidden_states.shape
    qkv = (
        self.qkv(hidden_states)
        .reshape(batch, length, 3, self.num_heads, channels // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)
    if self.qk_normalization:
        query = (
            self.q_norm(query.transpose(1, 2).flatten(-2, -1))
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_norm(key.transpose(1, 2).flatten(-2, -1))
            .view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
    scaled_query = query * self.scale
    contexts = []
    for start in range(0, batch, batch_size):
        stop = start + batch_size
        scores = scaled_query[start:stop] @ key[start:stop].transpose(-2, -1)
        attention = self.attn_drop(scores.softmax(dim=-1))
        contexts.append(attention @ value[start:stop])
        del scores, attention
    context = (
        torch.cat(contexts, dim=0).transpose(1, 2).reshape(batch, length, channels)
    )
    return self.proj_drop(self.proj(context))


def set_vision_attention_batch_size(
    vision_model: torch.nn.Module, batch_size: int
) -> int:
    """Limit frames per naive NPU attention matrix; zero restores native calls.

    Projections retain the full batch to preserve their BF16 rounding. Applies
    to frozen InternAttention modules; dropout and gradient paths stay native.
    Returns the number of modules enabled. Other accelerators are unchanged.
    """
    from rlinf.scheduler import AcceleratorType, Worker

    if batch_size < 0:
        raise ValueError("vision_attention_batch_size must be nonnegative")
    if Worker.accelerator_type != AcceleratorType.NPU:
        return 0
    enabled = 0
    for module in vision_model.modules():
        if type(module).__name__ != "InternAttention" or module.use_flash_attn:
            continue
        if not hasattr(module, "_rlinf_naive_attention"):
            module._rlinf_naive_attention = module._naive_attn
        module._rlinf_attention_batch_size = batch_size
        module._naive_attn = (
            MethodType(_chunked_naive_attention, module)
            if batch_size
            else module._rlinf_naive_attention
        )
        enabled += bool(batch_size)
    return enabled
