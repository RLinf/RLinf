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

from rlinf.scheduler import AcceleratorType, Worker


def _chunked_naive_attention(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size = self._rlinf_attention_batch_size
    if (
        hidden_states.device.type != Worker.torch_device_type
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


def _masked_qwen_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: Any,
    attention_mask: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, Any]:
    attn_output, attn_weights = self._rlinf_orig_forward(
        hidden_states, position_embeddings, attention_mask, *args, **kwargs
    )
    if (
        attention_mask is not None
        and attention_mask.ndim == 4
        and attention_mask.shape[-2] == attention_mask.shape[-1]
    ):
        diag = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
        valid_q = diag if diag.dtype == torch.bool else (diag == 0)
        attn_output = attn_output * valid_q.transpose(1, 2).to(attn_output.dtype)
    return attn_output, attn_weights


def mask_padded_language_attention(language_model: torch.nn.Module) -> int:
    """Zero masked query outputs on non-FA2 Qwen2 attention to match FA2 pad_input.

    Evo-1's FlowmatchingActionHead cross-attends over all fused VLM tokens without
    a key padding mask, relying on FA2's unpad/pad cycle to leave masked-view and
    right-padded query positions at zero in each decoder layer.
    """
    config = getattr(language_model, "config", None)
    if getattr(config, "_attn_implementation", None) == "flash_attention_2":
        return 0
    patched = 0
    for module in language_model.modules():
        if type(module).__name__ != "Qwen2Attention":
            continue
        if not hasattr(module, "_rlinf_orig_forward"):
            module._rlinf_orig_forward = module.forward
            module.forward = MethodType(_masked_qwen_attention_forward, module)
        patched += 1
    return patched
