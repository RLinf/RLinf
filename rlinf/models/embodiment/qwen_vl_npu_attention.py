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

"""Qwen2.5-VL attention using Ascend's native grouped key/value heads."""

from typing import Any

import torch


def npu_grouped_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Return attention in BSND layout without expanding key/value heads.

    Boolean masks follow SDPA's convention: True permits attention. Backward,
    float masks, dropout, and non-NPU inputs retain the Transformers SDPA path.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if (
        query.device.type != "npu"
        or query.dtype not in (torch.bfloat16, torch.float16)
        # Grouped BF16 backward accumulates key/value gradients differently
        # from SDPA's repeated heads. Preserve the trained model's reductions.
        or (
            torch.is_grad_enabled()
            and (query.requires_grad or key.requires_grad or value.requires_grad)
        )
        or dropout != 0.0
        or kwargs.get("output_attentions", False)
        or kwargs.get("head_mask") is not None
        or (attention_mask is not None and attention_mask.dtype != torch.bool)
    ):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )
    if is_causal and attention_mask is not None:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    from torch_npu import npu_fusion_attention
    from transformers.integrations.npu_flash_attention import get_attn_mask_npu

    if is_causal:
        mask = get_attn_mask_npu(query.device)
        # Match SDPA's top-left causal alignment, including unequal lengths.
        sparse_mode = 2
    else:
        mask = None if attention_mask is None else ~attention_mask[..., : key.shape[2]]
        sparse_mode = 0
    output = npu_fusion_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        query.shape[1],
        "BSND",
        atten_mask=mask,
        scale=query.shape[-1] ** -0.5 if scaling is None else scaling,
        keep_prob=1.0,
        sparse_mode=sparse_mode,
    )[0]
    return output, None


def enable_qwen_vl_npu_attention(model: torch.nn.Module) -> bool:
    """Enable grouped text inference and packed vision attention on Ascend.

    Args:
        model: A Transformers Qwen2.5-VL model or its text model.

    Returns:
        Whether the Ascend backends were enabled. Other devices are unchanged.

    Raises:
        ValueError: The model is not Qwen2.5-VL.
    """
    from rlinf.scheduler import AcceleratorType, Worker

    if Worker.accelerator_type != AcceleratorType.NPU:
        return False
    if model.config.model_type not in ("qwen2_5_vl", "qwen2_5_vl_text"):
        raise ValueError("Ascend grouped attention requires a Qwen2.5-VL model.")

    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    name = "rlinf_npu_grouped"
    AttentionInterface.register(name, npu_grouped_attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    implementation = (
        {"text_config": name, "vision_config": "flash_attention_2"}
        if model.config.model_type == "qwen2_5_vl"
        else name
    )
    model.set_attn_implementation(implementation)
    return True
