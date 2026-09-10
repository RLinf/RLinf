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

"""Ascend fused attention integration for packed Hugging Face inputs."""

from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_CAUSAL_MASK_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_CAUSAL_MASK_SIZE = 2048
_ORIGINAL_QWEN2_UPDATE_CAUSAL_MASK = None
_ORIGINAL_QWEN2_CREATE_CAUSAL_MASK = None


def _get_actual_seq_lengths(position_ids: torch.Tensor) -> tuple[int, ...]:
    """Return cumulative sequence lengths encoded by packed position ids."""
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise ValueError(
            "ascend_fusion only supports packed inputs with shape [1, sequence_length]."
        )

    sequence_length = position_ids.shape[1]
    sequence_starts = torch.nonzero(position_ids[0] == 0, as_tuple=False).flatten()
    if sequence_starts.numel() == 0 or sequence_starts[0].item() != 0:
        raise ValueError("Packed position_ids must start at zero.")

    sequence_ends = torch.cat(
        (
            sequence_starts[1:],
            torch.tensor([sequence_length], device=position_ids.device),
        )
    )
    return tuple(sequence_ends.cpu().tolist())


def _get_causal_mask(query: torch.Tensor) -> torch.Tensor:
    """Return Ascend's compressed causal mask for sparse mode 3.

    ``npu_fusion_attention`` requires a fixed [2048, 2048] mask in this
    mode, independently of the actual variable sequence lengths.
    """
    cache_key = (str(query.device), query.dtype)
    mask = _CAUSAL_MASK_CACHE.get(cache_key)
    if mask is None:
        mask = torch.triu(
            torch.ones(
                (_CAUSAL_MASK_SIZE, _CAUSAL_MASK_SIZE),
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=1,
        )
        _CAUSAL_MASK_CACHE[cache_key] = mask
    return mask


def ascend_fusion_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run packed causal attention with torch_npu.npu_fusion_attention.

    The Transformers FlashAttention2 integration derives cumulative sequence
    lengths from packed position_ids. This implementation applies the same
    convention to Ascend's TND fused-attention interface.
    """
    del module
    if attention_mask is not None:
        raise ValueError(
            "ascend_fusion expects packed inputs with attention_mask=None."
        )
    if dropout != 0.0:
        raise ValueError("ascend_fusion does not support attention dropout.")
    if kwargs.get("sliding_window") is not None:
        raise ValueError("ascend_fusion does not support sliding-window attention.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "ascend_fusion requires FP16 or BF16 query/key/value; "
            "set actor.model.precision to fp16 or bf16."
        )

    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        raise ValueError("ascend_fusion requires position_ids for packed inputs.")

    actual_seq_lengths = _get_actual_seq_lengths(position_ids)
    causal_mask = _get_causal_mask(query)

    batch_size, num_heads, sequence_length, head_dim = query.shape
    if batch_size != 1:
        raise ValueError("ascend_fusion only supports packed batch size 1.")

    query_tnd = query.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()
    key_tnd = key.transpose(1, 2).reshape(-1, key.shape[1], head_dim).contiguous()
    value_tnd = value.transpose(1, 2).reshape(-1, value.shape[1], head_dim).contiguous()

    import torch_npu

    output = torch_npu.npu_fusion_attention(
        query_tnd,
        key_tnd,
        value_tnd,
        head_num=num_heads,
        input_layout="TND",
        atten_mask=causal_mask,
        scale=scaling if scaling is not None else head_dim**-0.5,
        keep_prob=1.0,
        actual_seq_qlen=actual_seq_lengths,
        actual_seq_kvlen=actual_seq_lengths,
        sparse_mode=3,
    )[0]
    return output.view(batch_size, sequence_length, num_heads, head_dim), None


def _patch_qwen2_causal_mask() -> None:
    """Skip dense mask construction for packed Ascend fusion attention.

    Transformers 4.x implements this in ``Qwen2Model._update_causal_mask``.
    Newer Transformers releases construct masks through the module-level
    ``create_causal_mask`` helper instead. Patch the applicable interface so
    the custom backend always receives ``attention_mask=None``.
    """
    global _ORIGINAL_QWEN2_UPDATE_CAUSAL_MASK, _ORIGINAL_QWEN2_CREATE_CAUSAL_MASK

    from transformers.models.qwen2 import modeling_qwen2

    qwen2_model = modeling_qwen2.Qwen2Model
    if hasattr(qwen2_model, "_update_causal_mask"):
        if _ORIGINAL_QWEN2_UPDATE_CAUSAL_MASK is not None:
            return

        original_update_causal_mask = qwen2_model._update_causal_mask

        def update_causal_mask(self, attention_mask, *args, **kwargs):
            if (
                self.config._attn_implementation == "ascend_fusion"
                and attention_mask is None
            ):
                return None
            return original_update_causal_mask(self, attention_mask, *args, **kwargs)

        qwen2_model._update_causal_mask = update_causal_mask
        _ORIGINAL_QWEN2_UPDATE_CAUSAL_MASK = original_update_causal_mask
        return

    if _ORIGINAL_QWEN2_CREATE_CAUSAL_MASK is not None:
        return
    if not hasattr(modeling_qwen2, "create_causal_mask"):
        raise RuntimeError(
            "Unsupported Transformers Qwen2 mask API; expected "
            "_update_causal_mask or create_causal_mask."
        )

    original_create_causal_mask = modeling_qwen2.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        config = kwargs.get("config")
        if (
            config is not None
            and config._attn_implementation == "ascend_fusion"
            and kwargs.get("attention_mask") is None
        ):
            return None
        return original_create_causal_mask(*args, **kwargs)

    modeling_qwen2.create_causal_mask = create_causal_mask
    _ORIGINAL_QWEN2_CREATE_CAUSAL_MASK = original_create_causal_mask


def register_ascend_fusion_attention() -> None:
    """Register the Ascend fused attention backend with Transformers."""
    ALL_ATTENTION_FUNCTIONS.register("ascend_fusion", ascend_fusion_attention_forward)
    _patch_qwen2_causal_mask()
