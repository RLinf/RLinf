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

"""Actor fake-quantized pi0.5 projections outside ordinary Linear modules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.models.embodiment.openpi.modules.gemma import FeedForward
from rlinf.models.embodiment.openpi.modules.utils import gelu_glu
from rlinf.quantization.backends.base import QuantizationBackend


class FakeQuantFeedForward(FeedForward):
    """Apply QAT to gate/up/down GEMMs while preserving Gemma parameter keys."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        quantize_gating: bool = True,
        quantize_down: bool = True,
        *,
        backend: QuantizationBackend,
    ) -> None:
        super().__init__(features, hidden_dim)
        self.backend = backend
        self.quantize_gating = quantize_gating
        self.quantize_down = quantize_down

    @classmethod
    def from_feed_forward(
        cls,
        source: FeedForward,
        quantize_gating: bool = True,
        quantize_down: bool = True,
        *,
        backend: QuantizationBackend,
    ) -> FakeQuantFeedForward:
        # Share Parameters to avoid temporary copies of the large pi0.5 FFN weights.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.features = source.features
        result.hidden_dim = source.hidden_dim
        result.backend = backend
        result.quantize_gating = quantize_gating
        result.quantize_down = quantize_down
        result.w_gating = source.w_gating
        result.w_linear = source.w_linear
        result.train(source.training)
        if hasattr(source, "_fsdp_wrap_name"):
            result._fsdp_wrap_name = source._fsdp_wrap_name
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_gating:
            gate = self.backend.fake_linear(x, self.w_gating[0].T, weight_dtype=x.dtype)
            up = self.backend.fake_linear(x, self.w_gating[1].T, weight_dtype=x.dtype)
        else:
            gate = torch.matmul(x, self.w_gating[0].to(x.dtype))
            up = torch.matmul(x, self.w_gating[1].to(x.dtype))
        activations = gelu_glu(gate, up)
        if self.quantize_down:
            return self.backend.fake_linear(
                activations, self.w_linear.T, weight_dtype=x.dtype
            )
        return torch.matmul(activations, self.w_linear.to(x.dtype))


class FakeQuantPatchConv2d(nn.Conv2d):
    """Unfold nonoverlapping visual patches into input-block quantized GEMMs."""

    @classmethod
    def from_conv2d(
        cls,
        source: nn.Conv2d,
        *,
        backend: QuantizationBackend,
    ) -> FakeQuantPatchConv2d:
        if source.groups != 1 or source.padding_mode != "zeros":
            raise ValueError("QAT patch embedding requires groups=1 and zero padding")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        for name in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
            "output_padding",
            "transposed",
        ):
            setattr(result, name, getattr(source, name))
        result.backend = backend
        result.weight = source.weight
        result.bias = source.bias
        result.train(source.training)
        if hasattr(source, "_fsdp_wrap_name"):
            result._fsdp_wrap_name = source._fsdp_wrap_name
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError("QAT patch input must be [batch, channels, height, width]")
        batch, _, height, width = input.shape
        patches = F.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).transpose(1, 2)
        weight_rows = self.weight.reshape(self.out_channels, -1)
        output = self.backend.fake_linear(
            patches, weight_rows, self.bias, weight_dtype=input.dtype
        )
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        return output.transpose(1, 2).reshape(batch, self.out_channels, out_h, out_w)


class FakeQuantVisionMultiheadAttention(nn.MultiheadAttention):
    """Blockwise QAT for SigLIP self-attention QKV and output projections."""

    @classmethod
    def from_mha(
        cls,
        source: nn.MultiheadAttention,
        quantize_qkv: bool = True,
        quantize_out: bool = True,
        *,
        backend: QuantizationBackend,
    ) -> FakeQuantVisionMultiheadAttention:
        if (
            not source.batch_first
            or not source._qkv_same_embed_dim
            or source.bias_k is not None
            or source.bias_v is not None
            or source.add_zero_attn
            or source.dropout != 0.0
        ):
            raise ValueError(
                "QAT vision MHA requires SigLIP batch-first self-attention"
            )
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        for name in (
            "embed_dim",
            "kdim",
            "vdim",
            "_qkv_same_embed_dim",
            "num_heads",
            "dropout",
            "batch_first",
            "head_dim",
            "add_zero_attn",
        ):
            setattr(result, name, getattr(source, name))
        result.in_proj_weight = source.in_proj_weight
        result.in_proj_bias = source.in_proj_bias
        result.out_proj = source.out_proj
        result.bias_k = source.bias_k
        result.bias_v = source.bias_v
        result.backend = backend
        result.quantize_qkv = quantize_qkv
        result.quantize_out = quantize_out
        result.train(source.training)
        if hasattr(source, "_fsdp_wrap_name"):
            result._fsdp_wrap_name = source._fsdp_wrap_name
        return result

    def _project_qkv(self, query: torch.Tensor) -> torch.Tensor:
        if self.quantize_qkv:
            return self.backend.fake_linear(
                query, self.in_proj_weight, self.in_proj_bias
            )
        return F.linear(query, self.in_proj_weight, self.in_proj_bias)

    def _project_output(self, context: torch.Tensor) -> torch.Tensor:
        if self.quantize_out:
            return self.backend.fake_linear(
                context, self.out_proj.weight, self.out_proj.bias
            )
        return F.linear(context, self.out_proj.weight, self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            query is not key
            or key is not value
            or query.ndim != 3
            or key_padding_mask is not None
            or attn_mask is not None
            or is_causal
        ):
            raise ValueError("QAT vision MHA supports only unmasked self-attention")
        batch, tokens, width = query.shape
        if width != self.embed_dim:
            raise ValueError("QAT vision MHA input width does not match embed_dim")
        qkv = self._project_qkv(query)
        query_heads, key_heads, value_heads = (
            part.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        )
        scores = torch.matmul(query_heads, key_heads.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, value_heads)
        context = context.transpose(1, 2).reshape(batch, tokens, self.embed_dim)
        output = self._project_output(context)
        if not need_weights:
            return output, None
        return output, attention.mean(dim=1) if average_attn_weights else attention
