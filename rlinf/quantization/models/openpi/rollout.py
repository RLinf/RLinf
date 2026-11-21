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

"""Rollout pi0.5 projections shared by explicitly selected numerical backends."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.models.embodiment.openpi.modules.gemma import FeedForward
from rlinf.models.embodiment.openpi.modules.utils import gelu_glu
from rlinf.quantization.backends.base import QuantizationBackend
from rlinf.quantization.layers.linear import quantized_linear
from rlinf.quantization.layers.weights import QuantizedWeight
from rlinf.quantization.models.openpi.actor import (
    FakeQuantFeedForward,
    FakeQuantPatchConv2d,
    FakeQuantVisionMultiheadAttention,
)


class QuantizedFeedForward(FakeQuantFeedForward):
    """Compute gate/up/down using prepared quantized weights."""

    @classmethod
    def from_feed_forward(
        cls,
        source: FeedForward,
        quantize_gating: bool = True,
        quantize_down: bool = True,
        *,
        backend: QuantizationBackend,
    ) -> QuantizedFeedForward:
        result = super().from_feed_forward(
            source,
            quantize_gating=quantize_gating,
            quantize_down=quantize_down,
            backend=backend,
        )
        if quantize_gating:
            result.gate_weight = QuantizedWeight(backend)
            result.up_weight = QuantizedWeight(backend)
        if quantize_down:
            result.down_weight = QuantizedWeight(backend)
        return result

    def load_weights(self) -> None:
        if self.quantize_gating:
            for index, weight in enumerate((self.gate_weight, self.up_weight)):
                weight.load_weights(self.w_gating[index].T)
        if self.quantize_down:
            self.down_weight.load_weights(self.w_linear.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_gating:
            gate = quantized_linear(x, self.gate_weight)
            up = quantized_linear(x, self.up_weight)
        else:
            gate = torch.matmul(x, self.w_gating[0].to(x.dtype))
            up = torch.matmul(x, self.w_gating[1].to(x.dtype))
        activations = gelu_glu(gate, up)
        if self.quantize_down:
            return quantized_linear(activations, self.down_weight)
        return torch.matmul(activations, self.w_linear.to(x.dtype))


class QuantizedPatchConv2d(FakeQuantPatchConv2d):
    """Quantize unfolded patches dynamically and pack weights only on loading."""

    @classmethod
    def from_conv2d(
        cls,
        source: nn.Conv2d,
        *,
        backend: QuantizationBackend,
    ) -> QuantizedPatchConv2d:
        result = super().from_conv2d(source, backend=backend)
        result.quantized_weight = QuantizedWeight(backend)
        return result

    def load_weights(self) -> None:
        self.quantized_weight.load_weights(
            self.weight.reshape(self.out_channels, -1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError(
                "rollout patch input must be [batch, channels, height, width]"
            )
        batch, _, height, width = input.shape
        patches = F.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).transpose(1, 2)
        output = quantized_linear(patches, self.quantized_weight, self.bias)
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return output.transpose(1, 2).reshape(batch, self.out_channels, out_h, out_w)


class QuantizedVisionMultiheadAttention(FakeQuantVisionMultiheadAttention):
    """Preserve SigLIP self-attention with low-bit QKV and output projections."""

    @classmethod
    def from_mha(
        cls,
        source: nn.MultiheadAttention,
        quantize_qkv: bool = True,
        quantize_out: bool = True,
        *,
        backend: QuantizationBackend,
    ) -> QuantizedVisionMultiheadAttention:
        result = super().from_mha(
            source,
            quantize_qkv=quantize_qkv,
            quantize_out=quantize_out,
            backend=backend,
        )
        if quantize_qkv:
            result.qkv_weight = QuantizedWeight(backend)
        if quantize_out:
            result.out_weight = QuantizedWeight(backend)
        return result

    def load_weights(self) -> None:
        if self.quantize_qkv:
            self.qkv_weight.load_weights(self.in_proj_weight)
        if self.quantize_out:
            self.out_weight.load_weights(self.out_proj.weight)

    def _project_qkv(self, query: torch.Tensor) -> torch.Tensor:
        if self.quantize_qkv:
            return quantized_linear(query, self.qkv_weight, self.in_proj_bias)
        return F.linear(query, self.in_proj_weight, self.in_proj_bias)

    def _project_output(self, context: torch.Tensor) -> torch.Tensor:
        if self.quantize_out:
            return quantized_linear(context, self.out_weight, self.out_proj.bias)
        return F.linear(context, self.out_proj.weight, self.out_proj.bias)
