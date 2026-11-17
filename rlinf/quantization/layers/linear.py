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

"""Fake and native quantized Linear layers sharing floating-point master parameters."""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from rlinf.quantization.backends.base import QuantizationBackend
from rlinf.quantization.layers.weights import QuantizedWeight


@torch.no_grad()
def quantized_linear(
    input: torch.Tensor, weight: QuantizedWeight, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply dynamic activation quantization and native GEMM with prepared weights."""
    value = weight.value
    n, k = value.logical_shape
    if input.ndim < 1 or input.shape[-1] != k:
        raise ValueError("quantized linear input width does not match weights")
    activation = weight.backend.quantize(
        input.reshape(-1, k), role="activation", native=True
    )
    output = weight.backend.gemm(activation, value, bias=bias, output_dtype=input.dtype)
    return output.reshape(*input.shape[:-1], n)


class BaseQuantizedLinear(nn.Linear):
    """Share floating parameters and conversion logic between quantized Linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        backend: QuantizationBackend,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.backend = backend

    @classmethod
    def from_linear(cls, source: nn.Linear, *, backend: QuantizationBackend) -> Self:
        """Reuse source parameters, training state, checkpoint keys, and FSDP markers."""
        result = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            backend=backend,
            device=torch.device("meta"),
            dtype=source.weight.dtype,
        )
        result.weight, result.bias = source.weight, source.bias
        result.train(source.training)
        if hasattr(source, "_fsdp_wrap_name"):
            result._fsdp_wrap_name = source._fsdp_wrap_name
        return result


class FakeQuantLinear(BaseQuantizedLinear):
    """Fake-quantized actor projection with gradients to the floating master weights."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.backend.fake_linear(input, self.weight, self.bias)


class QuantizedLinear(BaseQuantizedLinear):
    """Native low-bit Linear with externally synchronized floating master weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        backend: QuantizationBackend,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias, backend=backend, device=device, dtype=dtype
        )
        self.quantized_weight = QuantizedWeight(backend)

    def load_weights(self) -> None:
        self.quantized_weight.load_weights(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantized_linear(input, self.quantized_weight, self.bias)
