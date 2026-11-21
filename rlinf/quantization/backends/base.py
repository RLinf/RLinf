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

"""Model-independent quantization operations and logical/native tensor representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from rlinf.quantization.config import BackendCapability

_FLOAT_DTYPES = frozenset({torch.float32, torch.bfloat16, torch.float16})
Role = Literal["activation", "weight"]


@dataclass(frozen=True)
class QuantizedTensor:
    """Codes and scales with logical shape and an explicit storage representation."""

    data: torch.Tensor
    scales: torch.Tensor
    logical_shape: tuple[int, ...]
    format: str
    storage: Literal["logical", "native"] = "logical"
    global_scale: torch.Tensor | None = None
    padded_shape: tuple[int, int] | None = None


class QuantizationBackend(ABC):
    """Stateless numerical recipe shared by actor projections and rollout caches."""

    quantization_format: str

    def __init__(self, *, block_size: int) -> None:
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size

    def autotune(self) -> AbstractContextManager[None]:
        """Return an eager tuning context, with no work for untuned backends."""
        return nullcontext()

    @abstractmethod
    def quantize(
        self, values: torch.Tensor, *, role: Role, native: bool = False
    ) -> QuantizedTensor:
        """Quantize values logically or into the backend's native GEMM representation."""

    @abstractmethod
    def dequantize(self, value: QuantizedTensor) -> torch.Tensor:
        """Restore logical values in FP32, excluding padding and gradient history."""

    @abstractmethod
    def gemm(
        self,
        activation: QuantizedTensor,
        weight: QuantizedTensor,
        *,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute with compatible native low-bit inputs on supported hardware."""

    @abstractmethod
    def probe(self, device: torch.device | str | int) -> BackendCapability:
        """Validate device and dependency compatibility and describe supported recipes."""

    def fake_quantize(self, values: torch.Tensor, *, role: Role) -> torch.Tensor:
        """Use dequantized values in forward and identity STE for the master tensor."""
        restored = self.dequantize(self.quantize(values, role=role)).to(values.dtype)
        return values + (restored - values).detach()

    def fake_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        weight_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Apply fake quantization and the model's optional floating parameter cast.

        Recipes with fixed compute precision override this operation independently
        of weight_dtype, which describes the model's floating GEMM convention.
        """
        activation = self.fake_quantize(input, role="activation")
        quantized_weight = self.fake_quantize(weight, role="weight")
        if weight_dtype is not None:
            quantized_weight = quantized_weight.to(weight_dtype)
            bias = None if bias is None else bias.to(weight_dtype)
        return F.linear(activation, quantized_weight, bias)

    def _check_native(self, *values: QuantizedTensor) -> None:
        if any(
            value.format != self.quantization_format or value.storage != "native"
            for value in values
        ):
            raise ValueError(
                "GEMM requires native inputs from the same quantization backend"
            )
