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

"""Blockwise INT8 reference numerics and native execution backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.quantization.backends.base import QuantizationBackend, QuantizedTensor, Role
from rlinf.quantization.config import BackendCapability


def quantize_int8_values(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round to nearest even with the given scale and saturate to signed INT8."""
    return torch.round(values.float() / scale.float()).clamp(-128, 127).to(torch.int8)


def quantize_int8_blocks(
    values: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dimension with [flattened rows, blocks] scales."""
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("INT8 block quantization requires a nonempty last dimension")
    k = values.shape[-1]
    flat = values.reshape(-1, k).float()
    blocks = (k + block_size - 1) // block_size
    padded = F.pad(flat, (0, blocks * block_size - k))
    grouped = padded.reshape(-1, blocks, block_size)
    scales = (grouped.abs().amax(dim=-1) * (1.0 / 127.0)).clamp_min(1e-12)
    quantized = quantize_int8_values(grouped, scales.unsqueeze(-1))
    return quantized.reshape(-1, blocks * block_size)[:, :k].reshape(
        values.shape
    ), scales


class Int8Backend(QuantizationBackend):
    """Signed INT8 with dynamic block activations and block weights."""

    quantization_format = "int8"

    @torch.no_grad()
    def quantize(
        self, values: torch.Tensor, *, role: Role, native: bool = False
    ) -> QuantizedTensor:
        if native:
            return self._quantize_native(values, role)
        else:
            return self._quantize_logical(values)

    def _quantize_logical(self, values: torch.Tensor) -> QuantizedTensor:
        data, scales = quantize_int8_blocks(values, self.block_size)
        return QuantizedTensor(
            data, scales, tuple(values.shape), self.quantization_format
        )

    def _quantize_native(self, values: torch.Tensor, role: Role) -> QuantizedTensor:
        from rlinf.quantization.kernels.int8 import quantize_int8_cuda_blocks

        if role == "weight" and not torch.isfinite(values).all():
            raise ValueError("native INT8 weights must be finite")
        data, scales = quantize_int8_cuda_blocks(values, self.block_size)
        return QuantizedTensor(
            data,
            scales,
            tuple(values.shape),
            self.quantization_format,
            storage="native",
            padded_shape=tuple(data.shape),
        )

    def dequantize(self, value: QuantizedTensor) -> torch.Tensor:
        if value.storage == "logical":
            return self._dequantize_logical(value)
        elif value.storage == "native":
            return self._dequantize_native(value)
        raise ValueError(f"INT8 storage {value.storage!r} is not supported")

    def _dequantize_logical(self, value: QuantizedTensor) -> torch.Tensor:
        return self._dequantize_blocks(value.data, value)

    def _dequantize_native(self, value: QuantizedTensor) -> torch.Tensor:
        return self._dequantize_blocks(value.data[:, : value.logical_shape[-1]], value)

    def _dequantize_blocks(
        self, codes: torch.Tensor, value: QuantizedTensor
    ) -> torch.Tensor:
        k = value.logical_shape[-1]
        scales = value.scales.repeat_interleave(self.block_size, dim=-1)[:, :k]
        return (codes.reshape(-1, k).float() * scales).reshape(value.logical_shape)

    def gemm(
        self,
        activation: QuantizedTensor,
        weight: QuantizedTensor,
        *,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        from rlinf.quantization.kernels.int8 import int8_block_gemm

        self._check_native(activation, weight)
        if activation.logical_shape[-1] != weight.logical_shape[-1]:
            raise ValueError("INT8 activation and weight logical K must agree")
        return int8_block_gemm(
            activation.data,
            activation.scales,
            weight.data,
            weight.scales,
            bias=bias,
            output_dtype=output_dtype,
        )

    def probe(self, device: torch.device | str | int) -> BackendCapability:
        from rlinf.quantization.backends.capabilities import probe_cuda_int8_capability

        return probe_cuda_int8_capability(device)
