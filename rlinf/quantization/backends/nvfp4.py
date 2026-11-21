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

"""NVFP4 block16 numerics with CUDA validity checks at training and inference boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.quantization.backends.base import QuantizationBackend, QuantizedTensor, Role
from rlinf.quantization.config import BackendCapability

_FLOAT_DTYPES = frozenset({torch.float32, torch.bfloat16, torch.float16})
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _canonicalize_nvfp4(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached BF16 input and a logical-matrix FP32 dequantization scale."""
    if values.ndim < 1 or values.numel() == 0:
        raise ValueError("NVFP4 requires a nonempty tensor with a K dimension")
    if values.dtype not in _FLOAT_DTYPES:
        raise ValueError("NVFP4 input dtype must be FP32, BF16, or FP16")
    canonical = values.detach().to(torch.bfloat16).contiguous()
    if values.device.type != "cuda":
        if not torch.isfinite(canonical).all():
            raise ValueError(
                "NVFP4 input must remain finite in the canonical BF16 dtype"
            )
        if (
            values.dtype == torch.float16
            and not torch.isfinite(canonical.to(torch.float16)).all()
        ):
            raise ValueError(
                "NVFP4 FP16 input overflows when canonical BF16 values are restored"
            )
    amax = canonical.abs().amax().float()
    global_scale = torch.where(amax > 0, amax / 2688.0, torch.ones_like(amax))
    if values.device.type != "cuda" and (
        global_scale <= 0 or not torch.isfinite(global_scale.reciprocal())
    ):
        raise ValueError(
            "NVFP4 global scale and its reciprocal must be positive and finite"
        )
    return canonical, global_scale


@torch.no_grad()
def quantize_nvfp4_blocks(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the last dimension into nibble codes, FP8 block scales, and a global scale.

    Args:
        values: Nonempty floating-point tensor, cast to BF16 before computing
            dynamic scales. CUDA callers ensure finite values and representable
            scales without synchronizing every invocation.

    Returns:
        Same-shaped uint8 codes, [flattened rows, ceil(K/16)] E4M3FN scales,
        and a scalar FP32 global dequantization scale. Blocks whose local
        scales underflow to zero dequantize to zero.
    """
    canonical, global_scale = _canonicalize_nvfp4(values)
    k = values.shape[-1]
    blocks = (k + 15) // 16
    grouped = F.pad(canonical.reshape(-1, k).float(), (0, blocks * 16 - k)).reshape(
        -1, blocks, 16
    )
    block_amax = grouped.abs().amax(dim=-1)
    scales = (block_amax / (6.0 * global_scale)).to(torch.float8_e4m3fn)
    denominator = global_scale * scales.float()
    encode_scale = torch.where(denominator > 0, denominator, 1.0).reciprocal()
    if values.device.type != "cuda" and not torch.isfinite(encode_scale).all():
        raise ValueError("NVFP4 local encode scale must be finite")
    # Match the reciprocal multiplication used by the execution kernels.
    normalized = grouped * encode_scale.unsqueeze(-1)
    normalized = torch.where((denominator > 0).unsqueeze(-1), normalized, 0.0)
    midpoints = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=values.device)
    magnitude = normalized.abs().contiguous()
    codes = torch.bucketize(magnitude, midpoints)
    lower_is_odd = codes.remainder(2) == 1
    at_midpoint = magnitude == midpoints[codes.clamp_max(6)]
    codes = (codes + (lower_is_odd & at_midpoint)).to(torch.uint8)
    codes |= torch.signbit(normalized).to(torch.uint8) * 8
    codes = codes.reshape(-1, blocks * 16)[:, :k].reshape(values.shape)
    return codes, scales, global_scale


def dequantize_nvfp4_blocks(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    logical_k: int,
) -> torch.Tensor:
    """Dequantize logical or padded unpacked codes to FP32 and crop the K tail.

    Args:
        codes: uint8 E2M1 codes stored in the low four bits of each element.
        scales: [flattened rows, ceil(code K/16)] E4M3FN block scales.
        global_scale: Positive finite scalar FP32 dequantization scale on the
            same device.
        logical_k: Positive logical width no larger than the code width.

    Returns:
        FP32 tensor retaining the leading code dimensions with logical_k as
        the last dimension.
    """
    if codes.ndim < 1 or codes.numel() == 0 or codes.dtype != torch.uint8:
        raise ValueError("NVFP4 codes must be a nonempty uint8 tensor")
    k = codes.shape[-1]
    if (
        isinstance(logical_k, bool)
        or not isinstance(logical_k, int)
        or not 0 < logical_k <= k
    ):
        raise ValueError(
            "NVFP4 logical_k must be a positive integer no larger than code K"
        )
    if scales.shape != (codes.numel() // k, (k + 15) // 16):
        raise ValueError("NVFP4 block scale shape conflicts with code shape")
    if scales.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 block scale dtype must be E4M3FN")
    if global_scale.dtype != torch.float32 or global_scale.shape != ():
        raise ValueError("NVFP4 global scale must be a scalar FP32 tensor")
    if scales.device != codes.device or global_scale.device != codes.device:
        raise ValueError("NVFP4 codes and scales must be on the same device")
    if codes.device.type != "cuda":
        if (codes > 15).any():
            raise ValueError("NVFP4 codes must be in [0, 15]")
        if not torch.isfinite(scales.float()).all() or (scales.float() < 0).any():
            raise ValueError("NVFP4 block scales must be nonnegative and finite")
        if global_scale <= 0 or not torch.isfinite(global_scale):
            raise ValueError("NVFP4 global scale must be positive and finite")
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    flat = codes.reshape(-1, k)[:, :logical_k].long()
    magnitude = table[flat & 7]
    signed = torch.where((flat & 8) != 0, -magnitude, magnitude)
    expanded_scales = (
        scales.detach().float().repeat_interleave(16, dim=-1)[:, :logical_k]
    )
    return (signed * (expanded_scales * global_scale.detach())).reshape(
        *codes.shape[:-1], logical_k
    )


class _Passthrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        return quantized

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


def fake_quantize_nvfp4_blocks(values: torch.Tensor) -> torch.Tensor:
    """Use BF16 dequantized forwards and identity input gradients.

    Args:
        values: Input tensor retaining floating-point master-parameter identity.

    Returns:
        Differentiable tensor with the input shape and dtype. Scales receive
        no gradients.
    """
    if values.device.type == "cuda":
        canonical, global_scale = _canonicalize_nvfp4(values)
        from rlinf.quantization.kernels.nvfp4_triton import fake_quantize_nvfp4_cuda

        quantized = fake_quantize_nvfp4_cuda(
            canonical.reshape(-1, values.shape[-1]), global_scale
        ).reshape(values.shape)
    else:
        codes, scales, global_scale = quantize_nvfp4_blocks(values)
        quantized = dequantize_nvfp4_blocks(
            codes, scales, global_scale, values.shape[-1]
        )
    restored = quantized.to(torch.bfloat16).to(values.dtype)
    if values.device.type != "cuda" and not torch.isfinite(restored).all():
        raise ValueError(f"NVFP4 restored {values.dtype} output must be finite")
    return _Passthrough.apply(values, restored)


class Nvfp4Backend(QuantizationBackend):
    """NVFP4 block16 with independent projection-global scales and native CUTLASS GEMM."""

    quantization_format = "nvfp4"

    def __init__(self, *, block_size: int) -> None:
        if type(block_size) is not int or block_size != 16:
            raise ValueError("NVFP4 block_size must be the integer 16")
        super().__init__(block_size=block_size)

    @torch.no_grad()
    def quantize(
        self, values: torch.Tensor, *, role: Role, native: bool = False
    ) -> QuantizedTensor:
        if native:
            return self._quantize_native(values, role)
        else:
            return self._quantize_logical(values)

    def _quantize_logical(self, values: torch.Tensor) -> QuantizedTensor:
        data, scales, global_scale = quantize_nvfp4_blocks(values)
        return QuantizedTensor(
            data,
            scales,
            tuple(values.shape),
            self.quantization_format,
            global_scale=global_scale,
        )

    def _quantize_native(self, values: torch.Tensor, role: Role) -> QuantizedTensor:
        if values.device.type != "cuda":
            raise ValueError("NVFP4 packing requires a CUDA tensor")
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("NVFP4 packing requires a nonempty 2D matrix")
        if values.dtype not in _FLOAT_DTYPES:
            raise ValueError("NVFP4 input dtype must be FP32, BF16, or FP16")
        from rlinf.quantization.kernels.nvfp4_triton import (
            canonicalize_nvfp4_cuda,
            quantize_and_pack_nvfp4_cuda,
        )

        with torch.cuda.device(values.device):
            canonical, global_scale = canonicalize_nvfp4_cuda(values)
            if role == "weight":
                self._validate_native_weights(canonical, global_scale)
            data, scales = quantize_and_pack_nvfp4_cuda(canonical, global_scale)
        rows, k = values.shape
        return QuantizedTensor(
            data,
            scales,
            (rows, k),
            self.quantization_format,
            storage="native",
            global_scale=global_scale,
            padded_shape=(((rows + 63) // 64) * 64, ((k + 63) // 64) * 64),
        )

    @staticmethod
    def _validate_native_weights(
        canonical: torch.Tensor, global_scale: torch.Tensor
    ) -> None:
        if not torch.isfinite(canonical).all():
            raise ValueError(
                "native NVFP4 weights must remain finite after BF16 conversion"
            )
        if (
            not torch.isfinite(global_scale)
            or global_scale <= 0
            or not torch.isfinite(global_scale.reciprocal())
        ):
            raise ValueError(
                "native NVFP4 weights require a finite positive invertible global scale"
            )

    def dequantize(self, value: QuantizedTensor) -> torch.Tensor:
        if value.storage == "logical":
            return self._dequantize_logical(value)
        elif value.storage == "native":
            return self._dequantize_native(value)
        raise ValueError(f"NVFP4 storage {value.storage!r} is not supported")

    def _dequantize_logical(self, value: QuantizedTensor) -> torch.Tensor:
        return dequantize_nvfp4_blocks(
            value.data, value.scales, value.global_scale, value.logical_shape[-1]
        )

    def _dequantize_native(self, value: QuantizedTensor) -> torch.Tensor:
        codes = self._unpack_native_codes(value)
        scales = self._unswizzle_native_scales(value)
        return dequantize_nvfp4_blocks(
            codes, scales, value.global_scale, value.logical_shape[-1]
        )

    @staticmethod
    def _unpack_native_codes(value: QuantizedTensor) -> torch.Tensor:
        rows, k = value.logical_shape
        # Even columns occupy the low nibble.
        codes = torch.stack((value.data & 15, value.data >> 4), dim=-1)
        return codes.reshape(value.padded_shape)[:rows, :k]

    @staticmethod
    def _unswizzle_native_scales(value: QuantizedTensor) -> torch.Tensor:
        rows, k = value.logical_shape
        padded_k = value.padded_shape[1]
        row = torch.arange(rows, device=value.data.device)[:, None]
        block = torch.arange((k + 15) // 16, device=value.data.device)[None, :]
        offsets = (
            block % 4
            + (block // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + (row // 128) * (128 * (padded_k // 16))
        )
        return value.scales.reshape(-1)[offsets].view(torch.float8_e4m3fn)

    def gemm(
        self,
        activation: QuantizedTensor,
        weight: QuantizedTensor,
        *,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        from rlinf.quantization.kernels.nvfp4 import nvfp4_gemm

        return nvfp4_gemm(activation, weight, bias=bias, output_dtype=output_dtype)

    def fake_quantize(self, values: torch.Tensor, *, role: Role) -> torch.Tensor:
        return fake_quantize_nvfp4_blocks(values)

    def fake_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        weight_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # This recipe fixes GEMM precision independently of model-side float casts.
        activation = self.fake_quantize(input, role="activation").bfloat16()
        quantized_weight = self.fake_quantize(weight, role="weight").bfloat16()
        with torch.autocast(device_type=input.device.type, enabled=False):
            output = F.linear(activation, quantized_weight, None).float()
        if bias is not None:
            output = output + bias.float()
        return output.to(input.dtype)

    def probe(self, device: torch.device | str | int) -> BackendCapability:
        from rlinf.quantization.backends.capabilities import probe_cuda_nvfp4_capability

        return probe_cuda_nvfp4_capability(device)
