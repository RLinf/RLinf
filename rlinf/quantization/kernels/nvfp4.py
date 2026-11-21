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

"""Native FP4 GEMM and epilogue through FlashInfer 0.7.1."""

from __future__ import annotations

import torch

from rlinf.quantization.backends.base import _FLOAT_DTYPES, QuantizedTensor


def _validate_native_tensor(value: QuantizedTensor) -> None:
    """Check actual tensor storage once at the native GEMM boundary."""
    if value.format != "nvfp4" or value.storage != "native":
        raise ValueError("NVFP4 GEMM requires native NVFP4 inputs")
    rows, k = value.logical_shape
    padded_rows, padded_k = value.padded_shape
    if (
        rows < 1
        or k < 1
        or padded_rows < rows
        or padded_k < k
        or padded_rows % 64
        or padded_k % 64
    ):
        raise ValueError("NVFP4 logical and padded matrix shapes conflict")
    tensors = (value.data, value.scales, value.global_scale)
    if value.data.device.type != "cuda" or any(
        t.device != value.data.device for t in tensors
    ):
        raise ValueError("NVFP4 native tensors must share a CUDA device")
    if value.data.dtype != torch.uint8 or value.data.shape != (
        padded_rows,
        padded_k // 2,
    ):
        raise ValueError("NVFP4 packed data must be uint8 [padded rows, padded K/2]")
    scale_rows = ((padded_rows + 127) // 128) * 128
    if value.scales.dtype != torch.uint8 or value.scales.shape != (
        scale_rows,
        padded_k // 16,
    ):
        raise ValueError("NVFP4 scale shape must retain physical rows padded to 128")
    if value.global_scale.dtype != torch.float32 or value.global_scale.shape != ():
        raise ValueError("NVFP4 global scale must be scalar FP32")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("NVFP4 native tensors must be contiguous")


def _apply_epilogue(
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Write the final CUDA inference output with FP32 bias addition."""
    if bias is None and result.dtype == output_dtype and result.is_contiguous():
        return result
    from rlinf.quantization.kernels.nvfp4_triton import nvfp4_gemm_epilogue_cuda

    return nvfp4_gemm_epilogue_cuda(result, bias, output_dtype)


@torch.no_grad()
def nvfp4_gemm(
    activation: QuantizedTensor,
    weight: QuantizedTensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run native FP4 inference with BF16 output rounding and FP32 bias.

    Args:
        activation: Native logical [M, K] activation matrix.
        weight: Native logical [N, K] weight matrix.
        bias: Optional floating-point [N] bias on the same device.
        output_dtype: Final FP32, BF16, or FP16 output dtype.

    Returns:
        Contiguous output cropped to logical [M, N].
    """
    _validate_native_tensor(activation)
    _validate_native_tensor(weight)
    if (
        activation.logical_shape[1] != weight.logical_shape[1]
        or activation.padded_shape[1] != weight.padded_shape[1]
    ):
        raise ValueError("NVFP4 activation and weight logical K must agree")
    device = activation.data.device
    if device != weight.data.device:
        raise ValueError("NVFP4 GEMM inputs must share the same CUDA device")
    if output_dtype not in _FLOAT_DTYPES:
        raise ValueError("NVFP4 GEMM output dtype must be FP32, BF16, or FP16")
    m, n = activation.logical_shape[0], weight.logical_shape[0]
    if bias is not None and (
        bias.device != device or bias.shape != (n,) or bias.dtype not in _FLOAT_DTYPES
    ):
        raise ValueError("NVFP4 GEMM bias must be floating [N] on the output device")
    alpha = activation.global_scale * weight.global_scale
    # SM100 CUTLASS requires N to retain 32-element alignment.
    gemm_n = ((n + 31) // 32) * 32

    import flashinfer

    with torch.cuda.device(device):
        result = flashinfer.mm_fp4(
            activation.data[:m],
            weight.data[:gemm_n].T,
            activation.scales,
            weight.scales.T,
            alpha=alpha,
            out_dtype=torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )[:m, :n]
    return _apply_epilogue(result, bias, output_dtype)
