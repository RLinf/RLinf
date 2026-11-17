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

"""Dynamic activation quantization and single-launch W8A8 K-block GEMM."""

from __future__ import annotations

import torch

_BLOCK_SIZES = frozenset({32, 64, 128, 256})
_FLOAT_DTYPES = frozenset({torch.float32, torch.bfloat16, torch.float16})


def quantize_int8_cuda_blocks(
    values: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize input rows in K blocks into padded INT8 data and FP32 scales.

    Args:
        values: Contiguous CUDA floating-point matrix.
        block_size: K-block width with supported Tensor Core alignment.

    Returns:
        INT8 [rows, padded_k] data and FP32 [rows, blocks] scales.
    """
    if values.device.type != "cuda":
        raise ValueError("INT8 Triton quantization requires a CUDA tensor")
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("INT8 Triton quantization requires a nonempty 2D matrix")
    if values.dtype not in _FLOAT_DTYPES:
        raise ValueError("INT8 Triton quantization requires a floating-point matrix")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size not in _BLOCK_SIZES
    ):
        raise ValueError(f"unsupported INT8 Triton block size: {block_size!r}")
    from rlinf.quantization.kernels.int8_triton import quantize_rows_kernel

    values = values.contiguous()
    rows, k = values.shape
    blocks = (k + block_size - 1) // block_size
    padded_k = blocks * block_size
    quantized = torch.empty((rows, padded_k), dtype=torch.int8, device=values.device)
    scales = torch.empty((rows, blocks), dtype=torch.float32, device=values.device)
    quantize_rows_kernel[((rows + 3) // 4, blocks)](
        values,
        quantized,
        scales,
        rows,
        k,
        padded_k,
        blocks,
        block_size,
        4,
        num_warps=4,
    )
    return quantized, scales


def int8_block_gemm(
    activation_q: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Accumulate INT32 dot products and per-K-block FP32 scales in one GEMM launch.

    Args:
        activation_q: Zero-padded INT8 [M, padded_k] activations.
        activation_scales: FP32 [M, blocks] activation scales.
        weight_q: Zero-padded INT8 [N, padded_k] weights.
        weight_scales: FP32 [N, blocks] weight scales.
        bias: Optional floating-point [N] bias added after FP32 accumulation.
        output_dtype: Floating-point dtype used for the final store.

    Returns:
        Quantized GEMM output with shape [M, N].
    """
    tensors = (activation_q, activation_scales, weight_q, weight_scales)
    if activation_q.device.type != "cuda" or any(
        tensor.device != activation_q.device for tensor in tensors
    ):
        raise ValueError("INT8 Triton GEMM requires tensors on the same CUDA device")
    if any(tensor.ndim != 2 or min(tensor.shape) < 1 for tensor in tensors):
        raise ValueError("INT8 Triton GEMM requires nonempty 2D matrices")
    if activation_q.dtype != torch.int8 or weight_q.dtype != torch.int8:
        raise ValueError("INT8 Triton GEMM requires INT8 activation and weight")
    if any(
        scale.dtype != torch.float32 for scale in (activation_scales, weight_scales)
    ):
        raise ValueError("INT8 Triton GEMM requires FP32 scales")
    m, padded_k = activation_q.shape
    n = weight_q.shape[0]
    blocks = activation_scales.shape[1]
    if (
        weight_q.shape[1] != padded_k
        or activation_scales.shape[0] != m
        or weight_scales.shape != (n, blocks)
        or padded_k % blocks
        or padded_k // blocks not in _BLOCK_SIZES
    ):
        raise ValueError("INT8 Triton GEMM matrix and block scale shapes conflict")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("INT8 Triton GEMM requires contiguous matrices and scales")
    if output_dtype not in _FLOAT_DTYPES:
        raise ValueError("INT8 Triton GEMM output must be floating-point")
    if bias is not None and (
        bias.device != activation_q.device
        or bias.shape != (n,)
        or bias.dtype not in _FLOAT_DTYPES
        or not bias.is_contiguous()
    ):
        raise ValueError("INT8 Triton GEMM bias must be a contiguous CUDA [N] vector")
    from rlinf.quantization.kernels.int8_triton import block_int8_gemm_kernel

    output = torch.empty((m, n), dtype=output_dtype, device=activation_q.device)
    tile_m, tile_n = (16 if m < 32 else 32), 64
    block_int8_gemm_kernel[((m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n)](
        activation_q,
        activation_scales,
        weight_q,
        weight_scales,
        output if bias is None else bias,
        output,
        m,
        n,
        padded_k,
        blocks,
        padded_k // blocks,
        tile_m,
        tile_n,
        bias is not None,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return output
