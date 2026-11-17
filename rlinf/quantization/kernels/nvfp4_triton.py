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

"""Shared block16 kernels for fake and packed NVFP4, loaded on CUDA paths."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _nvfp4_amax_kernel(X, PARTIALS, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    tl.store(PARTIALS + tl.program_id(0), tl.max(tl.abs(values), 0))


@triton.jit
def _nvfp4_global_scale_kernel(X, GLOBAL, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    amax = tl.max(tl.abs(values), 0)
    # Match PyTorch CUDA scalar division, which multiplies by a rounded reciprocal.
    scale = tl.where(amax > 0, amax * (1.0 / 2688.0), 1.0)
    tl.store(GLOBAL, scale)


def canonicalize_nvfp4_cuda(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse absolute maximum and global scale after the canonical BF16 cast."""
    canonical = values.detach().to(torch.bfloat16).contiguous()
    maximums = canonical
    while maximums.numel() > 4096:
        count = triton.cdiv(maximums.numel(), 4096)
        partials = torch.empty(count, dtype=torch.float32, device=values.device)
        _nvfp4_amax_kernel[(count,)](maximums, partials, maximums.numel(), 4096)
        maximums = partials
    global_scale = torch.empty((), dtype=torch.float32, device=values.device)
    _nvfp4_global_scale_kernel[(1,)](
        maximums,
        global_scale,
        maximums.numel(),
        triton.next_power_of_2(maximums.numel()),
    )
    return canonical, global_scale


@triton.jit
def _quantize_nvfp4_tile(x, global_scale):
    """Return codes, block scales, and QDQ values for one 16x64 tile."""
    blocks = tl.reshape(x, (16, 4, 16))
    block_max = tl.max(tl.abs(blocks), 2, keep_dims=True)
    sf8 = tl.minimum(block_max / (6.0 * global_scale), 448.0).to(tl.float8e4nv)
    denominator = sf8.to(tl.float32) * global_scale
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    magnitude = tl.abs(blocks) / safe_denominator
    code = tl.where(
        magnitude <= 0.25,
        0,
        tl.where(
            magnitude < 0.75,
            1,
            tl.where(
                magnitude <= 1.25,
                2,
                tl.where(
                    magnitude < 1.75,
                    3,
                    tl.where(
                        magnitude <= 2.5,
                        4,
                        tl.where(magnitude < 3.5, 5, tl.where(magnitude <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    )
    q = tl.where(
        code < 4,
        code * 0.5,
        tl.where(
            code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))
        ),
    )
    sign = (blocks.to(tl.int32, bitcast=True) >> 31) & 8
    code = tl.where(denominator > 0, code | sign, 0).to(tl.uint8)
    dequantized = tl.where(sign != 0, -(q * denominator), q * denominator)
    dequantized = tl.where(denominator > 0, dequantized, 0.0)
    return (
        tl.reshape(code, (16, 64)),
        tl.reshape(sf8, (16, 4)),
        tl.reshape(dequantized, (16, 64)),
    )


@triton.jit
def _fake_quantize_nvfp4_kernel(
    X,
    Y,
    GLOBAL,
    ROWS: tl.constexpr,
    K: tl.constexpr,
):
    """Quantize one tile and write its BF16 QDQ values."""
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    valid = (rows[:, None] < ROWS) & (cols[None, :] < K)
    x = tl.load(X + rows[:, None] * K + cols[None, :], valid, 0).to(tl.float32)
    _, _, dequantized = _quantize_nvfp4_tile(x, tl.load(GLOBAL))
    tl.store(Y + rows[:, None] * K + cols[None, :], dequantized, valid)


@triton.jit
def _quantize_and_pack_nvfp4_kernel(
    X,
    Y,
    SF,
    GLOBAL,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    OUT_K: tl.constexpr,
    SF_COLS: tl.constexpr,
):
    """Quantize one tile and write packed codes and swizzled scales."""
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    valid = (rows[:, None] < ROWS) & (cols[None, :] < K)
    x = tl.load(X + rows[:, None] * K + cols[None, :], valid, 0).to(tl.float32)
    code, sf, _ = _quantize_nvfp4_tile(x, tl.load(GLOBAL))
    pairs = tl.reshape(code.to(tl.int32), (16, 32, 2))
    packed = tl.sum(pairs * tl.where(tl.arange(0, 2) == 0, 1, 16)[None, None, :], 2)
    byte_cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    tl.store(
        Y + rows[:, None] * (OUT_K // 2) + byte_cols[None, :],
        packed,
        (rows[:, None] < OUT_ROWS) & (byte_cols[None, :] < OUT_K // 2),
    )
    block_cols = tl.program_id(1) * 4 + tl.arange(0, 4)
    offsets = (
        block_cols[None, :] % 4
        + (block_cols[None, :] // 4) * 512
        + (rows[:, None] % 32) * 16
        + ((rows[:, None] % 128) // 32) * 4
        + (rows[:, None] // 128) * (128 * SF_COLS)
    )
    tl.store(SF + offsets, sf.to(tl.uint8, bitcast=True))


def fake_quantize_nvfp4_cuda(
    canonical: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Write BF16 dequantized values without materializing full codes or FP32 scales."""
    rows, k = canonical.shape
    output = torch.empty_like(canonical, dtype=torch.bfloat16)
    with torch.cuda.device(canonical.device):
        _fake_quantize_nvfp4_kernel[(triton.cdiv(rows, 16), triton.cdiv(k, 64))](
            canonical,
            output,
            global_scale,
            rows,
            k,
            num_warps=4,
        )
    return output


def quantize_and_pack_nvfp4_cuda(
    canonical: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize into packed nibbles and scales padded to 128 physical rows."""
    rows, k = canonical.shape
    padded_rows, padded_k = triton.cdiv(rows, 64) * 64, triton.cdiv(k, 64) * 64
    sf_rows = triton.cdiv(padded_rows, 128) * 128
    output = torch.empty(
        (padded_rows, padded_k // 2), dtype=torch.uint8, device=canonical.device
    )
    scales = torch.empty(
        (sf_rows, padded_k // 16), dtype=torch.uint8, device=canonical.device
    )
    with torch.cuda.device(canonical.device):
        _quantize_and_pack_nvfp4_kernel[(triton.cdiv(sf_rows, 16), triton.cdiv(k, 64))](
            canonical,
            output,
            scales,
            global_scale,
            rows,
            k,
            padded_rows,
            padded_k,
            padded_k // 16,
            num_warps=4,
        )
    return output, scales


@triton.jit
def _nvfp4_epilogue_kernel(
    X,
    BIAS,
    Y,
    N: tl.constexpr,
    ELEMENTS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // N, offsets % N
    value = tl.load(
        X + rows * ROW_STRIDE + cols * COL_STRIDE,
        offsets < ELEMENTS,
        0,
    ).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + cols * BIAS_STRIDE, offsets < ELEMENTS, 0).to(tl.float32)
        value = value + bias
    tl.store(Y + offsets, value, offsets < ELEMENTS)


def nvfp4_gemm_epilogue_cuda(
    result: torch.Tensor, bias: torch.Tensor | None, output_dtype: torch.dtype
) -> torch.Tensor:
    """Cast rounded BF16 output directly while preserving FP32 bias addition."""
    output = torch.empty(result.shape, dtype=output_dtype, device=result.device)
    with torch.cuda.device(result.device):
        _nvfp4_epilogue_kernel[(triton.cdiv(result.numel(), 1024),)](
            result,
            bias,
            output,
            result.shape[1],
            result.numel(),
            result.stride(0),
            result.stride(1),
            0 if bias is None else bias.stride(0),
            bias is not None,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
