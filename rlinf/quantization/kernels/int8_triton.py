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

"""CUDA blockwise INT8 quantization and fused GEMM kernels, loaded lazily."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def quantize_rows_kernel(
    values,
    quantized,
    scales,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    PADDED_K: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    block = tl.program_id(1)
    columns = block * BLOCK_K + tl.arange(0, BLOCK_K)
    data = tl.load(
        values + rows[:, None] * K + columns[None, :],
        mask=(rows[:, None] < ROWS) & (columns[None, :] < K),
        other=0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(data), axis=1) * (1.0 / 127.0), 1e-12)
    normalized = tl.div_rn(data, scale[:, None])
    rounded = libdevice.nearbyint(normalized)
    integers = tl.minimum(tl.maximum(rounded, -128), 127).to(tl.int8)
    tl.store(
        quantized + rows[:, None] * PADDED_K + columns[None, :],
        integers,
        mask=rows[:, None] < ROWS,
    )
    tl.store(scales + rows * BLOCKS + block, scale, mask=rows < ROWS)


@triton.jit
def block_int8_gemm_kernel(
    activation_q,
    activation_scales,
    weight_q,
    weight_scales,
    bias,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    PADDED_K: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    total = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for block in range(BLOCKS):
        k = block * BLOCK_K + inner
        left = tl.load(
            activation_q + rows[:, None] * PADDED_K + k[None, :],
            mask=rows[:, None] < M,
            other=0,
        )
        right = tl.load(
            weight_q + columns[None, :] * PADDED_K + k[:, None],
            mask=columns[None, :] < N,
            other=0,
        )
        integer_sum = tl.dot(left, right, out_dtype=tl.int32)
        scale_a = tl.load(
            activation_scales + rows * BLOCKS + block, mask=rows < M, other=0
        )
        scale_w = tl.load(
            weight_scales + columns * BLOCKS + block, mask=columns < N, other=0
        )
        block_scale = scale_a[:, None] * scale_w[None, :]
        total += integer_sum.to(tl.float32) * block_scale
    if HAS_BIAS:
        total += tl.load(bias + columns, mask=columns < N, other=0).to(tl.float32)[
            None, :
        ]
    tl.store(
        output + rows[:, None] * N + columns[None, :],
        total,
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )
