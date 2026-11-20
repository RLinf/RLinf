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

import math

import torch
import triton
import triton.language as tl
from torch import nn
from torch.library import triton_op, wrap_triton


@triton.jit(do_not_specialize=["rows"])
def _linear_kernel(
    inputs,
    weight,
    bias,
    output,
    rows,
    cols: tl.constexpr,
    depth: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
    block_depth: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.program_id(1) * block_cols + tl.arange(0, block_cols)
    k = tl.arange(0, block_depth)
    total = tl.full((block_rows, block_cols), 0, tl.float32)

    # Row count changes only the grid and masks, keeping the K reduction fixed.
    for tile in range(tl.cdiv(depth, block_depth)):
        offset = tile * block_depth + k
        left = tl.load(
            inputs + row[:, None] * depth + offset[None, :],
            (row[:, None] < rows) & (offset[None, :] < depth),
            other=0,
        )
        right = tl.load(
            weight + col[None, :] * depth + offset[:, None],
            (col[None, :] < cols) & (offset[:, None] < depth),
            other=0,
        )
        total = tl.dot(left, right, total, input_precision="tf32x3")

    if bias is not None:
        total += tl.load(bias + col, col < cols, other=0).to(tl.float32)[None, :]
    tl.store(
        output + row[:, None] * cols + col[None, :],
        total,
        (row[:, None] < rows) & (col[None, :] < cols),
    )


@triton_op("rlinf::batch_invariant_linear", mutates_args={})
def _linear(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    rows = math.prod(inputs.shape[:-1])
    cols, depth = weight.shape
    output = inputs.new_empty((*inputs.shape[:-1], cols))
    if rows == 0 or cols == 0:
        return output

    wrap_triton(_linear_kernel)[(triton.cdiv(rows, 32), triton.cdiv(cols, 64))](
        inputs.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else None,
        output,
        rows,
        cols,
        depth,
        32,
        64,
        32,
    )
    return output


def _setup_context(ctx, inputs, output):
    inputs, weight, _ = inputs
    ctx.save_for_backward(inputs, weight)


def _backward(ctx, grad):
    inputs, weight = ctx.saved_tensors
    need_input, need_weight, need_bias = ctx.needs_input_grad
    rows = math.prod(inputs.shape[:-1])
    grad = grad.reshape(rows, weight.shape[0])
    with torch.autocast("cuda", enabled=False):
        grad_input = (grad @ weight).reshape_as(inputs) if need_input else None
        grad_weight = (
            grad.mT @ inputs.reshape(rows, weight.shape[1]) if need_weight else None
        )
        grad_bias = grad.sum(dim=0) if need_bias else None
    return grad_input, grad_weight, grad_bias


_linear.register_autograd(_backward, setup_context=_setup_context)


def triton_linear(
    inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply a linear projection with a fixed forward reduction on NVIDIA GPUs."""
    if (
        inputs.device.type != "cuda"
        or inputs.device != weight.device
        or (bias is not None and bias.device != inputs.device)
    ):
        raise ValueError(
            "Triton linear requires inputs, weight, and bias on the same CUDA device."
        )
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"Expected {weight.shape[1]} input features, got {inputs.shape[-1]}."
        )
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError(f"Expected bias shape {weight.shape[:1]}, got {bias.shape}.")
    supported_dtypes = (torch.float32, torch.bfloat16, torch.float16)
    if inputs.dtype not in supported_dtypes or weight.dtype not in supported_dtypes:
        raise ValueError(
            "Triton linear requires FP32, BF16, or FP16 inputs and weights."
        )
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        inputs, weight = inputs.to(dtype), weight.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)
    elif inputs.dtype != weight.dtype or (
        bias is not None and bias.dtype != weight.dtype
    ):
        raise ValueError(
            "Triton linear requires matching input, weight, and bias dtypes."
        )
    return _linear(inputs, weight, bias)


class TritonLinear(nn.Linear):
    """Linear layer with a batch-invariant forward pass on NVIDIA GPUs."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project inputs while preserving the Linear parameter layout."""
        return triton_linear(inputs, self.weight, self.bias)
