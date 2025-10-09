# Copyright 2025 The RLinf Authors.
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

# use a fp8Tensor class to package fp8 operations

from typing import List, Literal, Tuple, Union

import torch


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


inductor_optimizer = torch._dynamo.optimize("inductor")


def per_block_cast_to_fp8_cpu(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


# this function don't support cpu tensor
per_block_cast_to_fp8_gpu = inductor_optimizer(per_block_cast_to_fp8_cpu)


def per_block_cast_to_bf16_cpu(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.empty(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    scale_expanded = scale.view(x_view.size(0), 1, x_view.size(2), 1)
    x_bf16 = x_view.to(torch.bfloat16) * scale_expanded * (448.0 / 448.0)
    return x_bf16.view_as(x_padded)[:m, :n].contiguous().to(torch.bfloat16)


# this function don't support cpu tensor
per_block_cast_to_bf16_gpu = inductor_optimizer(per_block_cast_to_bf16_cpu)


def per_block_cast_to_fp8(
    x: torch.Tensor, device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = x.device
    if torch.device(device).type == "cpu":
        return per_block_cast_to_fp8_cpu(x)
    else:
        return per_block_cast_to_fp8_gpu(x)


def per_block_cast_to_bf16(
    x: torch.Tensor, scale: torch.Tensor, device: torch.device = None
) -> torch.Tensor:
    if device is None:
        device = x.device
    if torch.device(device).type == "cpu":
        return per_block_cast_to_bf16_cpu(x, scale)
    else:
        return per_block_cast_to_bf16_gpu(x, scale)


class fp8Tensor:
    def __init__(self, tensor, scale_inv):
        self.tensor = tensor
        self.scale_inv = scale_inv

    def to_bf16(self) -> torch.Tensor:
        # return weight_dequant(self.tensor, self.scale_inv)
        return per_block_cast_to_bf16(self.tensor, self.scale_inv)

    @staticmethod
    def from_bf16(tensor: torch.Tensor) -> torch.Tensor:
        return fp8Tensor(*per_block_cast_to_fp8(tensor))

    @staticmethod
    def group(
        tensors: List["fp8Tensor"], fc1orfc2: Literal["fc1", "fc2"]
    ) -> "fp8Tensor":
        # TODO:
        raise NotImplementedError()

    @staticmethod
    def cat(tensors: list["fp8Tensor"], dim=0) -> "fp8Tensor":
        for i in tensors:
            m, n = i.tensor.shape
            assert m == ceil_div(m, 128) * 128 and n == ceil_div(n, 128) * 128, (
                "need fp8Tensor shape is blocked by 128"
            )
        tensor = torch.cat([i.tensor for i in tensors], dim=dim)
        scale_inv = torch.cat([i.scale_inv for i in tensors], dim=dim)
        return fp8Tensor(tensor, scale_inv)


def dict_push(
    the_dict: dict, name, tensor: Union[torch.Tensor, fp8Tensor], check_unique=True
):
    if check_unique:
        assert name not in the_dict, f"{name} already in the_dict"
    if isinstance(tensor, fp8Tensor):
        name_scale = f"{name}_scale_inv"
        if check_unique:
            assert name_scale not in the_dict, f"{name_scale} already in the_dict"
        the_dict[name_scale] = tensor.scale_inv
        the_dict[name] = tensor.tensor
    elif isinstance(tensor, torch.Tensor):
        the_dict[name] = tensor
    else:
        assert False
