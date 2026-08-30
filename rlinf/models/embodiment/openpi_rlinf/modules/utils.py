# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComputeDtypeLinear(nn.Linear):
    """``nn.Linear`` that matches input dtype to the live compute weight.

    FSDP mixed precision rewrites ``weight.dtype`` at the start of this
    module's forward (bf16 compute, fp32 master). The parent still sees the
    storage dtype, so casting before the call is too early. Aligning here
    keeps ``precision: fp32`` + ``param_dtype: bf16`` on the original
    mixed-precision path, and stays full fp32 when mixed precision is off.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if input.dtype != weight.dtype:
            input = input.to(dtype=weight.dtype)
        return F.linear(input, weight, self.bias)


class Float32Conv2d(nn.Conv2d):
    """Patch-embed conv that always runs in float32 (JAX SigLIP stem).

    The op must live inside this module so FSDP can unflatten ``weight``
    before we touch it. ``F.conv2d(self.stem.weight)`` from the parent sees
    a 1-D ``FlatParameter`` under ``full_shard`` / ``shard_grad_op``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input.float(),
            self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


@torch.compile
def gelu_glu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """Fused GELU-GLU activation: ``gelu(gate_input) * value_input``."""
    return F.gelu(gate_input) * value_input


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    # if dtype_str == "mp_bfloat16":
    #     assert False
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]
