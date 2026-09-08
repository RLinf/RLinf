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

"""Run Qwen-VL's patch embedding as a matmul, which ROCm survives.

Its Conv3d segfaults the process on ROCm 6.4, on transformers 4.57 and 5.16 alike.
"""

import types

import torch
import torch.nn.functional as F

from rlinf.utils.logging import get_logger

_PATCH_EMBED_CLASSES = ("Qwen3VLVisionPatchEmbed", "Qwen2_5_VisionPatchEmbed")


def _linear_patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Exact, not an approximation: the caller has already reshaped the input to one
    # block per patch, and kernel_size == stride == a whole block, so each output
    # element is one dot product of a flattened block with a flattened filter.
    weight = self.proj.weight.reshape(self.proj.out_channels, -1)
    hidden_states = hidden_states.reshape(-1, weight.shape[1]).to(weight.dtype)
    return F.linear(hidden_states, weight, self.proj.bias)


def patch_vision_patch_embed(model: torch.nn.Module, force: bool = False) -> int:
    """Rebind the patch embedding of one model, off ROCm only when forced."""
    if not force and torch.version.hip is None:
        return 0

    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ not in _PATCH_EMBED_CLASSES:
            continue
        # Bound per instance, so models built elsewhere and state_dict stay untouched.
        module.forward = types.MethodType(_linear_patch_embed_forward, module)
        patched += 1
        get_logger().info("Running %s as a matmul: ROCm segfaults on its Conv3d", name)
    return patched
