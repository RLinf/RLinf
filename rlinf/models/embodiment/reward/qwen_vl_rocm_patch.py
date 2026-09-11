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
import torch.nn as nn
import torch.nn.functional as F

from rlinf.utils.logging import get_logger


def _is_rocm() -> bool:
    """Whether this worker runs on an AMD GPU, per the Worker device API."""
    from rlinf.scheduler import AcceleratorType, Worker

    return Worker.accelerator_type == AcceleratorType.AMD_GPU


def _is_patch_embed(module: nn.Module) -> bool:
    """Whether ``module`` is a Qwen-VL patch embedding the matmul form fits.

    Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3-VL-MoE each name this module
    differently, so match the shape contract instead: a ``proj`` convolution
    whose kernel covers a whole patch and steps by exactly one patch.
    """
    if not type(module).__name__.endswith("PatchEmbed"):
        return False
    proj = getattr(module, "proj", None)
    return isinstance(proj, nn.Conv3d) and proj.kernel_size == proj.stride


def _linear_patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Exact, not an approximation: the caller has already reshaped the input to one
    # block per patch, and kernel_size == stride == a whole block, so each output
    # element is one dot product of a flattened block with a flattened filter.
    weight = self.proj.weight.reshape(self.proj.out_channels, -1)
    hidden_states = hidden_states.reshape(-1, weight.shape[1]).to(weight.dtype)
    return F.linear(hidden_states, weight, self.proj.bias)


def patch_vision_patch_embed(model: torch.nn.Module) -> int:
    """Rebind ``model``'s patch embeddings to the matmul form on ROCm.

    A no-op returning 0 on every other accelerator. The rewrite bypasses the
    ``proj`` submodule, so forward hooks and wrappers attached to it stop
    firing; call this before anything wraps ``proj``.

    Returns:
        How many modules were rebound.
    """
    if not _is_rocm():
        return 0

    patched = 0
    for name, module in model.named_modules():
        if not _is_patch_embed(module):
            continue
        # Bound per instance, so models built elsewhere and state_dict stay untouched.
        module.forward = types.MethodType(_linear_patch_embed_forward, module)
        patched += 1
        get_logger().info("Running %s as a matmul: ROCm segfaults on its Conv3d", name)

    if patched == 0:
        get_logger().warning(
            "No Qwen-VL patch embedding found in %s; its Conv3d will run as-is "
            "and may segfault on ROCm",
            type(model).__name__,
        )
    return patched
