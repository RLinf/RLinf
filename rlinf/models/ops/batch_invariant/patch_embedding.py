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


import torch

from .linear import triton_linear


def triton_patch_embedding(
    image: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Project non-overlapping NCHW image patches with no padding."""
    _, in_channels, patch_h, patch_w = weight.shape
    patches = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    batch, _, rows, cols, _, _ = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch, rows, cols, in_channels * patch_h * patch_w
    )
    tokens = triton_linear(patches, weight.flatten(1), bias)
    return tokens.permute(0, 3, 1, 2)
