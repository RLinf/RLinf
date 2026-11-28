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

"""Differentiable column-based rotation representations."""

import torch
import torch.nn.functional as F


def canonicalize_rot6d(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the first two orthonormal columns after Gram–Schmidt.

    Positive rescaling and adding the first column to the second leave the
    result unchanged. Degenerate inputs use a deterministic orthogonal axis;
    gradients follow Gram–Schmidt everywhere outside that fallback region.
    """
    if value.shape[-1] != 6:
        raise ValueError(f"Expected rot6d trailing dimension 6, got {value.shape}.")
    first, second = value[..., :3], value[..., 3:]
    norm = first.norm(dim=-1, keepdim=True)
    identity_first = torch.zeros_like(first)
    identity_first[..., 0] = 1
    first = torch.where(norm > eps, first / norm.clamp_min(eps), identity_first)
    second = second - (first * second).sum(-1, keepdim=True) * first
    norm = second.norm(dim=-1, keepdim=True)
    axis = F.one_hot(first.abs().argmin(-1), num_classes=3).to(value.dtype)
    fallback = axis - (first * axis).sum(-1, keepdim=True) * first
    fallback = F.normalize(fallback, dim=-1, eps=eps)
    second = torch.where(norm > eps, second / norm.clamp_min(eps), fallback)
    return torch.cat((first, second), dim=-1)


def rot6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Decode column-based rot6d to a right-handed rotation matrix."""
    canonical = canonicalize_rot6d(value)
    first, second = canonical[..., :3], canonical[..., 3:]
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def quaternion_wxyz_to_rot6d(value: torch.Tensor) -> torch.Tensor:
    """Encode nonzero wxyz quaternions as two rotation-matrix columns."""
    if value.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion trailing dimension 4, got {value.shape}."
        )
    w, x, y, z = F.normalize(value, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ),
        dim=-1,
    )
