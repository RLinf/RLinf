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

"""Normalization and canonical network inputs for ten-dimensional EEF actions."""

import json
from pathlib import Path

import torch
from torch import nn

from rlinf.utils.torch_rot6d import canonicalize_rot6d, rot6d_to_matrix


class EEFNormalizer(nn.Module):
    """Own pi05 quantile statistics for xyz, column rot6d, and gripper coordinates."""

    def __init__(self, path: str) -> None:
        super().__init__()
        stats = json.loads(Path(path).read_text())["norm_stats"]
        for key in ("state", "actions"):
            low = torch.tensor(stats[key]["q01"], dtype=torch.float32)
            high = torch.tensor(stats[key]["q99"], dtype=torch.float32)
            if (
                low.shape != (10,)
                or high.shape != (10,)
                or not torch.isfinite(low).all()
                or not torch.isfinite(high).all()
                or (high < low).any()
            ):
                raise ValueError(
                    f"Expected finite ordered 10-D EEF {key} quantiles in {path}."
                )
            self.register_buffer(key + "_offset", (high + low + 1e-6) / 2)
            self.register_buffer(key + "_scale", (high - low + 1e-6) / 2)

    def normalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        """Map raw dataset coordinates to pi05 quantile-normalized coordinates."""
        return (value - getattr(self, key + "_offset")) / getattr(self, key + "_scale")

    def unnormalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        """Restore raw dataset coordinates without clipping out-of-range values."""
        return value * getattr(self, key + "_scale") + getattr(self, key + "_offset")

    def network_input(self, value: torch.Tensor, key: str) -> torch.Tensor:
        """Keep normalized xyz/gripper and canonicalize rotation in physical coordinates."""
        rotation = canonicalize_rot6d(self.unnormalize(value, key)[..., 3:9])
        return torch.cat((value[..., :3], rotation, value[..., 9:]), -1)


def decode_eef_actions(actions: torch.Tensor) -> torch.Tensor:
    """Map raw xyz/rot6d/gripper labels to normalized GPU EEF controller commands.

    Translation and gripper are already controller coordinates. Intrinsic XYZ
    rotation uses ManiSkill's signed rotation scale, -0.1 radians.
    """
    if actions.shape[-1] != 10 or not torch.isfinite(actions).all():
        raise ValueError(
            f"Expected finite EEF actions ending in 10, got {actions.shape}."
        )
    matrix = rot6d_to_matrix(actions[..., 3:9])
    pitch = torch.asin(matrix[..., 0, 2].clamp(-1, 1))
    roll = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    yaw = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    singular = torch.cos(pitch).abs() < 1e-6
    roll = torch.where(
        singular, torch.atan2(matrix[..., 2, 1], matrix[..., 1, 1]), roll
    )
    yaw = torch.where(singular, torch.zeros_like(yaw), yaw)
    angles = torch.stack((roll, pitch, yaw), -1) / -0.1
    return torch.cat((actions[..., :3], angles, actions[..., 9:10]), -1)
