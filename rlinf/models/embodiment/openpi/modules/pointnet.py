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

"""PointNet encoder for PyTorch, aligned with JAX models/pointnet.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

Variant = Literal["pcd", "rgb_pcd"]


@dataclass
class Config:
    n_coordinates: int
    output_dim: int
    hidden_dim: int
    hidden_depth: int
    n_color: int = 0


def get_config(variant: Variant) -> Config:
    if variant == "pcd":
        return Config(
            n_coordinates=3,
            output_dim=2048,
            hidden_dim=1024,
            hidden_depth=2,
        )
    if variant == "rgb_pcd":
        return Config(
            n_coordinates=3,
            n_color=3,
            output_dim=2048,
            hidden_dim=1024,
            hidden_depth=2,
        )
    raise ValueError(f"Unknown variant: {variant}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        output_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        layers = []
        cur_dim = in_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(cur_dim, hidden_dim))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, output_dim))
        layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PointNetSimplified(nn.Module):
    def __init__(
        self,
        point_channels: int,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self._mlp = MLP(
            in_dim=point_channels,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., num_points, point_channels)
        x = self._mlp(x)
        return torch.max(x, dim=-2).values


class UncoloredPointNet(nn.Module):
    def __init__(
        self,
        n_coordinates: int = 3,
        output_dim: int = 2048,
        hidden_dim: int = 2048,
        hidden_depth: int = 2,
        activation: str = "gelu",
        subtract_mean: bool = False,
    ):
        super().__init__()
        self.subtract_mean = subtract_mean
        pn_in_channels = n_coordinates
        if subtract_mean:
            pn_in_channels += n_coordinates
        self.pointnet = PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_points, 3)
        point = x
        if self.subtract_mean:
            mean = torch.mean(point, dim=-2, keepdim=True)
            point = point - mean
            point = torch.cat([point, mean], dim=-1)
        return self.pointnet(point)
