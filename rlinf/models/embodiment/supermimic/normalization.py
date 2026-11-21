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

"""Running input normalization of the SuperMimic networks.

The env ranks merge one vector step's rows into a single packed
``[count, mean..., var...]`` record; every copy of a norm folds that record,
so the rollout and actor copies stay equal without moving buffers.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def _chan_merge(
    n_a: torch.Tensor,
    mean_a: torch.Tensor,
    var_a: torch.Tensor,
    n_b: torch.Tensor,
    mean_b: torch.Tensor,
    var_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count, mean and population variance of two batches together."""
    total = n_a + n_b
    rate = n_b / total
    delta = mean_b - mean_a
    mean = mean_a + rate * delta
    var = var_a + rate * (var_b - var_a + delta * (mean_b - mean))
    return total, mean, var


def pack_moments(values: torch.Tensor) -> torch.Tensor:
    """``[count, mean..., var...]`` in float64 over the rows of ``values``."""
    flat = values.detach().reshape(-1, values.shape[-1]).double()
    var, mean = torch.var_mean(flat, dim=0, correction=0)
    return torch.cat((flat.new_full((1,), float(flat.shape[0])), mean, var))


def _unpack(packed: torch.Tensor):
    width = (packed.shape[0] - 1) // 2
    return packed[0], packed[1 : 1 + width], packed[1 + width :]


def merged_moments(
    values: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """:func:`pack_moments` of the rows of every rank of ``group``.

    Every rank of ``group`` must call this with a non-empty batch.
    """
    packed = pack_moments(values)
    if group is None or dist.get_world_size(group) == 1:
        return packed
    gathered = [torch.empty_like(packed) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, packed, group=group)
    count, mean, var = _unpack(gathered[0])
    for other in gathered[1:]:
        count, mean, var = _chan_merge(count, mean, var, *_unpack(other))
    return torch.cat((count.reshape(1), mean, var))


class EmpiricalNorm(nn.Module):
    """``(x - mean) / (std + 0.01)`` with the moments of every row folded so far.

    Before the first fold the mean is 0 and the variance 1.
    """

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def fold(self, packed: torch.Tensor) -> None:
        """Fold one record of :func:`pack_moments` or :func:`merged_moments`."""
        count, mean, var = _unpack(packed.to(self.count.device, torch.float64))
        total, mean, var = _chan_merge(
            self.count, self.mean.double(), self.var.double(), count, mean, var
        )
        self.count.copy_(total)
        self.mean.copy_(mean)
        self.var.copy_(var)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / (self.var.sqrt() + self.eps)
