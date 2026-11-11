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

"""Running mean and variance of SuperMimic network inputs.

Statistics are folded batch by batch with Chan's parallel update. Data
parallel ranks merge their batches first and fold the merged batch once,
so every rank holds the same buffers.
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
    """Merge two population-variance batches. A non-positive count is skipped.

    The skip is selected on the device, so no count is read back to the host.
    """
    has_a = n_a.reshape(()) > 0
    has_b = n_b.reshape(()) > 0
    n_a = n_a.reshape(()).clamp_min(0)
    n_b = n_b.reshape(()).clamp_min(0)
    total = n_a + n_b
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / safe_total)
    var = (
        var_a * n_a + var_b * n_b + delta.square() * (n_a * n_b / safe_total)
    ) / safe_total
    mean = torch.where(has_b, torch.where(has_a, mean, mean_b), mean_a)
    var = torch.where(has_b, torch.where(has_a, var, var_b), var_a)
    return total, mean, var


def merge_welford_batches(
    counts: list[torch.Tensor],
    means: list[torch.Tensor],
    variances: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chan-merge population batches into one count, mean, and variance.

    A zero count is skipped. In exact arithmetic the result is the mean and
    population variance of the concatenated rows. Averaging variances that
    were already folded into a shared history would count that history twice.
    """
    if not counts:
        raise ValueError("merge_welford_batches needs at least one batch")
    acc_n = counts[0].detach().reshape(()).to(dtype=torch.float64) * 0
    acc_mean = means[0].detach().to(dtype=torch.float64) * 0
    acc_var = variances[0].detach().to(dtype=torch.float64) * 0
    for count, mean, var in zip(counts, means, variances, strict=True):
        acc_n, acc_mean, acc_var = _chan_merge(
            acc_n,
            acc_mean,
            acc_var,
            count.detach().to(dtype=torch.float64),
            mean.detach().to(dtype=torch.float64),
            var.detach().to(dtype=torch.float64),
        )
    return acc_n, acc_mean, acc_var


def batch_stats(
    values: torch.Tensor, dim: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count, mean, and population variance over the rows of ``values``.

    An empty batch has count 0. One row has variance 0. ``dim`` is the
    feature width of an empty batch whose last axis is not known.
    """
    width = int(values.shape[-1]) if dim is None else int(dim)
    flat = values.detach().reshape(-1, width).float()
    if flat.shape[0] == 0:
        zeros = flat.new_zeros(width)
        return flat.new_zeros(()), zeros, zeros.clone()
    var, mean = torch.var_mean(flat, dim=0, correction=0)
    # ``new_full`` fills on the device. ``torch.tensor(..., device=cuda)``
    # copies from the host and waits for every queued kernel.
    return flat.new_full((), float(flat.shape[0])), mean, var


def pack_stats(
    count: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """``[count, mean..., var...]`` as one float32 vector."""
    return torch.cat((count.reshape(1).float(), mean.float(), var.float()))


def unpack_stats(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_stats`."""
    width = (int(packed.shape[0]) - 1) // 2
    return packed[0], packed[1 : 1 + width], packed[1 + width :]


def merge_rank_stats(
    count: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chan-merge one batch from every rank of ``group``.

    Every rank must call this, including a rank whose count is 0. The merge
    runs in float64 and the result is cast back to the input dtype.
    """
    world = dist.get_world_size(group)
    packed = pack_stats(count, mean, var).double()
    gathered = [torch.empty_like(packed) for _ in range(world)]
    dist.all_gather(gathered, packed, group=group)
    parts = [unpack_stats(item) for item in gathered]
    merged = merge_welford_batches(
        [part[0] for part in parts],
        [part[1] for part in parts],
        [part[2] for part in parts],
    )
    return (
        merged[0].to(dtype=count.dtype),
        merged[1].to(dtype=mean.dtype),
        merged[2].to(dtype=var.dtype),
    )


def _distributed(group: dist.ProcessGroup | None) -> bool:
    return (
        group is not None and dist.is_initialized() and dist.get_world_size(group) > 1
    )


def merged_batch_stats(
    values: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    dim: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Packed statistics of ``values`` over every rank, and the host-side count.

    Without a multi-rank ``group`` the count is the local row count and no
    device value is read back.
    """
    width = int(values.shape[-1]) if dim is None else int(dim)
    rows = 0 if values.numel() == 0 else int(values.reshape(-1, width).shape[0])
    count, mean, var = batch_stats(values, width)
    if not _distributed(group):
        return pack_stats(count, mean, var), float(rows)
    count, mean, var = merge_rank_stats(count, mean, var, group)
    return pack_stats(count, mean, var), float(count)


class RunningMeanStd(nn.Module):
    """Mean and population variance folded batch by batch, without gradients.

    Statistics stop changing after ``max_count`` rows. ``eps`` is added in
    quadrature to the standard deviation: 1e-2 on network inputs and 1e-4 on
    the world-model residual.
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = float(max_count)
        self.eps = float(eps)
        # Host copy of ``count``. The freeze check reads it, so an update does
        # not wait for the device to finish the previous kernel.
        self._count_cpu = 0.0
        self.register_load_state_dict_post_hook(
            lambda module, _incompatible: module.refresh_host_count()
        )

    def refresh_host_count(self) -> None:
        """Read ``count`` back into its host copy.

        ``load_state_dict`` does this itself. Code that writes the buffers in
        place, such as a weight syncer, must call it before the next fold.
        """
        self._count_cpu = float(self.count.detach().cpu())

    def batch_stats(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Count, mean, and population variance of one batch."""
        return batch_stats(values, int(self.mean.shape[0]))

    def frozen(self) -> bool:
        """Whether ``max_count`` rows have already been folded."""
        return self._count_cpu >= self.max_count

    @torch.no_grad()
    def update_from_stats(
        self,
        count: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        host_count: float | None = None,
    ) -> None:
        """Fold one batch given its count, mean, and population variance.

        A count of 0 leaves the buffers unchanged, and so does a frozen norm.
        ``host_count`` is that count already known on the CPU; without it the
        count tensor is read back once.
        """
        if self.frozen():
            return
        if host_count is None:
            host_count = float(count)
        if host_count <= 0.0:
            return
        count = count.detach().to(device=self.count.device, dtype=self.count.dtype)
        mean = mean.detach().to(device=self.mean.device, dtype=self.mean.dtype)
        var = var.detach().to(device=self.var.device, dtype=self.var.dtype)
        total, new_mean, new_var = _chan_merge(
            self.count, self.mean, self.var, count, mean, var
        )
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total.reshape(()))
        added = torch.tensor(self._count_cpu, dtype=self.count.dtype) + torch.tensor(
            host_count, dtype=self.count.dtype
        )
        self._count_cpu = float(added)

    def update_from_packed(self, packed: torch.Tensor, host_count: float) -> None:
        """Fold statistics produced by :func:`merged_batch_stats`."""
        self.update_from_stats(*unpack_stats(packed), host_count=host_count)

    @torch.no_grad()
    def fold(
        self, values: torch.Tensor, group: dist.ProcessGroup | None = None
    ) -> None:
        """Fold ``values`` after merging every rank's rows over ``group``."""
        if self.frozen():
            return
        width = int(self.mean.shape[0])
        rows = 0 if values.numel() == 0 else int(values.reshape(-1, width).shape[0])
        if rows == 0 and not _distributed(group):
            return
        packed, host = merged_batch_stats(values, group, width)
        self.update_from_packed(packed, host)

    def std(self) -> torch.Tensor:
        """Standard deviation including ``eps``."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize ``values`` on the last dimension."""
        return (values - self.mean) / self.std()
