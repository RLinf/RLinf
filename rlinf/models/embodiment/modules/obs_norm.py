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

"""Running empirical normalization for vector observations."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize vector observations using running mean and standard deviation.

    Statistics live in persistent buffers so weight syncers propagate them to
    rollout workers and checkpoints save them alongside model weights.

    Args:
        shape: Feature dimension of the observation vector.
        eps: Floor added to the standard deviation to avoid division by zero.
        until: Optional sample count cap after which statistics freeze.
    """

    def __init__(
        self,
        shape: int,
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()
        if shape < 1:
            raise ValueError(f"shape must be positive, got {shape}.")
        self.shape = int(shape)
        self.eps = float(eps)
        self.until = until
        self.register_buffer("_mean", torch.zeros(1, self.shape, dtype=torch.float32))
        self.register_buffer("_var", torch.ones(1, self.shape, dtype=torch.float32))
        self.register_buffer("_std", torch.ones(1, self.shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def _apply(self, fn, recurse: bool = True):
        """Keep running statistics in float32 when the parent module casts dtype."""
        super()._apply(fn, recurse=recurse)
        for name in ("_mean", "_var", "_std"):
            buf = getattr(self, name)
            if buf.dtype != torch.float32:
                setattr(self, name, buf.float())
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` using the current running statistics."""
        return (x - self._mean.to(x.dtype)) / (self._std.to(x.dtype) + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics from a batch ``[..., shape]``.

        When distributed training is initialized, batch moments are all-reduced
        across data-parallel ranks so every worker keeps identical statistics.
        """
        if self.until is not None and int(self.count.item()) >= self.until:
            return
        flat = (
            x.detach()
            .reshape(-1, self.shape)
            .to(device=self._mean.device, dtype=torch.float64)
        )
        batch_count = torch.tensor(
            flat.shape[0], dtype=torch.float64, device=flat.device
        )
        batch_sum = flat.sum(dim=0, keepdim=True)
        batch_sq_sum = flat.pow(2).sum(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sq_sum, op=dist.ReduceOp.SUM)

        n = int(batch_count.item())
        if n == 0:
            return
        batch_mean = batch_sum / n
        batch_var = (batch_sq_sum / n - batch_mean.pow(2)).clamp_min(0.0)

        prev_count = int(self.count.item())
        total_count = prev_count + n
        if prev_count == 0:
            new_mean = batch_mean
            new_var = batch_var
        else:
            prev_mean = self._mean.to(torch.float64)
            prev_var = self._var.to(torch.float64)
            delta = batch_mean - prev_mean
            new_mean = prev_mean + delta * (n / total_count)
            m_a = prev_var * prev_count
            m_b = batch_var * n
            m2 = m_a + m_b + delta.pow(2) * (prev_count * n / total_count)
            new_var = (m2 / total_count).clamp_min(0.0)

        self._mean.copy_(new_mean.to(self._mean.dtype))
        self._var.copy_(new_var.to(self._var.dtype))
        self._std.copy_(torch.sqrt(self._var).to(self._std.dtype))
        self.count.fill_(total_count)
