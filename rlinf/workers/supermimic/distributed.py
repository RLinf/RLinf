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

"""Synchronous data parallelism for SuperMimic.

Every rank runs the whole graph on its own envs and replay. Before each
optimizer step the gradients of that optimizer's parameters are summed
across ranks, weighted by the local sample count, and divided by the global
count. Clipping and the non-finite check happen after the reduction, so all
ranks see the same gradient, take the same step, and keep identical
parameters and Adam state. Running-norm buffers are not parameters.
``update_norm`` merges each rank's batch before folding it, so those
buffers stay identical too. Averaging an already-updated variance would
count the shared history twice.

This is not DDP: one SuperMimic update has five optimizers, freezes
different parameters per phase, and backpropagates through a frozen world
model, which DDP's per-step reducer does not model.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import timedelta

import torch
import torch.distributed as dist


class DataParallel:
    """Rank info and the collectives SuperMimic needs. World size 1 is a no-op."""

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        device: torch.device | str | None = None,
    ):
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.comm_seconds = 0.0
        if self.world_size > 1 and not dist.is_initialized():
            raise RuntimeError(
                "world_size > 1 needs torch.distributed to be initialized first"
            )

    @classmethod
    def initialize(
        cls,
        world_size: int,
        rank: int,
        device: torch.device,
        timeout: timedelta | None = None,
    ) -> DataParallel:
        """Join the process group from RLinf's env (MASTER_ADDR/PORT, RANK)."""
        if int(world_size) > 1 and not dist.is_initialized():
            backend = "nccl" if torch.device(device).type == "cuda" else "gloo"
            kwargs = {
                "backend": backend,
                "world_size": int(world_size),
                "rank": int(rank),
            }
            if timeout is not None:
                kwargs["timeout"] = timeout
            dist.init_process_group(**kwargs)
        return cls(world_size, rank, device)

    @property
    def enabled(self) -> bool:
        """True when there is more than one rank."""
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """True on rank 0."""
        return self.rank == 0

    def _collective_device(self) -> torch.device:
        if self.enabled and dist.get_backend() == "nccl":
            return self.device
        return torch.device("cpu")

    def reduce_grads(
        self, params: Iterable[torch.nn.Parameter], weight: float, mean: bool = True
    ) -> tuple[float, bool]:
        """Average gradients over ranks, weighted by ``weight`` samples.

        A parameter without a gradient counts as zero. A rank with weight 0
        contributes nothing. With ``mean=False`` the caller has already
        scaled its loss by its global share, and gradients are summed.
        Returns the global weight and whether the reduced gradient is
        finite. With global weight 0 every gradient is cleared and no rank
        should step.
        """
        params = list(params)
        weight = float(weight)
        if not self.enabled:
            finite = True
            if weight <= 0:
                for p in params:
                    p.grad = None
                return 0.0, True
            grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
            if grads:
                finite = bool(torch.isfinite(torch.cat(grads)).all())
            return weight, finite
        start = time.perf_counter()
        scale = weight if mean else 1.0
        parts = []
        for p in params:
            if weight > 0 and p.grad is not None:
                parts.append(p.grad.detach().reshape(-1) * scale)
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        flat = torch.cat(
            parts
            + [torch.tensor([weight], device=params[0].device, dtype=parts[0].dtype)]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        summary = torch.stack(
            (flat[-1].detach(), torch.isfinite(flat).all().to(dtype=flat.dtype))
        )
        total_value, finite_value = summary.tolist()
        total = float(total_value)
        offset = 0
        finite = bool(finite_value) if total > 0 else True
        divisor = total if mean else 1.0
        for p in params:
            n = p.numel()
            if total > 0:
                p.grad = (flat[offset : offset + n] / divisor).view_as(p).clone()
            else:
                p.grad = None
            offset += n
        self.comm_seconds += time.perf_counter() - start
        return total, finite

    def all_reduce(self, values: list[float], op: str = "sum") -> list[float]:
        """Reduce a few Python numbers. ``op`` is ``sum``, ``max`` or ``min``."""
        if not self.enabled:
            return [float(v) for v in values]
        start = time.perf_counter()
        tensor = torch.tensor(
            values, dtype=torch.float64, device=self._collective_device()
        )
        reduce_op = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }[op]
        dist.all_reduce(tensor, op=reduce_op)
        self.comm_seconds += time.perf_counter() - start
        return [float(v) for v in tensor.cpu().tolist()]

    def all_reduce_tensor(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """In-place reduce of one tensor on the collective device."""
        if not self.enabled:
            return tensor
        start = time.perf_counter()
        reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op]
        work = tensor.to(self._collective_device())
        dist.all_reduce(work, op=reduce_op)
        if work is not tensor:
            tensor.copy_(work.to(tensor.device))
        self.comm_seconds += time.perf_counter() - start
        return tensor

    def merge_batch_stats(
        self, count: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chan-merge one batch's count, mean, and population variance.

        World size 1 returns the inputs. Every rank must call this, including
        a rank whose count is 0.
        """
        if not self.enabled:
            return count, mean, var
        from rlinf.models.embodiment.supermimic import merge_welford_batches

        start = time.perf_counter()
        device = self._collective_device()
        count_l = count.detach().reshape(1).to(device=device, dtype=torch.float64)
        mean_l = mean.detach().to(device=device, dtype=torch.float64)
        var_l = var.detach().to(device=device, dtype=torch.float64)
        counts = [torch.empty_like(count_l) for _ in range(self.world_size)]
        means = [torch.empty_like(mean_l) for _ in range(self.world_size)]
        variances = [torch.empty_like(var_l) for _ in range(self.world_size)]
        dist.all_gather(counts, count_l)
        dist.all_gather(means, mean_l)
        dist.all_gather(variances, var_l)
        merged_n, merged_mean, merged_var = merge_welford_batches(
            counts, means, variances
        )
        self.comm_seconds += time.perf_counter() - start
        return (
            merged_n.to(dtype=count.dtype, device=count.device),
            merged_mean.to(dtype=mean.dtype, device=mean.device),
            merged_var.to(dtype=var.dtype, device=var.device),
        )

    def update_norm(self, norm: torch.nn.Module | None, values: torch.Tensor) -> None:
        """Fold ``values`` into a running norm after merging every rank's batch."""
        if norm is None:
            return
        count, mean, var = norm.batch_stats(values)
        if self.enabled:
            count, mean, var = self.merge_batch_stats(count, mean, var)
        norm.update_from_stats(count, mean, var)

    def broadcast_module(self, module: torch.nn.Module, src: int = 0) -> None:
        """Copy parameters and buffers from ``src`` to every rank."""
        if not self.enabled:
            return
        with torch.no_grad():
            for tensor in list(module.parameters()) + list(module.buffers()):
                work = tensor.data.to(self._collective_device())
                dist.broadcast(work, src=src)
                if work is not tensor.data:
                    tensor.data.copy_(work.to(tensor.device))

    def param_divergence(self, module: torch.nn.Module) -> float:
        """Max minus min over ranks of a checksum of parameters and buffers.

        Ranks that took identical steps and applied the same running-norm
        update give exactly 0. Buffers are included so split normalization
        statistics are visible here.
        """
        with torch.no_grad():
            tensors = list(module.parameters()) + list(module.buffers())
            checksum = sum(
                float(t.detach().double().sum())
                + 1e-3 * float(t.detach().double().abs().sum())
                for t in tensors
            )
        if not self.enabled:
            return 0.0
        high = self.all_reduce([checksum], op="max")[0]
        low = self.all_reduce([checksum], op="min")[0]
        return high - low

    def barrier(self) -> None:
        """Wait for every rank; a no-op on one rank."""
        if self.enabled:
            dist.barrier()

    def destroy(self) -> None:
        """Destroy the process group if one was created."""
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
