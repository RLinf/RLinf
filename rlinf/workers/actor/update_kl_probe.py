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

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class _CapturedBatch:
    inputs: Any
    mean: torch.Tensor
    std: torch.Tensor
    valid: torch.Tensor


@dataclass
class _ProbeReference:
    batches: list[tuple[Any, int]]
    mean: torch.Tensor
    std: torch.Tensor
    stochastic: torch.Tensor
    valid: torch.Tensor


class UpdateKLProbe:
    """Measure conditional KL against the actor at the start of an update round."""

    def __init__(
        self,
        score: Callable[[Any], dict[str, Any]],
        *,
        target_kl: float,
        group: dist.ProcessGroup,
        cache_prefix: bool = False,
    ) -> None:
        self._score = score
        self._group = group
        self._target_kl = target_kl
        self._cache_prefix = cache_prefix
        self.capture_kwargs: dict[str, Any] = {"denoise_probe": "capture"}
        self._remaining = 0
        self._captured: list[_CapturedBatch] = []
        self._reference: _ProbeReference | None = None
        self.valid_count = 0
        self._first_kl = 0.0
        self.last_kl: float | None = None
        self.stopped = False

    def configure_capture(self, choose_cache: Callable[[bool], str]) -> None:
        """Resolve the model's replay cache before the optimizer freezes parameters."""
        self.capture_kwargs["replay_cache"] = choose_cache(self._cache_prefix)

    def begin_round(self, batch_size_per_rank: int) -> None:
        """Select the leading samples of the first optimizer batch as the probe."""
        self._remaining = min(128, batch_size_per_rank)
        self.valid_count = 0
        self._first_kl = 0.0
        self.last_kl = None
        self.stopped = False

    @property
    def needs_reference(self) -> bool:
        """Whether the training forward should return reference statistics."""
        return self._remaining > 0

    def capture_reference(
        self, stats: dict[str, Any], loss_mask: torch.Tensor | None
    ) -> None:
        """Record reference statistics from a microbatch in the first update."""
        count = min(self._remaining, stats["denoise_mean"].size(0))
        self._remaining -= count
        mean = stats["denoise_mean"][:count]
        std = stats["denoise_std"][:count]
        valid = (std > 0).flatten(1).any(dim=1)
        if loss_mask is not None:
            valid &= loss_mask[:count].reshape(count).to(valid)
        self._captured.append(_CapturedBatch(stats["probe_inputs"], mean, std, valid))

    def finish_reference(self) -> None:
        """Assemble the reference after the first gradient accumulation."""
        std = torch.cat([batch.std for batch in self._captured])
        stochastic = std > 0
        valid = torch.cat([batch.valid for batch in self._captured])

        valid_count = valid.sum()
        dist.all_reduce(valid_count, group=self._group)
        self.valid_count = int(valid_count.item())
        self._reference = _ProbeReference(
            batches=[(batch.inputs, batch.mean.size(0)) for batch in self._captured],
            mean=torch.cat([batch.mean for batch in self._captured]),
            std=torch.where(stochastic, std, 1.0),
            stochastic=stochastic,
            valid=valid,
        )
        self._captured.clear()

    @contextmanager
    def check(self) -> Iterator[None]:
        """Reduce the current KL while the caller handles host-only bookkeeping."""
        reference = self._reference
        if reference is None or self.valid_count == 0:
            yield
            return

        # Replay the original microbatch shape even when only its leading samples
        # belong to the probe.
        mean = torch.cat(
            [
                self._score(batch)["denoise_mean"][:count]
                for batch, count in reference.batches
            ]
        )
        shift = (mean - reference.mean) / reference.std
        chunk_kl = 0.5 * shift.square().masked_fill(~reference.stochastic, 0).sum(
            dim=(1, 2)
        )
        kl_sum = chunk_kl.masked_fill(~reference.valid, 0).sum()
        reduction = dist.all_reduce(kl_sum, group=self._group, async_op=True)
        try:
            yield
        finally:
            reduction.wait()

        kl = kl_sum.item() / self.valid_count
        if not math.isfinite(kl):
            raise RuntimeError(f"Expected finite policy update KL, got {kl}.")
        if self.last_kl is None:
            self._first_kl = kl
        self.last_kl = kl
        self.stopped = kl >= self._target_kl

    def round_metrics(
        self, steps_used: int, step_budget: int
    ) -> dict[str, int | float]:
        """Report update usage and measured KL for the current round."""
        metrics = {
            "actor/optimizer_steps_used": steps_used,
            "actor/update_kl_stopped": int(self.stopped),
            "actor/update_kl_probe_count": self.valid_count,
            "actor/kl_stop/used_frac": steps_used / step_budget,
        }
        if self.last_kl is not None:
            metrics["actor/kl_stop/first_kl"] = self._first_kl
            metrics["actor/update_kl"] = self.last_kl
        return metrics

    def end_round(self) -> None:
        """Release the staged inputs and reference tensors for the completed round."""
        self._reference = None
        self._captured.clear()
        self._remaining = 0
