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

from typing import Any

import torch

BEHAVIOR_TRANSITION_MEAN = "behavior_transition_mean"
BEHAVIOR_TRANSITION_STD = "behavior_transition_std"


def require_transition_stats(policy: Any) -> None:
    """Check that a loaded policy can return Gaussian transition statistics."""
    validate = getattr(policy, "validate_transition_stats", None)
    if validate is None:
        raise ValueError(
            f"KL clipping requires transition statistics, unsupported by {type(policy).__name__}."
        )
    validate()


@torch.no_grad()
def transition_kl(
    behavior: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return per-sample Gaussian KL over dimensions with stochastic logprobs."""
    old_mean, old_std = behavior["mean"], behavior["std"]
    new_mean, new_std = current["mean"], current["std"]
    if old_mean.shape != new_mean.shape:
        raise ValueError(
            f"Expected matching Gaussian transition shapes, got "
            f"{tuple(old_mean.shape)} and {tuple(new_mean.shape)}."
        )
    old_deterministic = old_std == 0
    new_deterministic = new_std == 0
    new_std = new_std.masked_fill(new_deterministic, 1.0)
    ratio = old_std.masked_fill(old_deterministic, 1.0) / new_std
    shift = (old_mean - new_mean) / new_std
    variance_kl = ratio.square() - 1.0 - 2.0 * ratio.log()
    kl = 0.5 * (shift.square() + variance_kl.clamp_min(0.0))
    kl = kl.masked_fill(old_deterministic & new_deterministic, 0.0)
    kl = kl.masked_fill(old_deterministic ^ new_deterministic, torch.inf)
    return kl.flatten(1).sum(dim=1)


_KL_CLIP_TOTALS = (
    "actor/conditional_kl_sum",
    "actor/conditional_kl_count",
    "actor/kl_clip_tightened_count",
    "actor/kl_clip_extra_count",
)


class KLClip:
    """Apply one KL target and accumulate rollout metrics on the actor device."""

    def __init__(self, target_kl: float, device: torch.device) -> None:
        self.target_kl = target_kl
        self._totals = torch.zeros(
            len(_KL_CLIP_TOTALS), device=device, dtype=torch.float32
        )

    def clip_ratio(
        self,
        ratio: torch.Tensor,
        clip_low: float,
        clip_high: float,
        kl: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the clipped ratio and detached statistics for this microbatch."""
        kl = kl.detach().masked_fill(~loss_mask, 0.0)
        scale = (self.target_kl / kl.clamp_min(self.target_kl)).sqrt().clamp_min(0.25)
        clipped = ratio.clamp(min=1.0 - clip_low * scale, max=1.0 + clip_high * scale)
        return clipped, kl, scale < 1

    def record(
        self,
        kl: torch.Tensor,
        loss_mask: torch.Tensor,
        tightened: torch.Tensor,
        extra_clipped: torch.Tensor,
    ) -> None:
        """Add detached sample counts without reading device values."""
        self._totals.add_(
            torch.stack(
                (kl.sum(), loss_mask.sum(), tightened.sum(), extra_clipped.sum())
            )
        )

    def totals(self) -> dict[str, float]:
        """Read the packed totals once at the end of the rollout update."""
        return dict(zip(_KL_CLIP_TOTALS, self._totals.tolist()))


def finalize_kl_clip_metrics(metrics: dict[str, float]) -> None:
    """Report sample-weighted KL metrics from the reduced totals."""
    kl_sum, count, tightened, extra = (metrics.pop(key) for key in _KL_CLIP_TOTALS)
    metrics["actor/conditional_kl_mean"] = kl_sum / count if count else 0.0
    metrics["actor/kl_clip/tight_frac"] = tightened / count if count else 0.0
    metrics["actor/kl_clip/extra_frac"] = extra / count if count else 0.0
