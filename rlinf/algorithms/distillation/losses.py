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

"""Token objectives, policy-gradient adaptation and global reduction."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from rlinf.algorithms.losses import compute_ppo_actor_loss_tokens

from .config import DISTRIBUTION_MODES, DistillationLossConfig, DistillationPlan

_SAMPLED_LOSSES: dict[str, Callable[[Tensor], Tensor]] = {}


def register_distillation_loss(name: str):
    """Register a canonical sampled-token objective in each importing process."""

    def register(function: Callable[[Tensor], Tensor]):
        if name in _SAMPLED_LOSSES:
            raise ValueError(f"Distillation loss {name!r} is already registered")
        _SAMPLED_LOSSES[name] = function
        return function

    return register


@register_distillation_loss("k1")
def _k1(d: Tensor) -> Tensor:
    return d


@register_distillation_loss("k2")
def _k2(d: Tensor) -> Tensor:
    return 0.5 * d.square()


@register_distillation_loss("abs")
def _abs(d: Tensor) -> Tensor:
    return d.abs()


@register_distillation_loss("k3")
def _k3(d: Tensor) -> Tensor:
    log_ratio = (-d).clamp(-20, 20)
    return (log_ratio.exp() - log_ratio - 1).clamp(-10, 10)


@dataclass
class DistillationForwardOutput:
    """Small differentiable outputs retained after the model forward."""

    logprobs: Tensor | None = None
    entropy: Tensor | None = None
    distribution_losses: Tensor | None = None
    student_mass: Tensor | None = None
    teacher_mass: Tensor | None = None
    tail_loss: Tensor | None = None


@dataclass(frozen=True)
class DistillationReductionContext:
    """Global denominator and compensation for the trainer's gradient reductions.

    The denominator counts tokens or nonempty sequences over the entire update.
    Set gradient_divisor to any divisor applied by the trainer after this loss.
    dp_size compensates data-parallel gradient averaging, not tensor parallelism.
    """

    denominator: float
    mode: str = "token-mean"
    dp_size: int = 1
    gradient_divisor: float = 1.0

    def reduce(self, losses: Tensor, mask: Tensor) -> Tensor:
        """Reduce one microbatch using the whole update's normalization."""
        if self.denominator < 0 or self.dp_size < 1 or self.gradient_divisor <= 0:
            raise ValueError(
                "Expected nonnegative denominator and positive gradient scales"
            )
        values = torch.where(mask, losses, 0.0)
        if self.mode == "seq-mean-token-mean":
            numerator = (values.sum(-1) / mask.sum(-1).clamp_min(1)).sum()
        elif self.mode in {"token-mean", "seq-mean-token-sum"}:
            numerator = values.sum()
        else:
            raise ValueError(f"Unsupported distillation reduction: {self.mode}")
        if self.denominator == 0:
            return numerator * 0
        return numerator * (self.dp_size * self.gradient_divisor / self.denominator)


def compute_distillation_tokens(
    current_logprobs: Tensor | None,
    teacher_logprobs: Tensor,
    *,
    config: DistillationLossConfig,
    distribution_losses: Tensor | None = None,
) -> Tensor:
    """Compute fp32 sampled estimators or clamp already evaluated distributions."""
    mode = config.canonical_loss_mode
    if mode in DISTRIBUTION_MODES:
        if distribution_losses is None:
            raise ValueError(f"{mode} requires distribution_losses from the forward")
        tokens = distribution_losses.float()
    else:
        if current_logprobs is None:
            raise ValueError(f"{mode} requires current Student logprobs")
        teacher = teacher_logprobs.detach().float()
        if teacher.ndim == current_logprobs.ndim + 1 and teacher.shape[-1] == 1:
            teacher = teacher.squeeze(-1)
        if teacher.shape != current_logprobs.shape:
            raise ValueError("Sampled Student and Teacher logprob shapes must match")
        d = current_logprobs.float() - teacher
        tokens = _SAMPLED_LOSSES[mode](d)
    if config.loss_max_clamp is not None:
        tokens = tokens.clamp(-config.loss_max_clamp, config.loss_max_clamp)
    return tokens


def compute_distillation_pg_tokens(
    current_logprobs: Tensor,
    old_logprobs: Tensor,
    distillation_tokens: Tensor,
    *,
    config: DistillationLossConfig,
    loss_mask: Tensor | None = None,
    rollout_is_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Use fresh detached distillation advantages with vanilla PPO clipping."""
    if loss_mask is None:
        loss_mask = torch.ones_like(current_logprobs, dtype=torch.bool)
    tokens, metrics = compute_ppo_actor_loss_tokens(
        current_logprobs.float(),
        old_logprobs.detach().float(),
        -distillation_tokens.detach().float(),
        loss_mask,
        clip_ratio_low=config.clip_ratio_low,
        clip_ratio_high=config.clip_ratio_high,
        clip_ratio_c=config.clip_ratio_c,
        clip_log_ratio_min=-20,
        clip_log_ratio_max=20,
    )
    if rollout_is_weights is not None:
        tokens = tokens * rollout_is_weights.detach()
    return tokens, metrics


def compute_mc_pg_tokens(
    current: Tensor,
    behavior: Tensor,
    teacher: Tensor,
    mask: Tensor,
    config: DistillationLossConfig,
) -> Tensor:
    """Clip each sampled action against its behavior probability, then average K draws."""
    if (
        current.shape != teacher.shape
        or current.shape != behavior.shape
        or current.shape[-1] != config.mc_k
    ):
        raise ValueError(
            "MC Student, behavior and Teacher scores must have the configured candidate width"
        )
    active = mask.unsqueeze(-1)
    current = torch.where(active, current, 0.0)
    teacher = torch.where(active, teacher.detach(), 0.0)
    behavior = torch.where(active, behavior.detach(), 0.0)
    signal = compute_distillation_tokens(current, teacher, config=config)
    losses, _ = compute_distillation_pg_tokens(
        current, behavior, signal, config=config, loss_mask=active
    )
    return losses.mean(-1)


def compute_distillation_loss(
    forward: DistillationForwardOutput,
    batch: dict[str, Tensor],
    *,
    plan: DistillationPlan,
    config: DistillationLossConfig,
    reduction: DistillationReductionContext,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Mask, reweight and reduce distillation exactly once."""
    mask = batch["response_mask"].bool() & batch["teacher_valid_mask"].bool()
    current = forward.logprobs
    if current is not None:
        current = torch.where(mask, current, 0.0)
    teacher = batch["teacher_logprobs"]
    if plan.target_format == "sampled":
        teacher = torch.where(mask.unsqueeze(-1), teacher, 0.0)
    if plan.mc_k > 1:
        tokens = forward.distribution_losses
        if tokens is None:
            raise ValueError("MC-K requires candidate losses from the Student forward")
    else:
        tokens = compute_distillation_tokens(
            current,
            teacher,
            config=config,
            distribution_losses=forward.distribution_losses,
        )
    if plan.use_policy_gradient and plan.mc_k == 1:
        old = (
            batch.get("rollout_logprobs")
            if plan.separate_async
            else batch.get("recomputed_logprobs", batch.get("rollout_logprobs"))
        )
        if old is None or current is None:
            raise ValueError(
                "Distillation PG requires current and old Student logprobs"
            )
        tokens, _ = compute_distillation_pg_tokens(
            current,
            torch.where(mask, old, 0.0),
            tokens,
            config=config,
            loss_mask=mask,
            rollout_is_weights=batch.get("rollout_is_weights"),
        )
    scales = batch.get("distillation_loss_scales", batch.get("loss_scales"))
    if scales is not None:
        tokens = tokens * scales
    loss = reduction.reduce(tokens, mask)
    metrics = {
        "distillation/loss": loss.detach(),
        "distillation/token_count": mask.sum(dtype=torch.float32).detach(),
    }
    if len(plan.teacher_keys) > 1:
        indices = batch["teacher_index"].to(mask.device)
        # Both trainers average microbatch metrics across accumulation and DP.
        scale = reduction.dp_size * reduction.gradient_divisor
        for index, key in enumerate(plan.teacher_keys):
            selected = mask & (indices == index).unsqueeze(-1)
            prefix = f"distillation/{key}"
            metrics[f"{prefix}/valid_tokens"] = (
                selected.sum(dtype=torch.float32) * scale
            )
            metrics[f"{prefix}/loss_sum"] = (
                tokens.detach().masked_fill(~selected, 0).sum() * scale
            )
            if forward.teacher_mass is not None:
                metrics[f"{prefix}/teacher_mass_sum"] = (
                    forward.teacher_mass.detach().masked_fill(~selected, 0).sum()
                    * scale
                )
    return loss, metrics


def combine_distillation_and_task_loss(
    task_loss: Tensor | None,
    distillation_loss: Tensor,
    *,
    use_task_rewards: bool,
    coefficient: float,
) -> Tensor:
    """Keep the pure-distillation coefficient equal to one."""
    if not use_task_rewards:
        return distillation_loss
    if task_loss is None:
        raise ValueError("Joint distillation requires a task loss")
    return task_loss + coefficient * distillation_loss
