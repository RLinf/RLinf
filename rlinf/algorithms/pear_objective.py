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

"""PEAR REINFORCE policy loss with optional Score Centering."""

from __future__ import annotations

from collections.abc import Callable

import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.utils.utils import masked_mean


def pear_reinforce_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    loss_agg_func: Callable[..., torch.Tensor] | None = None,
    sc_correction: torch.Tensor | None = None,
    sc_sampler_head_mass: torch.Tensor | None = None,
    sc_train_head_mass: torch.Tensor | None = None,
    global_valid_tokens: float | torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Token-mean REINFORCE with optional PEAR Score Centering.

    Matches PEAR's bypass-mode REINFORCE loss (verl patch 0011):
    ``pg_losses = -advantages * logprobs + advantages * sc_correction``.
    No PPO clip, KL term, or entropy bonus.

    When ``global_valid_tokens`` is provided (global token count across all
    micro-batches and DP ranks), the loss is normalized by that denominator so
    gradient accumulation yields an exact batch-wide token-mean rather than a
    sequence-mean of per-microbatch token-means.
    """
    del kwargs
    if loss_agg_func is None:
        loss_agg_func = masked_mean
    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs, dtype=torch.bool)
    mask_bool = loss_mask.to(device=logprobs.device, dtype=torch.bool)
    token_loss = -advantages * logprobs
    if sc_correction is not None:
        token_loss = token_loss + advantages * sc_correction
    if global_valid_tokens is not None:
        denom = (
            global_valid_tokens.to(device=token_loss.device, dtype=token_loss.dtype)
            if torch.is_tensor(global_valid_tokens)
            else token_loss.new_tensor(float(global_valid_tokens))
        ).clamp_min(1.0)
        policy_loss = (token_loss * mask_bool.to(dtype=token_loss.dtype)).sum() / denom
    else:
        policy_loss = loss_agg_func(token_loss, mask_bool, None)
    metrics = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/clip_fraction": torch.zeros((), device=policy_loss.device),
        "actor/ratio": torch.ones((), device=policy_loss.device),
    }
    if sc_correction is not None:
        metrics["actor/sc_correction"] = masked_mean(sc_correction.detach(), mask_bool)
    if sc_sampler_head_mass is not None:
        metrics["actor/sc_sampler_head_mass"] = masked_mean(
            sc_sampler_head_mass.detach(), mask_bool
        )
    if sc_train_head_mass is not None:
        metrics["actor/sc_train_head_mass"] = masked_mean(
            sc_train_head_mass.detach(), mask_bool
        )
    return policy_loss, metrics


@register_policy_loss("pear_reinforce")
def compute_pear_reinforce_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """Registered PEAR REINFORCE policy loss."""
    return pear_reinforce_loss(**kwargs)
