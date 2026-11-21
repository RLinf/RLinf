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

"""Score centering for asynchronous PEAR rollouts.

The correction matches PEAR's top-k head formula: the sampler tail is modeled
as the trainer tail rescaled by ``rho``. On-policy heads therefore contribute
zero correction. The synchronous MVP leaves this disabled.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def score_centering_weight_fn(
    rollout_is: str | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the identity IS weight used by the synchronous correction."""
    if rollout_is is not None:
        raise ValueError(
            "PEAR score centering in RLinf currently composes with no "
            f"importance sampling, got {rollout_is!r}."
        )
    return torch.ones_like


def score_centering_correction(
    train_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the per-token centering term and both head masses.

    Args:
        train_head_log_probs: Trainer log probabilities at the sampler head,
            shape ``[N, K]``, with gradient.
        sampler_head_log_probs: Sampler log probabilities at the same ids.
        weight_fn: Ratio weight. Defaults to no importance sampling.
        eps: Floor for tail masses.

    Returns:
        Correction ``[N]``, sampler head mass ``[N]``, and trainer head mass.
    """
    if weight_fn is None:
        weight_fn = score_centering_weight_fn()
    train_head_log_probs = train_head_log_probs.float()
    with torch.no_grad():
        trainer_prob = train_head_log_probs.exp()
        sampler_prob = sampler_head_log_probs.float().exp()
        trainer_mass = trainer_prob.sum(-1)
        sampler_mass = sampler_prob.sum(-1)
        rho = (1 - sampler_mass).clamp_min(eps) / (1 - trainer_mass).clamp_min(eps)
        alpha = rho * weight_fn(1.0 / rho)
        head_weights = weight_fn(
            (train_head_log_probs - sampler_head_log_probs.float()).exp()
        )
        residual = sampler_prob * head_weights - alpha.unsqueeze(-1) * trainer_prob
    correction = (residual * train_head_log_probs).sum(-1)
    return correction, sampler_mass, trainer_mass
