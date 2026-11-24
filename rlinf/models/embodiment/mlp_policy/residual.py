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

"""Bounded stochastic action edits for frozen reference policies."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.mlp_policy.heads import (
    ActorHeadCapabilities,
    PrefixActorHead,
    register_prefix_actor,
)


def expand_edit_scales(value: Any, action_dim: int) -> list[float]:
    """Expand a scalar or repeating per-coordinate bounds across an action chunk."""
    scales = [float(value)] if isinstance(value, (int, float)) else list(value)
    if not scales or action_dim % len(scales):
        raise ValueError(
            "edit_scale length must divide the flattened action chunk dimension."
        )
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("edit_scale values must be finite and strictly positive.")
    return scales * (action_dim // len(scales))


class ResidualMLPActor(PrefixActorHead):
    """Learn a bounded tanh-Gaussian edit of the reference action."""

    capabilities = ActorHeadCapabilities(
        distribution="stochastic",
        bounded_residual=True,
    )

    def __init__(
        self, state_dim: int, reference_dim: int, action_dim: int, cfg: Any
    ) -> None:
        super().__init__()
        if reference_dim != action_dim:
            raise ValueError(
                "Residual actors require matching reference and action dimensions."
            )
        initial_log_std = float(cfg.get("initial_log_std", -1.5))
        if not math.isfinite(initial_log_std) or not -5 <= initial_log_std <= 2:
            raise ValueError("initial_log_std must be finite and within [-5, 2].")
        layers = []
        width = state_dim + reference_dim
        for hidden in cfg.get("hidden_dims", [256, 256, 256]):
            layers.extend(
                (nn.Linear(width, int(hidden)), nn.LayerNorm(int(hidden)), nn.SiLU())
            )
            width = int(hidden)
        output = nn.Linear(width, 2 * action_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        with torch.no_grad():
            output.bias[action_dim:].fill_(initial_log_std)
        self.editor = nn.Sequential(*layers, output)
        self.register_buffer(
            "edit_scale",
            torch.tensor(expand_edit_scales(cfg.get("edit_scale", 0.02), action_dim)),
        )

    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic: bool = False,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float | None = None,
        apply_action_noise: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return composed actions and per-coordinate log densities of the edits."""
        if apply_reference_dropout and reference_dropout_prob:
            raise ValueError("Residual edits require the unmodified reference action.")
        mean, log_std = self.editor(torch.cat((state.detach(), reference), -1)).chunk(
            2, -1
        )
        log_std = log_std.clamp(-5, 2)
        noise = torch.zeros_like(mean) if deterministic else torch.randn_like(mean)
        raw = mean + log_std.exp() * noise
        action = reference + raw.tanh() * self.edit_scale
        log_prob = -0.5 * (noise.square() + math.log(2 * math.pi)) - log_std
        log_prob -= 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob -= self.edit_scale.log()
        return action, log_prob

    def mean_action(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Return the reference plus the deterministic bounded edit."""
        return self(state, reference, deterministic=True)[0]


@register_prefix_actor("residual_mlp", capabilities=ResidualMLPActor.capabilities)
def build_residual_mlp_actor(
    state_dim: int, reference_dim: int, action_dim: int, cfg: Any
) -> ResidualMLPActor:
    """Build a bounded editor from its registered head configuration."""
    return ResidualMLPActor(state_dim, reference_dim, action_dim, cfg)
