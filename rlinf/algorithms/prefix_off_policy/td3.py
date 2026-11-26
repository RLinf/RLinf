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
from typing import Any

import torch

from rlinf.algorithms.prefix_off_policy.base import (
    PrefixOffPolicyAlgorithm,
    PrefixOffPolicyContext,
)
from rlinf.algorithms.prefix_off_policy.capabilities import (
    PrefixAlgorithmRequirements,
)
from rlinf.algorithms.prefix_off_policy.registry import (
    register_prefix_off_policy_algorithm,
)
from rlinf.models.embodiment.base_policy import ForwardType


@register_prefix_off_policy_algorithm("td3")
class PrefixTD3Algorithm(PrefixOffPolicyAlgorithm):
    """RLT fixed-variance policy learning with TD3 critics and delayed updates."""

    requirements = PrefixAlgorithmRequirements(
        actor_distribution="deterministic",
        requires_action_noise=True,
        delayed_policy_updates=True,
    )

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.actor_agg_q = self._configured_actor_agg_q(default="q1")
        self.target_policy_noise = float(
            cfg.algorithm.get(
                "target_policy_noise",
                cfg.actor.model.actor_head.get("noise_sigma", 0.1),
            )
        )
        self.target_noise_clip = float(cfg.algorithm.get("target_noise_clip", 0.5))
        for name, value in (
            ("target_policy_noise", self.target_policy_noise),
            ("target_noise_clip", self.target_noise_clip),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"algorithm.{name} must be finite and nonnegative, got {value}."
                )

        policy_delay = cfg.algorithm.get("critic_actor_ratio", 2)
        if policy_delay < 1 or int(policy_delay) != policy_delay:
            raise ValueError(
                "algorithm.critic_actor_ratio must be a positive integer, "
                f"got {policy_delay}."
            )
        target_update_freq = cfg.algorithm.get("target_update_freq", policy_delay)
        if target_update_freq != policy_delay:
            raise ValueError(
                "TD3 updates targets with the delayed actor; expected "
                "algorithm.target_update_freq to equal critic_actor_ratio="
                f"{policy_delay}, got {target_update_freq}."
            )
        target_update_type = cfg.algorithm.get("target_update_type", "all")
        if target_update_type != "all":
            raise ValueError(
                "TD3 updates both target networks; expected "
                f"algorithm.target_update_type='all', got {target_update_type!r}."
            )

    def actor_loss(
        self,
        context: PrefixOffPolicyContext,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return the actor objective using shared BC and weight schedules."""
        return self._actor_loss(
            context,
            batch,
            actor_agg_q=self.actor_agg_q,
            apply_reference_dropout=(
                float(self.cfg.algorithm.get("reference_dropout_prob", 0.0)) > 0.0
            ),
            apply_action_noise=bool(
                self.cfg.algorithm.get("actor_update_action_noise", True)
            ),
        )

    def target_actions(
        self,
        context: PrefixOffPolicyContext,
        next_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return target-actor actions with clipped target policy smoothing."""
        actions, _, _ = context.target_model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
            deterministic=True,
            apply_action_noise=False,
        )
        if self.target_policy_noise > 0 and self.target_noise_clip > 0:
            noise = (torch.randn_like(actions) * self.target_policy_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            actions = (actions + noise).clamp(-1.0, 1.0)
        return actions
