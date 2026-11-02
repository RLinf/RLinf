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
from rlinf.models.embodiment.prefix.types import PrefixForwardType


@register_prefix_off_policy_algorithm("td3")
class PrefixTD3Algorithm(PrefixOffPolicyAlgorithm):
    """TD3 semantics over a configurable prefix policy."""

    requirements = PrefixAlgorithmRequirements(
        actor_distribution="deterministic",
        requires_action_noise=True,
        uses_target_actor=True,
    )

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.actor_agg_q = self._configured_actor_agg_q(default="min")

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
        """Sample next-state actions from the target actor."""
        actions, _, _ = context.target_model(
            forward_type=PrefixForwardType.ACTOR,
            obs=next_obs,
        )
        return actions
