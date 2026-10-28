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
    """TD3 semantics over a configurable prefix policy."""

    requirements = PrefixAlgorithmRequirements(
        actor_distribution="deterministic",
        requires_action_noise=True,
        uses_target_actor=True,
    )

    def actor_forward_kwargs(self) -> dict[str, object]:
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        return {
            "apply_reference_dropout": reference_dropout_prob > 0.0,
            "reference_dropout_prob": reference_dropout_prob,
            "apply_action_noise": bool(
                self.cfg.algorithm.get("actor_update_action_noise", True)
            ),
        }

    def target_actions(
        self,
        context: PrefixOffPolicyContext,
        next_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        actions, _, _ = context.target_model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
        )
        return actions

    def aggregate_actor_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        actor_agg_q = self.cfg.algorithm.get("actor_agg_q", "min")
        if actor_agg_q == "min":
            return self._min_twin_q(all_q_values)
        if actor_agg_q == "q1":
            return self._q1(all_q_values)
        if actor_agg_q == "mean":
            self._require_twin_q(all_q_values)
            return torch.mean(all_q_values, dim=-1, keepdim=True)
        raise ValueError(f"Unsupported TD3 actor_agg_q={actor_agg_q!r}.")
