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


@register_prefix_off_policy_algorithm("ac")
class PrefixACAlgorithm(PrefixOffPolicyAlgorithm):
    """Actor-critic semantics over a configurable prefix policy."""

    requirements = PrefixAlgorithmRequirements(actor_distribution="stochastic")

    def actor_forward_kwargs(self) -> dict[str, object]:
        return {
            "apply_reference_dropout": True,
            "reference_dropout_prob": float(
                self.cfg.algorithm.get("reference_dropout_prob", 0.0)
            ),
        }

    def target_actions(
        self,
        context: PrefixOffPolicyContext,
        next_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        actions, _, _ = context.model(
            forward_type=ForwardType.SAC,
            obs=next_obs,
        )
        return actions

    def aggregate_actor_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        return self._q1(all_q_values)
