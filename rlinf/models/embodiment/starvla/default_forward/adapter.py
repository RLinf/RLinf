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

"""Training-time forward pass for starVLA Qwen Adapter action heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions.normal import Normal

from ..utils import data_pipeline as data_pipeline_utils
from ..utils.action_heads import build_adapter_backbone_context, run_adapter_action_head
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def run_default_forward_adapter(
    policy: StarVLAForRLActionPrediction,
    *,
    data: dict[str, torch.Tensor],
    compute_logprobs: bool,
    compute_entropy: bool,
    compute_values: bool,
    use_cache: bool,
) -> dict[str, torch.Tensor | None]:
    """Compute training-time PPO terms for the Adapter action head."""
    data_pipeline_utils.ensure_default_forward_replay_batch(data)

    # 1) Resolve optional state tensor from rollout cache.
    state = data.get("state")
    if state is None:
        state = data.get("states")

    model_inputs = data_pipeline_utils.collect_default_forward_model_inputs(
        data,
        skip_keys={"action", "action_tokens"},
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )

    # 2) Run adapter backbone path and produce action-head features.
    adapter_context = build_adapter_backbone_context(
        policy,
        model_inputs=model_inputs,
    )
    backbone_output = run_backbone_pipeline(
        policy,
        model_inputs=model_inputs,
        use_cache=use_cache,
        input_embedding_hook=adapter_context.input_embedding_hook,
    )
    mean_actions, critic_features = run_adapter_action_head(
        policy,
        backbone_output=backbone_output,
        model_inputs=model_inputs,
        state=state,
        adapter_context=adapter_context,
    )

    action = data_pipeline_utils.prepare_actions_for_default_forward(
        policy,
        env_actions=data["action"],
        reference=mean_actions,
    )

    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))
    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }

    # 3) Return requested RL terms only (logprob / entropy / value).
    if compute_logprobs:
        result["logprobs"] = dist.log_prob(action).to(dtype=torch.float32)
    if compute_entropy:
        result["entropy"] = dist.entropy().to(dtype=torch.float32)
    if compute_values:
        if policy.value_head is None:
            result["values"] = torch.zeros(
                (mean_actions.shape[0], 1),
                device=mean_actions.device,
                dtype=torch.float32,
            )
        else:
            result["values"] = policy.value_head(critic_features.float()).to(
                dtype=torch.float32
            )
    return result
