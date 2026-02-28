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

"""Training-time forward pass for starVLA Qwen OFT action heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions.normal import Normal

from ..utils import data_pipeline as data_pipeline_utils
from ..utils.action_heads import run_oft_action_head
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE
from ..utils.backbone_pipeline import compute_values_from_hidden

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def run_default_forward_oft(
    policy: StarVLAForRLActionPrediction,
    *,
    data: dict[str, torch.Tensor],
    compute_logprobs: bool,
    compute_entropy: bool,
    compute_values: bool,
    use_cache: bool,
) -> dict[str, torch.Tensor | None]:
    """Compute training-time PPO terms for the OFT action head."""
    # 1) Validate rollout replay tensors required by OFT training path.
    data_pipeline_utils.ensure_default_forward_replay_batch(data)

    # 2) Rebuild model inputs from cached tensors and restore packed vision fields.
    model_inputs = data_pipeline_utils.collect_default_forward_model_inputs(
        data,
        skip_keys={
            "action",
            "action_tokens",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
        },
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )

    # 3) Run shared backbone and map hidden states to OFT action means.
    backbone_output = run_backbone_pipeline(
        policy,
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    mean_actions, last_hidden = run_oft_action_head(
        policy,
        backbone_output=backbone_output,
        model_inputs=model_inputs,
    )

    # 4) Convert env actions into the model-normalized action space.
    action = data_pipeline_utils.prepare_actions_for_default_forward(
        policy,
        env_actions=data["action"],
        reference=mean_actions,
    )

    # 5) Build Gaussian actor and fill requested RL terms.
    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))
    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }

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
            result["values"] = compute_values_from_hidden(
                value_head=policy.value_head,
                hidden=last_hidden,
                attention_mask=model_inputs.get("attention_mask"),
            )
    return result
