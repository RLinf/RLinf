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

"""Rollout-time action sampling for starVLA Qwen Adapter action heads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.distributions.normal import Normal

from ..utils import action_space as action_space_utils
from ..utils import data_pipeline as data_pipeline_utils
from ..utils import state as state_utils
from ..utils import vlm_preprocess as vlm_input_utils
from ..utils.action_heads import (
    build_adapter_backbone_context,
    run_adapter_action_head,
)
from ..utils.backbone_pipeline import run_backbone_pipeline

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def run_rollout_adapter(
    policy: StarVLAForRLActionPrediction,
    *,
    examples: list[dict[str, Any]],
    mode: str,
    calculate_logprobs: bool,
    calculate_values: bool,
    sampling_kwargs: dict[str, Any],
    env_obs: dict[str, Any],
) -> dict[str, Any]:
    """Roll out the Adapter action head and pack replay caches for training."""
    # 1) Prepare state tensor from env observation for adapter head.
    state = state_utils.prepare_state_tensor(
        env_obs.get("states"),
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        warned_keys=policy._warned_once,
        state_adapter_name="adapter",
        context="predict_action_batch_state",
    )

    # 2) Build adapter-style VLM inputs from rollout examples.
    model_inputs = vlm_input_utils.build_adapter_vlm_inputs(
        starvla_model=policy.starvla_model,
        num_action_chunks=policy.num_action_chunks,
        examples=examples,
    )

    # 3) Build adapter context and run backbone with query injection hook.
    adapter_context = build_adapter_backbone_context(
        policy,
        model_inputs=model_inputs,
    )
    backbone_output = run_backbone_pipeline(
        policy,
        model_inputs=model_inputs,
        use_cache=False,
        input_embedding_hook=adapter_context.input_embedding_hook,
    )

    # 4) Run adapter action head to get continuous action mean and critic features.
    mean_actions, critic_features = run_adapter_action_head(
        policy,
        backbone_output=backbone_output,
        model_inputs=model_inputs,
        state=state,
        adapter_context=adapter_context,
    )

    # 5) Build Gaussian actor, sample/greedy action, and clip to env action bounds.
    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))
    sample_actions = bool(sampling_kwargs.get("do_sample")) and mode == "train"
    actions_t = dist.sample() if sample_actions else mean_actions
    actions_exec = action_space_utils.clip_actions_for_env(actions_t)

    # 6) Optionally compute rollout-time logprob/value baselines for training replay.
    prev_logprobs: Optional[torch.Tensor] = None
    prev_values: Optional[torch.Tensor] = None
    if calculate_logprobs:
        # Keep old/new logprobs in the same action space used by training forward.
        actions_for_logprob = data_pipeline_utils.project_rollout_actions_for_logprob(
            policy,
            rollout_actions=actions_exec,
        )
        prev_logprobs = dist.log_prob(actions_for_logprob).to(dtype=torch.float32)
    if calculate_values:
        if policy.value_head is None:
            prev_values = torch.zeros(
                (mean_actions.shape[0], 1),
                device=mean_actions.device,
                dtype=torch.float32,
            )
        else:
            prev_values = policy.value_head(critic_features.float()).to(
                dtype=torch.float32
            )

    # 7) Return env actions and cached tensors used by default_forward training.
    return {
        "output": {
            "normalized_actions": data_pipeline_utils.tensor_to_numpy_compatible(
                actions_exec
            )
        },
        "model_inputs": model_inputs,
        "prev_logprobs": prev_logprobs,
        "prev_values": prev_values,
        "extra_forward_inputs": {},
        "state": state,
    }
