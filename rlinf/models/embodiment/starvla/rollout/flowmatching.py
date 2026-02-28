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

"""Rollout-time action sampling for starVLA flow-matching action heads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from ..utils import data_pipeline as data_pipeline_utils
from ..utils import state as state_utils
from ..utils import vlm_preprocess as vlm_input_utils
from ..utils.action_heads import (
    build_flowmatching_backbone_context,
    resolve_flowmatching_prefix,
    run_flowmatching_rollout_action_stage,
)
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.backbone_pipeline import compute_values_from_hidden

_FLOWMATCHING_HEADS = {"pi", "gr00t", "dual"}

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def run_rollout_flowmatching(
    policy: StarVLAForRLActionPrediction,
    *,
    examples: list[dict[str, Any]],
    mode: str,
    calculate_logprobs: bool,
    calculate_values: bool,
    sampling_kwargs: dict[str, Any],
    env_obs: dict[str, Any],
) -> dict[str, Any]:
    """Roll out flowmatching actions and pack replay caches for training."""
    # 1) Resolve runtime profile from policy.
    action_head_name = str(policy.action_head_type).lower()
    if action_head_name not in _FLOWMATCHING_HEADS:
        raise NotImplementedError(
            "run_rollout_flowmatching only supports flowmatching heads "
            f"{sorted(_FLOWMATCHING_HEADS)}, got action_head_type={action_head_name!r}."
        )
    state_adapter_name = str(policy.state_adapter_type).lower() or action_head_name
    state_context = f"rollout_{action_head_name}"
    raw_state = env_obs.get("states")

    # 2) Run shared backbone once per rollout call.
    backbone_output = run_backbone_pipeline(
        policy,
        examples=examples,
        use_cache=False,
    )

    # 3) Prepare action-head specific extras before building head context.
    action_head_extras: dict[str, torch.Tensor] = {}
    if action_head_name == "dual":
        dino_features = vlm_input_utils.build_dual_dino_features(
            policy.starvla_model,
            examples=examples,
        )
        action_head_extras["dino_features"] = dino_features
        backbone_output.model_inputs["dino_features"] = dino_features

    action_head_inputs = build_flowmatching_backbone_context(
        policy,
        action_head_name=action_head_name,
        backbone_output=backbone_output,
        action_head_extras=action_head_extras,
    )
    rollout_hidden = action_head_inputs.rollout_hidden

    # 4) Build state tensor for the selected head on the same device/dtype as rollout hidden.
    head = policy.starvla_model.action_model
    state = state_utils.prepare_state_tensor(
        raw_state,
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        warned_keys=policy._warned_once,
        state_adapter_name=state_adapter_name,
        head=head,
        device=rollout_hidden.device,
        dtype=rollout_hidden.dtype,
        context=state_context,
    )

    # 5) Execute rollout sampling/integration stage for the flow-matching head.
    action_horizon = int(getattr(head, "action_horizon", policy.num_action_chunks))
    action_dim = int(getattr(head, "action_dim", policy.action_dim))
    num_steps = max(1, int(getattr(head, "num_inference_timesteps", 16)))
    sample_actions = bool(sampling_kwargs.get("do_sample")) and mode == "train"

    rollout_result = run_flowmatching_rollout_action_stage(
        policy,
        head=head,
        action_head_inputs=action_head_inputs,
        state_t=state,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_steps=num_steps,
        sample_actions=sample_actions,
        calculate_logprobs=calculate_logprobs,
    )
    actions_t = rollout_result["actions_t"]

    # 6) Compute optional value baseline from cached value_hidden.
    prev_values: Optional[torch.Tensor] = None
    if calculate_values:
        if policy.value_head is None:
            prev_values = torch.zeros(
                (actions_t.shape[0], 1), device=actions_t.device, dtype=torch.float32
            )
        else:
            prev_values = compute_values_from_hidden(
                value_head=policy.value_head,
                hidden=action_head_inputs.value_hidden,
                attention_mask=backbone_output.attention_mask,
            )

    # 7) Pack rollout caches that default_forward(flowmatching) will replay in training.
    flow_prefix = resolve_flowmatching_prefix(action_head_name)
    chain_actions = rollout_result["chain_actions"]
    t_bucket_indices = rollout_result["t_bucket_indices"]
    if chain_actions.ndim != 4:
        raise ValueError(
            f"Expected chain_actions [B,S+1,T,D], got {tuple(chain_actions.shape)}"
        )
    if t_bucket_indices.ndim != 2:
        raise ValueError(
            f"Expected t_bucket_indices [B,S], got {tuple(t_bucket_indices.shape)}"
        )

    flow_cache: dict[str, torch.Tensor] = {
        f"{flow_prefix}_chain_actions": chain_actions,
        f"{flow_prefix}_t_bucket_indices": t_bucket_indices,
        f"{flow_prefix}_num_steps": torch.tensor(
            int(rollout_result["num_steps"]),
            device=chain_actions.device,
            dtype=torch.int64,
        ),
        f"{flow_prefix}_sample_actions": torch.tensor(
            int(rollout_result["sample_actions"]),
            device=chain_actions.device,
            dtype=torch.int64,
        ),
    }
    step_std = rollout_result["step_std"]
    if step_std is not None:
        flow_cache[f"{flow_prefix}_step_std"] = step_std

    # 8) Return env actions + training replay payload.
    output = {
        "normalized_actions": data_pipeline_utils.tensor_to_numpy_compatible(actions_t)
    }
    return {
        "output": output,
        "model_inputs": backbone_output.model_inputs,
        "prev_logprobs": rollout_result["prev_logprobs"],
        "prev_values": prev_values,
        "extra_forward_inputs": flow_cache,
        "state": state,
    }
