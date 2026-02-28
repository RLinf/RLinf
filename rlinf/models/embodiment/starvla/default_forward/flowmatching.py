# Copyright 2025 The RLinf Authors.
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

"""Training-time forward pass for starVLA flow-matching action heads."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

import torch
from torch.distributions.normal import Normal

from ..utils import data_pipeline as data_pipeline_utils
from ..utils import state as state_utils
from ..utils.action_heads import (
    _predict_velocity,
    build_flowmatching_backbone_context,
    resolve_flowmatching_prefix,
)
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE
from ..utils.backbone_pipeline import compute_values_from_hidden

_FLOWMATCHING_HEADS = {"pi", "gr00t", "dual"}

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def run_default_forward_flowmatching(
    policy: StarVLAForRLActionPrediction,
    *,
    data: dict[str, torch.Tensor],
    compute_logprobs: bool,
    compute_entropy: bool,
    compute_values: bool,
    use_cache: bool,
) -> dict[str, torch.Tensor | None]:
    """Compute training-time PPO terms for PI/GR00T/Dual flowmatching heads."""
    # 1) Resolve runtime profile from policy.
    action_head_name = str(policy.action_head_type).lower()
    if action_head_name not in _FLOWMATCHING_HEADS:
        raise NotImplementedError(
            "run_default_forward_flowmatching only supports flowmatching heads "
            f"{sorted(_FLOWMATCHING_HEADS)}, got action_head_type={action_head_name!r}."
        )
    state_adapter_name = str(policy.state_adapter_type).lower() or action_head_name
    state_context = f"default_forward_{action_head_name}"

    # 2) Validate required rollout caches and prompt tensors.
    if "action" not in data:
        raise KeyError(
            "Missing 'action' in training batch. Rollout must store forward_inputs['action']."
        )
    if "input_ids" not in data or "attention_mask" not in data:
        raise KeyError(
            "Missing prompt inputs ('input_ids'/'attention_mask') in training batch. "
            "Rollout must cache VLM prompt tensors in forward_inputs."
        )

    prefix = resolve_flowmatching_prefix(action_head_name)
    required_keys = {f"{prefix}_chain_actions", f"{prefix}_t_bucket_indices"}
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(
            f"Missing {action_head_name} rollout cache keys in training batch: "
            f"{missing}. Rollout must store these fields in forward_inputs."
        )

    # 3) Rebuild VLM inputs and run backbone once for this training forward.
    flow_skip_keys = {
        "action",
        "action_tokens",
        f"{prefix}_chain_actions",
        f"{prefix}_t_bucket_indices",
        f"{prefix}_num_steps",
        f"{prefix}_sample_actions",
        f"{prefix}_step_std",
    }
    model_inputs = data_pipeline_utils.collect_tensor_inputs(
        data,
        skip_keys=flow_skip_keys,
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )
    model_inputs = data_pipeline_utils.restore_pixel_values_for_forward(model_inputs)

    backbone_output = run_backbone_pipeline(
        policy,
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    action_head_inputs = build_flowmatching_backbone_context(
        policy,
        action_head_name=action_head_name,
        backbone_output=backbone_output,
    )
    rollout_hidden = action_head_inputs.rollout_hidden

    # 4) Load cached flow-matching trajectory tensors from rollout.
    chain_actions_key = f"{prefix}_chain_actions"
    t_bucket_key = f"{prefix}_t_bucket_indices"
    chain_actions = data[chain_actions_key].to(
        device=rollout_hidden.device, dtype=rollout_hidden.dtype
    )
    t_bucket_indices = data[t_bucket_key].to(
        device=rollout_hidden.device, dtype=torch.long
    )

    if chain_actions.ndim != 4:
        raise ValueError(
            f"Expected '{chain_actions_key}' [B,S+1,T,D], got {chain_actions.shape}"
        )
    if t_bucket_indices.ndim != 2:
        raise ValueError(
            f"Expected '{t_bucket_key}' [B,S], got {t_bucket_indices.shape}"
        )

    num_steps_key = f"{prefix}_num_steps"
    sample_actions_key = f"{prefix}_sample_actions"
    step_std_key = f"{prefix}_step_std"
    num_steps = data_pipeline_utils.get_scalar(
        data.get(num_steps_key),
        default=t_bucket_indices.shape[1],
        cast=int,
    )
    if num_steps != t_bucket_indices.shape[1]:
        raise ValueError(
            f"{num_steps_key} mismatch: got {num_steps}, but {t_bucket_key} has {t_bucket_indices.shape[1]} steps"
        )
    if chain_actions.shape[1] != num_steps + 1:
        raise ValueError(
            f"{chain_actions_key} mismatch: expected S+1={num_steps + 1}, got {chain_actions.shape[1]}"
        )

    # 5) Decide whether this step uses stochastic transition distributions.
    rollout_sample_actions = bool(
        data_pipeline_utils.get_scalar(
            data.get(sample_actions_key),
            default=1,
            cast=int,
        )
    )
    if action_head_name == "pi":
        do_sample = rollout_sample_actions and compute_logprobs
    else:
        do_sample = rollout_sample_actions and (compute_logprobs or compute_entropy)

    # 6) Prepare state/action tensors for head-specific velocity prediction.
    head = policy.starvla_model.action_model
    state = data.get("state")
    if state is None:
        state = data.get("states")
    state = state_utils.prepare_state_tensor(
        state,
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        warned_keys=policy._warned_once,
        state_adapter_name=state_adapter_name,
        head=head,
        device=rollout_hidden.device,
        dtype=rollout_hidden.dtype,
        context=state_context,
    )

    action = data["action"]
    if action.ndim == 2:
        bsz = action.shape[0]
        action = action.view(bsz, policy.num_action_chunks, policy.action_dim)
    elif action.ndim != 3:
        raise ValueError(f"Expected 'action' [B, T*D] or [B,T,D], got {action.shape}")
    action = action.to(device=rollout_hidden.device, dtype=rollout_hidden.dtype)

    # 7) Resolve per-step standard deviation for sampled transitions.
    dt = 1.0 / float(max(1, num_steps))
    resolved_step_std = data.get(step_std_key)
    if resolved_step_std is not None:
        if not isinstance(resolved_step_std, torch.Tensor):
            raise TypeError(
                f"Expected '{step_std_key}' to be torch.Tensor, got {type(resolved_step_std)}."
            )
        resolved_step_std = resolved_step_std.to(
            device=rollout_hidden.device,
            dtype=rollout_hidden.dtype,
        )
    elif do_sample:
        resolved_step_std = (
            torch.exp(policy.actor_logstd)
            .to(
                device=rollout_hidden.device,
                dtype=rollout_hidden.dtype,
            )
            .view(1, 1, -1)
        )
        if resolved_step_std.shape[-1] != chain_actions.shape[-1]:
            resolved_step_std = torch.full(
                (1, 1, chain_actions.shape[-1]),
                0.1,
                device=rollout_hidden.device,
                dtype=rollout_hidden.dtype,
            )
        resolved_step_std = resolved_step_std * float(sqrt(dt))
    else:
        resolved_step_std = None

    # 8) Replay rollout chain and accumulate step-wise logprob / entropy.
    step_logprobs: list[torch.Tensor] = []
    step_entropy: list[torch.Tensor] = []
    for step in range(num_steps):
        actions_pre = chain_actions[:, step]
        actions_next = chain_actions[:, step + 1]
        t_bucket_step = t_bucket_indices[:, step]

        pred_velocity = _predict_velocity(
            policy,
            head=head,
            action_head_inputs=action_head_inputs,
            actions_t=actions_pre,
            state_t=state,
            t_bucket_index=t_bucket_step,
        )

        mean_next = actions_pre + dt * pred_velocity
        if do_sample:
            if resolved_step_std is None:
                raise RuntimeError(
                    "Internal error: missing step_std for flowmatching sampled transition."
                )
            dist_step = Normal(mean_next, resolved_step_std.expand_as(mean_next))
            if compute_logprobs:
                step_logprobs.append(dist_step.log_prob(actions_next))
            if compute_entropy:
                step_entropy.append(dist_step.entropy())
        else:
            if compute_logprobs:
                step_logprobs.append(torch.zeros_like(actions_next))
            if compute_entropy:
                step_entropy.append(torch.zeros_like(actions_next))

    # 9) Build RL outputs (policy terms + optional value head).
    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }
    if compute_logprobs:
        result["logprobs"] = (
            torch.stack(step_logprobs, dim=1).sum(dim=1).to(dtype=torch.float32)
        )
    if compute_entropy:
        result["entropy"] = (
            torch.stack(step_entropy, dim=1).sum(dim=1).to(dtype=torch.float32)
        )
    if compute_values:
        if policy.value_head is None:
            result["values"] = torch.zeros(
                (action.shape[0], 1),
                device=action.device,
                dtype=torch.float32,
            )
        else:
            result["values"] = compute_values_from_hidden(
                value_head=policy.value_head,
                hidden=action_head_inputs.value_hidden,
                attention_mask=backbone_output.attention_mask,
            )
    return result
