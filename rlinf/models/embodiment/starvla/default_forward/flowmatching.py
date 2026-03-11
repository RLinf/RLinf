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

"""Training-time forward pass for starVLA flow-matching action heads."""
# TODO(agent): Flowmatching path is included in the current commit but still
# needs full end-to-end training validation before treating it as fully stable.

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
from ..utils.backbone_pipeline import compute_values_from_hidden, run_backbone_pipeline
from ..utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE

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

    # 2) Validate prompt tensors required for backbone replay.
    if "input_ids" not in data or "attention_mask" not in data:
        raise KeyError(
            "Missing prompt inputs ('input_ids'/'attention_mask') in training batch. "
            "Rollout must cache VLM prompt tensors in forward_inputs."
        )

    flow_prefix = resolve_flowmatching_prefix(action_head_name)
    required_keys = {
        f"{flow_prefix}_chain_actions",
        f"{flow_prefix}_t_bucket_indices",
        f"{flow_prefix}_denoise_inds",
        f"{flow_prefix}_sample_actions",
    }
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(
            f"Missing {action_head_name} rollout cache keys in training batch: {missing}. "
            "Rollout must store these fields in forward_inputs."
        )

    # 3) Rebuild VLM inputs and run backbone once for this training forward.
    model_inputs = data_pipeline_utils.collect_default_forward_model_inputs(
        data,
        skip_keys={
            "action",
            "action_tokens",
            f"{flow_prefix}_chain_actions",
            f"{flow_prefix}_t_bucket_indices",
            f"{flow_prefix}_denoise_inds",
            f"{flow_prefix}_sample_actions",
        },
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )

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

    # 4) Load cached flow-matching tensors from rollout.
    sample_actions_key = f"{flow_prefix}_sample_actions"
    denoise_inds_key = f"{flow_prefix}_denoise_inds"
    head = policy.starvla_model.action_model
    num_steps = max(1, int(getattr(head, "num_inference_timesteps", 16)))

    chain_actions_key = f"{flow_prefix}_chain_actions"
    t_bucket_key = f"{flow_prefix}_t_bucket_indices"
    chain_actions = data[chain_actions_key].to(
        device=rollout_hidden.device, dtype=rollout_hidden.dtype
    )
    t_bucket_indices = data[t_bucket_key].to(
        device=rollout_hidden.device, dtype=torch.long
    )
    denoise_inds = data[denoise_inds_key].to(
        device=rollout_hidden.device,
        dtype=torch.long,
    )

    if chain_actions.ndim != 4:
        raise ValueError(
            f"Expected '{chain_actions_key}' [B,S+1,T,D], got {chain_actions.shape}"
        )
    if t_bucket_indices.ndim != 2:
        raise ValueError(
            f"Expected '{t_bucket_key}' [B,S], got {t_bucket_indices.shape}"
        )

    if num_steps != t_bucket_indices.shape[1]:
        raise ValueError(
            f"num_steps mismatch: head has S={num_steps}, but {t_bucket_key} has {t_bucket_indices.shape[1]} steps"
        )
    if chain_actions.shape[1] != num_steps + 1:
        raise ValueError(
            f"{chain_actions_key} mismatch: expected S+1={num_steps + 1}, got {chain_actions.shape[1]}"
        )
    if denoise_inds.ndim != 2:
        raise ValueError(
            f"Expected '{denoise_inds_key}' [B,S], got {denoise_inds.shape}"
        )
    if denoise_inds.shape != t_bucket_indices.shape:
        raise ValueError(
            f"{denoise_inds_key} mismatch: expected {tuple(t_bucket_indices.shape)}, "
            f"got {tuple(denoise_inds.shape)}"
        )

    # 5) Decide whether this step uses stochastic transition distributions.
    rollout_sample_actions = bool(
        data_pipeline_utils.get_scalar(
            data[sample_actions_key],
            default=1,
            cast=int,
        )
    )
    do_sample = rollout_sample_actions and (compute_logprobs or compute_entropy)

    # 6) Prepare state tensors for head-specific velocity prediction.
    head = policy.starvla_model.action_model
    state = data.get("state")
    if state is None:
        state = data.get("states")
    state = state_utils.prepare_state_tensor(
        state,
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        state_adapter_name=state_adapter_name,
        head=head,
        device=rollout_hidden.device,
        dtype=rollout_hidden.dtype,
        context=state_context,
    )

    # 7) Resolve per-step standard deviation for sampled transitions.
    dt = 1.0 / float(max(1, num_steps))
    resolved_step_std = None
    if do_sample:
        expected_action_dim = int(chain_actions.shape[-1])

        resolved_step_std = (
            torch.exp(policy.actor_logstd)
            .to(
                device=rollout_hidden.device,
                dtype=rollout_hidden.dtype,
            )
            .view(1, 1, -1)
        )
        if resolved_step_std.shape[-1] != expected_action_dim:
            raise RuntimeError(
                "Mismatch between resolved_step_std shape "
                f"{tuple(resolved_step_std.shape)} and expected action_dim "
                f"{expected_action_dim} for sampled flowmatching default_forward."
            )
        resolved_step_std = resolved_step_std * float(sqrt(dt))

    # 8) Compute requested PPO terms.
    step_logprobs: list[torch.Tensor] = []
    step_entropy: list[torch.Tensor] = []
    for step in range(num_steps):
        actions_pre_step = chain_actions[:, step]
        actions_next_step = chain_actions[:, step + 1]
        t_bucket_step = t_bucket_indices[:, step]

        pred_velocity = _predict_velocity(
            policy,
            head=head,
            action_head_inputs=action_head_inputs,
            actions_t=actions_pre_step,
            state_t=state,
            t_bucket_index=t_bucket_step,
        )

        mean_next = actions_pre_step + dt * pred_velocity
        if do_sample:
            active_step_mask = denoise_inds[:, step].eq(step)
            if bool(active_step_mask.any()):
                if resolved_step_std is None:
                    raise RuntimeError(
                        "Internal error: missing step_std for flowmatching sampled transition."
                    )
                dist_step = Normal(mean_next, resolved_step_std.expand_as(mean_next))
                active_step_mask_3d = active_step_mask.view(-1, 1, 1)
                if compute_logprobs:
                    logprob_step = dist_step.log_prob(actions_next_step)
                    logprob_step = torch.where(
                        active_step_mask_3d,
                        logprob_step,
                        torch.zeros_like(logprob_step),
                    )
                    step_logprobs.append(logprob_step)
                if compute_entropy:
                    entropy_step = dist_step.entropy()
                    entropy_step = torch.where(
                        active_step_mask_3d,
                        entropy_step,
                        torch.zeros_like(entropy_step),
                    )
                    step_entropy.append(entropy_step)
            else:
                if compute_logprobs:
                    step_logprobs.append(torch.zeros_like(actions_next_step))
                if compute_entropy:
                    step_entropy.append(torch.zeros_like(actions_next_step))
        else:
            if compute_logprobs:
                step_logprobs.append(torch.zeros_like(actions_next_step))
            if compute_entropy:
                step_entropy.append(torch.zeros_like(actions_next_step))

    # 9) Fill requested RL terms.
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
                (chain_actions.shape[0], 1),
                device=rollout_hidden.device,
                dtype=torch.float32,
            )
        else:
            result["values"] = compute_values_from_hidden(
                value_head=policy.value_head,
                hidden=action_head_inputs.value_hidden,
                attention_mask=backbone_output.attention_mask,
            )
    return result
