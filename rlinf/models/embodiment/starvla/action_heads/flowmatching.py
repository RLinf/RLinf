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

"""Shared flowmatching handlers for rollout and default_forward."""
# TODO(agent): Flowmatching path is included in the current commit but still
# needs full end-to-end training validation before treating it as fully stable.

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ..utils import action_space as action_space_utils
from ..utils import data_pipeline as data_pipeline_utils
from ..utils import state as state_utils
from ..utils import vlm_preprocess as vlm_input_utils
from ..utils.backbone_pipeline import (
    BackboneOutput,
    compute_values_from_hidden,
    run_backbone_pipeline,
)
from ..utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction

_FLOW_PREFIX_BY_ACTION_HEAD = {
    "pi": "pi",
    "gr00t": "gr00t",
    "dual": "dual",
}


def _build_dual_dino_features(
    starvla_model,
    *,
    examples: list[dict[str, Any]],
) -> torch.Tensor:
    """Extract DINO wrist-view features for Dual action head conditioning."""
    wrist_views = []
    for ex in examples:
        if "wrist_images" in ex:
            wrist = ex["wrist_images"]
            if isinstance(wrist, (list, tuple)):
                wrist_views.append(
                    [vlm_input_utils.to_pil_preserve(img) for img in wrist]
                )
            else:
                wrist_views.append([vlm_input_utils.to_pil_preserve(wrist)])
        else:
            fallback_views = vlm_input_utils.to_pil_preserve(ex["image"])
            if isinstance(fallback_views, (list, tuple)):
                wrist_views.append(
                    [vlm_input_utils.to_pil_preserve(img) for img in fallback_views]
                )
            else:
                wrist_views.append([vlm_input_utils.to_pil_preserve(fallback_views)])

    train_size = vlm_input_utils.get_train_image_size(starvla_model)
    if train_size:
        wrist_views = [
            vlm_input_utils.resize_images(ws, target_size=train_size)
            for ws in wrist_views
        ]

    dino_encoder = getattr(starvla_model, "dino_encoder", None)
    dino_pro = getattr(starvla_model, "dino_pro", None)
    if dino_encoder is None or dino_pro is None:
        raise RuntimeError(
            "Dual action head requires dino_encoder and dino_pro on the model."
        )

    flat_views = [img for ws in wrist_views for img in ws]
    dino_input = dino_encoder.prepare_dino_input(flat_views)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dino_feats = dino_encoder(dino_input)

    bsz = len(examples)
    dino_feats = dino_feats.reshape(bsz, -1, dino_feats.shape[-1])
    return dino_pro(dino_feats)


def _resolve_flowmatching_head_profile(
    policy: StarVLAForRLActionPrediction,
) -> tuple[str, str, nn.Module]:
    """Resolve the active flowmatching head profile shared by rollout/training."""
    action_head_name = str(policy.action_head_type).lower()
    if action_head_name not in _FLOW_PREFIX_BY_ACTION_HEAD:
        raise NotImplementedError(
            "Flowmatching handler only supports action heads "
            f"{sorted(_FLOW_PREFIX_BY_ACTION_HEAD)}, got action_head_type={action_head_name!r}."
        )
    flow_prefix = _FLOW_PREFIX_BY_ACTION_HEAD[action_head_name]
    head = policy.starvla_model.action_model
    return action_head_name, flow_prefix, head


def _finalize_flowmatching_context(
    policy: StarVLAForRLActionPrediction,
    *,
    action_head_name: str,
    flow_prefix: str,
    head: nn.Module,
    backbone_output: BackboneOutput,
    state_source: Optional[torch.Tensor],
    state_adapter_name: str,
    state_context: str,
) -> dict[str, Any]:
    """Build head-specific hidden/state tensors and return unified runtime context."""
    # Convert backbone outputs into the hidden-state layout expected by each flowmatching head.
    if action_head_name == "pi":
        expected_layers = len(
            policy.starvla_model.action_model.model.transformer_blocks
        )
        if len(backbone_output.hidden_layers) < expected_layers:
            raise RuntimeError(
                "Backbone does not provide enough hidden layers for PI action head: "
                f"need {expected_layers}, got {len(backbone_output.hidden_layers)}. "
                "This backbone cannot drive layer-wise PI head as configured."
            )
        vl_embs_list = tuple(backbone_output.hidden_layers[-expected_layers:])
        base_hidden = vl_embs_list[-1]
        action_head_inputs: dict[str, Any] = {
            "velocity_mode": "pi",
            "rollout_hidden": base_hidden,
            "value_hidden": base_hidden,
            "vl_embs": None,
            "vl_embs_list": vl_embs_list,
        }
    elif action_head_name == "gr00t":
        action_head_inputs = {
            "velocity_mode": "gr00t",
            "rollout_hidden": backbone_output.last_hidden,
            "value_hidden": backbone_output.last_hidden,
            "vl_embs": backbone_output.last_hidden,
            "vl_embs_list": None,
        }
    elif action_head_name == "dual":
        cfg = getattr(
            getattr(getattr(policy.starvla_model, "config", None), "framework", None),
            "action_model",
            None,
        )
        connect_idx = int(getattr(cfg, "connect_layer_index", -1))
        try:
            cond_hidden = backbone_output.hidden_layers[connect_idx]
        except IndexError as exc:
            raise RuntimeError(
                f"Invalid connect_layer_index={connect_idx} for hidden_layers size "
                f"{len(backbone_output.hidden_layers)}."
            ) from exc

        dino_features = backbone_output.model_inputs.get("dino_features")
        if isinstance(dino_features, torch.Tensor):
            cond_hidden = torch.cat(
                (
                    cond_hidden,
                    dino_features.to(cond_hidden.device, dtype=cond_hidden.dtype),
                ),
                dim=1,
            )
        action_head_inputs = {
            "velocity_mode": "gr00t",
            "rollout_hidden": cond_hidden,
            "value_hidden": backbone_output.last_hidden,
            "vl_embs": cond_hidden,
            "vl_embs_list": None,
        }
    else:
        raise NotImplementedError(
            "Flowmatching context builder only supports action heads "
            f"{sorted(_FLOW_PREFIX_BY_ACTION_HEAD)}, got action_head_type={action_head_name!r}."
        )

    # Normalize proprio/state inputs onto the same device/dtype as the rollout hidden states.
    rollout_hidden = action_head_inputs["rollout_hidden"]
    state = state_utils.prepare_state_tensor(
        state_source,
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        state_adapter_name=state_adapter_name,
        head=head,
        device=rollout_hidden.device,
        dtype=rollout_hidden.dtype,
        context=state_context,
    )
    return {
        "action_head_name": action_head_name,
        "flow_prefix": flow_prefix,
        "head": head,
        "backbone_output": backbone_output,
        "action_head_inputs": action_head_inputs,
        "rollout_hidden": rollout_hidden,
        "state": state,
    }


def _predict_velocity(
    policy: StarVLAForRLActionPrediction,
    *,
    head: nn.Module,
    action_head_inputs: dict[str, Any],
    actions_t: torch.Tensor,
    state_t: Optional[torch.Tensor],
    t_bucket_index: torch.Tensor,
) -> torch.Tensor:
    """Predict flow velocity for PI/GR00T-style action heads."""
    # Select the correct visual conditioning view for the active flowmatching head.
    velocity_mode = str(action_head_inputs["velocity_mode"])
    if velocity_mode == "pi":
        vl_ctx = action_head_inputs.get("vl_embs_list")
        if vl_ctx is None:
            raise RuntimeError("Missing vl_embs_list for PI velocity prediction.")
        state_ctx = "predict_pi_velocity"
    elif velocity_mode == "gr00t":
        vl_ctx = action_head_inputs.get("vl_embs")
        if vl_ctx is None:
            raise RuntimeError("Missing vl_embs for GR00T-style velocity prediction.")
        state_ctx = "predict_gr00t_velocity"
    else:
        raise NotImplementedError(
            f"Unsupported velocity_mode={velocity_mode!r} in flowmatching backbone context."
        )

    # Optionally encode proprio/state features when the head exposes a state encoder.
    state_features = None
    if state_t is not None and getattr(head, "state_encoder", None) is not None:
        state_t = state_utils.prepare_state_tensor(
            state_t,
            starvla_model=policy.starvla_model,
            default_state_adapter_name=policy.state_adapter_type,
            head=head,
            device=actions_t.device,
            dtype=actions_t.dtype,
            context=state_ctx,
        )
        state_features = head.state_encoder(state_t)

    # Build the action-token sequence consumed by the DiT blocks.
    action_features = head.action_encoder(actions_t, t_bucket_index)
    if getattr(head.config, "add_pos_embed", False):
        pos_ids = torch.arange(
            action_features.shape[1],
            dtype=torch.long,
            device=actions_t.device,
        )
        action_features = action_features + head.position_embedding(pos_ids).unsqueeze(
            0
        )

    future_tokens = head.future_tokens.weight.unsqueeze(0).expand(
        actions_t.shape[0], -1, -1
    )
    sa_embs = (
        torch.cat((state_features, future_tokens, action_features), dim=1)
        if state_features is not None
        else torch.cat((future_tokens, action_features), dim=1)
    )

    # Encode the denoising timestep using the head's dtype-safe timestep path.
    timestep_encoder = head.model.timestep_encoder
    timestep_embedder = timestep_encoder.timestep_embedder
    weight = getattr(getattr(timestep_embedder, "linear_1", None), "weight", None)
    if weight is None:
        weight = getattr(getattr(head.model, "proj_out_1", None), "weight", None)
    timestep_dtype = weight.dtype if isinstance(weight, torch.Tensor) else torch.float32
    timestep_proj = timestep_encoder.time_proj(t_bucket_index.long()).to(
        dtype=timestep_dtype
    )
    temb = timestep_embedder(timestep_proj)

    # Run the DiT stack in the head-specific conditioning mode.
    model_output = sa_embs
    if velocity_mode == "pi":
        assert isinstance(vl_ctx, tuple)
        for layer_idx, layer in enumerate(head.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_ctx[layer_idx],
                temb=temb,
            )
    else:
        assert isinstance(vl_ctx, torch.Tensor)
        interleave = bool(
            getattr(head.model.config, "interleave_self_attention", False)
        )
        for idx, layer in enumerate(head.model.transformer_blocks):
            # If interleaving is enabled, odd layers attend only to the SA embeddings
            # and even layers attend to the visual context.
            if idx % 2 == 1 and interleave:
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_ctx,
                    encoder_attention_mask=None,
                    temb=temb,
                )
        # Apply FiLM conditioning from the timestep embedding after the DiT stack.
        shift, scale = head.model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        model_output = (
            head.model.norm_out(model_output) * (1 + scale[:, None]) + shift[:, None]
        )
        model_output = head.model.proj_out_2(model_output)

    # Decode the predicted velocity and keep only the action-horizon suffix.
    pred = head.action_decoder(model_output)
    action_horizon = int(getattr(head, "action_horizon", actions_t.shape[1]))
    return pred[:, -action_horizon:]


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
    action_head_name, flow_prefix, head = _resolve_flowmatching_head_profile(policy)

    # Validate that rollout cached prompt tensors and stochastic replay metadata.
    if "input_ids" not in data or "attention_mask" not in data:
        raise KeyError(
            "Missing prompt inputs ('input_ids'/'attention_mask') in training batch. "
            "Rollout must cache VLM prompt tensors in forward_inputs."
        )

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

    # Rebuild VLM inputs from replay and re-run the shared backbone once for training.
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
    state_source = data.get("state")
    if state_source is None:
        state_source = data.get("states")
    state_adapter_name = str(policy.state_adapter_type).lower() or action_head_name
    flow_ctx = _finalize_flowmatching_context(
        policy,
        action_head_name=action_head_name,
        flow_prefix=flow_prefix,
        head=head,
        backbone_output=backbone_output,
        state_source=state_source,
        state_adapter_name=state_adapter_name,
        state_context=f"default_forward_{action_head_name}",
    )
    flow_prefix = flow_ctx["flow_prefix"]
    head = flow_ctx["head"]
    backbone_output = flow_ctx["backbone_output"]
    action_head_inputs = flow_ctx["action_head_inputs"]
    rollout_hidden = flow_ctx["rollout_hidden"]
    state = flow_ctx["state"]

    sample_actions_key = f"{flow_prefix}_sample_actions"
    denoise_inds_key = f"{flow_prefix}_denoise_inds"
    num_steps = max(1, int(getattr(head, "num_inference_timesteps", 16)))

    chain_actions_key = f"{flow_prefix}_chain_actions"
    t_bucket_key = f"{flow_prefix}_t_bucket_indices"
    chain_actions = data[chain_actions_key].to(
        device=rollout_hidden.device,
        dtype=rollout_hidden.dtype,
    )
    t_bucket_indices = data[t_bucket_key].to(
        device=rollout_hidden.device,
        dtype=torch.long,
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

    rollout_sample_actions = bool(
        data_pipeline_utils.get_scalar(
            data[sample_actions_key],
            default=1,
            cast=int,
        )
    )
    do_sample = rollout_sample_actions and (compute_logprobs or compute_entropy)

    dt = 1.0 / float(max(1, num_steps))
    resolved_step_std = None
    if do_sample:
        expected_action_dim = int(chain_actions.shape[-1])
        resolved_step_std = (
            torch.exp(policy.actor_logstd)
            .to(device=rollout_hidden.device, dtype=rollout_hidden.dtype)
            .view(1, 1, -1)
        )
        if resolved_step_std.shape[-1] != expected_action_dim:
            raise RuntimeError(
                "Mismatch between resolved_step_std shape "
                f"{tuple(resolved_step_std.shape)} and expected action_dim "
                f"{expected_action_dim} for sampled flowmatching default_forward."
            )
        resolved_step_std = resolved_step_std * float(sqrt(dt))

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
                hidden=action_head_inputs["value_hidden"],
                attention_mask=backbone_output.attention_mask,
            )
    return result


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
    action_head_name, flow_prefix, head = _resolve_flowmatching_head_profile(policy)

    # Run the backbone on live examples and attach any head-specific extras needed downstream.
    backbone_output = run_backbone_pipeline(
        policy,
        examples=examples,
        use_cache=False,
    )
    if action_head_name == "dual":
        dino_features = _build_dual_dino_features(
            policy.starvla_model,
            examples=examples,
        )
        backbone_output.model_inputs["dino_features"] = dino_features

    flow_ctx = _finalize_flowmatching_context(
        policy,
        action_head_name=action_head_name,
        flow_prefix=flow_prefix,
        head=head,
        backbone_output=backbone_output,
        state_source=env_obs.get("states"),
        state_adapter_name=str(policy.state_adapter_type).lower(),
        state_context=f"rollout_{action_head_name}",
    )
    flow_prefix = flow_ctx["flow_prefix"]
    head = flow_ctx["head"]
    backbone_output = flow_ctx["backbone_output"]
    action_head_inputs = flow_ctx["action_head_inputs"]
    rollout_hidden = flow_ctx["rollout_hidden"]
    state = flow_ctx["state"]

    action_horizon = int(getattr(head, "action_horizon", policy.num_action_chunks))
    action_dim = int(getattr(head, "action_dim", policy.action_dim))
    num_steps = max(1, int(getattr(head, "num_inference_timesteps", 16)))
    sample_actions = bool(sampling_kwargs.get("do_sample")) and mode == "train"

    batch_size = int(rollout_hidden.shape[0])
    if sample_actions:
        chosen_step = int(
            torch.randint(
                low=0,
                high=max(1, num_steps),
                size=(1,),
                device=rollout_hidden.device,
            ).item()
        )
        denoise_inds = torch.full(
            (batch_size, num_steps),
            fill_value=chosen_step,
            device=rollout_hidden.device,
            dtype=torch.long,
        )
    else:
        denoise_inds = torch.full(
            (batch_size, num_steps),
            fill_value=-1,
            device=rollout_hidden.device,
            dtype=torch.long,
        )

    dt = 1.0 / float(max(1, num_steps))
    actions_t = torch.randn(
        (batch_size, action_horizon, action_dim),
        dtype=rollout_hidden.dtype,
        device=rollout_hidden.device,
    )

    step_std = None
    if sample_actions:
        step_std = (
            torch.exp(policy.actor_logstd)
            .to(device=actions_t.device, dtype=actions_t.dtype)
            .view(1, 1, -1)
        )
        if step_std.shape[-1] != action_dim:
            raise RuntimeError(
                f"Mismatch between resolved step_std shape {step_std.shape} and expected action_dim {action_dim} for sampled flowmatching rollout."
            )
        step_std = step_std * float(dt**0.5)

    chain_actions: list[torch.Tensor] = [actions_t]
    t_bucket_indices: list[torch.Tensor] = []
    step_logprobs: list[torch.Tensor] = []
    num_timestep_buckets = int(getattr(head, "num_timestep_buckets", 1000))

    for step in range(num_steps):
        t_continuous = step / float(max(1, num_steps))
        t_bucket = int(t_continuous * num_timestep_buckets)
        t_bucket_index = torch.full(
            (batch_size,),
            t_bucket,
            device=actions_t.device,
            dtype=torch.long,
        )
        t_bucket_indices.append(t_bucket_index)

        pred_velocity = _predict_velocity(
            policy,
            head=head,
            action_head_inputs=action_head_inputs,
            actions_t=actions_t,
            state_t=state,
            t_bucket_index=t_bucket_index,
        )
        mean_next = actions_t + dt * pred_velocity

        if sample_actions:
            if step_std is None:
                raise RuntimeError(
                    "Internal error: missing step_std for sampled transition."
                )
            active_step_mask = denoise_inds[:, step].eq(step)
            if bool(active_step_mask.any()):
                dist_step = Normal(mean_next, step_std.expand_as(mean_next))
                sampled_actions = dist_step.rsample()
                active_step_mask_3d = active_step_mask.view(-1, 1, 1)
                next_actions_preclip = torch.where(
                    active_step_mask_3d,
                    sampled_actions,
                    mean_next,
                )
                if calculate_logprobs:
                    logprob_step = dist_step.log_prob(next_actions_preclip)
                    logprob_step = torch.where(
                        active_step_mask_3d,
                        logprob_step,
                        torch.zeros_like(logprob_step),
                    )
                    step_logprobs.append(logprob_step)
            else:
                next_actions_preclip = mean_next
        else:
            next_actions_preclip = mean_next

        actions_t = next_actions_preclip
        chain_actions.append(actions_t)

    prev_logprobs: Optional[torch.Tensor] = None
    if calculate_logprobs:
        if step_logprobs:
            prev_logprobs = (
                torch.stack(step_logprobs, dim=1).sum(dim=1).to(dtype=torch.float32)
            )
        else:
            prev_logprobs = torch.zeros_like(actions_t, dtype=torch.float32)

    actions_exec = action_space_utils.clip_actions_for_env(actions_t)

    prev_values: Optional[torch.Tensor] = None
    if calculate_values:
        if policy.value_head is None:
            prev_values = torch.zeros(
                (actions_exec.shape[0], 1),
                device=actions_exec.device,
                dtype=torch.float32,
            )
        else:
            prev_values = compute_values_from_hidden(
                value_head=policy.value_head,
                hidden=action_head_inputs["value_hidden"],
                attention_mask=backbone_output.attention_mask,
            )

    flow_cache: dict[str, torch.Tensor] = {
        f"{flow_prefix}_chain_actions": torch.stack(chain_actions, dim=1),
        f"{flow_prefix}_t_bucket_indices": torch.stack(t_bucket_indices, dim=1),
        f"{flow_prefix}_denoise_inds": denoise_inds,
        f"{flow_prefix}_sample_actions": torch.tensor(
            int(sample_actions),
            device=actions_t.device,
            dtype=torch.int64,
        ),
    }
    return {
        "output": {
            "normalized_actions": data_pipeline_utils.tensor_to_numpy_compatible(
                actions_exec
            )
        },
        "model_inputs": backbone_output.model_inputs,
        "prev_logprobs": prev_logprobs,
        "prev_values": prev_values,
        "extra_forward_inputs": flow_cache,
        "state": state,
    }
