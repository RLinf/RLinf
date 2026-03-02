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

"""Action-head adapters used by the starVLA RLinf wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from . import action_space as action_space_utils
from . import state as state_utils
from .backbone_pipeline import BackboneOutput


@dataclass(frozen=True)
class AdapterBackboneContext:
    """Runtime context needed by Adapter head and its backbone hook."""

    action_positions_tensor: torch.Tensor
    valid_counts: torch.Tensor
    action_query_num: int
    input_embedding_hook: Callable[[Any, Any, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FlowmatchingBackboneContext:
    """Standardized flowmatching head inputs derived from backbone outputs."""

    velocity_mode: str
    rollout_hidden: torch.Tensor
    value_hidden: torch.Tensor
    vl_embs: Optional[torch.Tensor] = None
    vl_embs_list: Optional[tuple[torch.Tensor, ...]] = None


def build_adapter_backbone_context(
    policy,
    *,
    model_inputs: dict[str, torch.Tensor],
) -> AdapterBackboneContext:
    """Find adapter action-query positions and build embedding injection hook."""
    model = policy.starvla_model
    input_ids = model_inputs["input_ids"]
    action_mask = input_ids == model.dummy_action_token_id
    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    action_query_num = int(getattr(model, "action_query_num"))
    action_positions_tensor = torch.full(
        (batch_size, action_query_num),
        0,
        dtype=torch.long,
        device=device,
    )
    valid_counts = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for b in range(batch_size):
        act_pos = torch.where(action_mask[b])[0]
        if len(act_pos) == action_query_num:
            action_positions_tensor[b] = act_pos
            valid_counts[b] = True

    def inject_query_hook(_module, _inputs, output):
        query_embed = model.action_query.to(dtype=output.dtype, device=output.device)
        batch_indices = (
            torch.arange(batch_size, device=output.device)
            .unsqueeze(1)
            .expand(-1, action_query_num)
        )
        valid = valid_counts.to(device=output.device)
        if bool(valid.any()):
            valid_batch_indices = batch_indices[valid]
            valid_action_positions = action_positions_tensor.to(device=output.device)[
                valid
            ]
            output[valid_batch_indices, valid_action_positions, :] = (
                query_embed.unsqueeze(0)
            )
        return output

    return AdapterBackboneContext(
        action_positions_tensor=action_positions_tensor,
        valid_counts=valid_counts,
        action_query_num=action_query_num,
        input_embedding_hook=inject_query_hook,
    )


def build_flowmatching_backbone_context(
    policy,
    *,
    action_head_name: str,
    backbone_output: BackboneOutput,
    action_head_extras: Optional[dict[str, torch.Tensor]] = None,
) -> FlowmatchingBackboneContext:
    """Map backbone outputs into PI/GR00T/Dual flowmatching input format."""
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
        return FlowmatchingBackboneContext(
            velocity_mode="pi",
            rollout_hidden=base_hidden,
            value_hidden=base_hidden,
            vl_embs_list=vl_embs_list,
        )

    if action_head_name == "gr00t":
        return FlowmatchingBackboneContext(
            velocity_mode="gr00t",
            rollout_hidden=backbone_output.last_hidden,
            value_hidden=backbone_output.last_hidden,
            vl_embs=backbone_output.last_hidden,
        )

    if action_head_name == "dual":
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

        dino_features = None
        if action_head_extras:
            dino_features = action_head_extras.get("dino_features")
        if dino_features is None:
            dino_features = backbone_output.extras.get("dino_features")
        if dino_features is not None:
            cond_hidden = torch.cat(
                (
                    cond_hidden,
                    dino_features.to(cond_hidden.device, dtype=cond_hidden.dtype),
                ),
                dim=1,
            )
        return FlowmatchingBackboneContext(
            velocity_mode="gr00t",
            rollout_hidden=cond_hidden,
            value_hidden=backbone_output.last_hidden,
            vl_embs=cond_hidden,
        )

    raise NotImplementedError(
        f"Flow-matching action-head adapter supports only pi/gr00t/dual, got {action_head_name!r}."
    )


def run_oft_action_head(
    policy,
    *,
    backbone_output: BackboneOutput,
    model_inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run OFT action head on last hidden states and return action means."""
    model = policy.starvla_model
    last_hidden = backbone_output.last_hidden
    input_ids = model_inputs["input_ids"]
    action_queries = model._gather_action_token_embeddings(
        last_hidden,
        input_ids,
        action_token_id=getattr(model, "action_token_id", None),
    )
    with torch.autocast("cuda", dtype=torch.float32):
        mean_actions = model.action_model.predict_action(action_queries)
    return mean_actions, last_hidden


def run_adapter_action_head(
    policy,
    *,
    backbone_output: BackboneOutput,
    model_inputs: dict[str, torch.Tensor],
    state: Optional[torch.Tensor],
    adapter_context: AdapterBackboneContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run adapter head with multi-layer vision/query fusion and optional state."""
    model = policy.starvla_model
    input_ids = model_inputs["input_ids"]
    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    hidden_states = backbone_output.hidden_layers
    action_query_num = int(adapter_context.action_query_num)
    action_positions_tensor = adapter_context.action_positions_tensor.to(device=device)
    multi_layer_hidden_states = []
    image_token_id = getattr(model, "_rlinf_image_token_id", None)
    if image_token_id is None:
        token_id = model_inputs.get("image_token_id")
        if isinstance(token_id, torch.Tensor) and token_id.numel() > 0:
            token_id = int(token_id.reshape(-1)[0].item())
        vlm_interface = getattr(model, "qwen_vl_interface", None)
        model_cfg = getattr(getattr(vlm_interface, "model", None), "config", None)
        if token_id is None:
            token_id = getattr(model_cfg, "image_token_id", None)
        if token_id is None:
            token_id = getattr(model_cfg, "vision_token_id", None)
        if token_id is None:
            raise RuntimeError(
                "Cannot resolve image_token_id for adapter action head. "
                "Expected 'qwen_vl_interface.model.config.image_token_id' "
                "(or 'vision_token_id') on the loaded VLM."
            )
        image_token_id = int(token_id)
        setattr(model, "_rlinf_image_token_id", image_token_id)
    else:
        image_token_id = int(image_token_id)

    image_mask = input_ids == image_token_id
    num_tokens_per_sample = image_mask.sum(dim=1)
    if bool((num_tokens_per_sample <= 0).any()):
        bad_idx = (
            torch.nonzero(num_tokens_per_sample <= 0, as_tuple=False).flatten().tolist()
        )
        raise RuntimeError(
            "Adapter action head requires image tokens in 'input_ids', "
            f"but none were found for batch indices {bad_idx} with image_token_id={image_token_id}."
        )
    seq_len = int(input_ids.shape[1])
    seq_indices = torch.arange(
        seq_len, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    first_index_per_sample = (
        torch.where(
            image_mask,
            seq_indices,
            torch.full_like(seq_indices, fill_value=seq_len),
        )
        .min(dim=1)
        .values
    )
    last_index_per_sample = (
        torch.where(
            image_mask,
            seq_indices,
            torch.full_like(seq_indices, fill_value=-1),
        )
        .max(dim=1)
        .values
    )
    vision_patch_lengths = last_index_per_sample - first_index_per_sample + 1
    max_patch_len = int(vision_patch_lengths.max().item())

    for layer_hidden in hidden_states:
        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, max_patch_len)
        )
        seq_indices = (
            torch.arange(max_patch_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        seq_indices = seq_indices + first_index_per_sample.unsqueeze(1)
        seq_indices = torch.clamp(seq_indices, max=last_index_per_sample.unsqueeze(1))
        batch_vision_states = layer_hidden[batch_indices, seq_indices, :]

        padding_mask = torch.arange(max_patch_len, device=device).unsqueeze(
            0
        ) >= vision_patch_lengths.unsqueeze(1)
        batch_vision_states = batch_vision_states.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )

        batch_indices_action = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, action_query_num)
        )
        action_query_states = layer_hidden[
            batch_indices_action, action_positions_tensor, :
        ]

        all_hidden_states = torch.cat(
            [
                batch_vision_states.unsqueeze(1),
                action_query_states.unsqueeze(1),
            ],
            dim=2,
        )
        multi_layer_hidden_states.append(all_hidden_states)

    multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

    state_projected = None
    state_t = state_utils.prepare_state_tensor(
        state,
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        warned_keys=policy._warned_once,
        state_adapter_name="adapter",
        device=multi_layer_hidden_states.device,
        dtype=multi_layer_hidden_states.dtype,
        context="run_adapter_action_head",
    )
    if state_t is not None and getattr(model, "proprio_projector", None) is not None:
        proprio = state_t.squeeze(1)
        state_projected = model.proprio_projector(proprio=proprio)

    with torch.autocast("cuda", dtype=torch.float32):
        predicted_actions = model.action_model.predict_action(
            multi_layer_hidden_states,
            vision_hidden_len=max_patch_len,
            state_projected=state_projected,
            phase=getattr(model, "phase", "Training"),
        )

    critic_features = multi_layer_hidden_states.mean(dim=1).mean(dim=1)
    if (
        state_projected is not None
        and state_projected.shape[-1] == critic_features.shape[-1]
    ):
        critic_features = 0.5 * (
            critic_features + state_projected.to(dtype=critic_features.dtype)
        )
    return predicted_actions, critic_features


_FLOW_PREFIX_BY_ACTION_HEAD = {
    "pi": "pi",
    "gr00t": "gr00t",
    "dual": "dual",
}


def resolve_flowmatching_prefix(action_head_name: str) -> str:
    """Resolve rollout-cache key prefix for flowmatching head type."""
    prefix = _FLOW_PREFIX_BY_ACTION_HEAD.get(action_head_name)
    if prefix is None:
        raise NotImplementedError(
            f"Flow-matching does not support action head {action_head_name}."
        )
    return prefix


def _encode_timestep_safe(dit_model: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    """Encode timestep using dtype-safe path for DiT timestep encoder."""
    timestep_encoder = dit_model.timestep_encoder
    timestep_embedder = timestep_encoder.timestep_embedder

    linear_1 = getattr(timestep_embedder, "linear_1", None)
    weight = getattr(linear_1, "weight", None)
    if weight is not None:
        dtype = weight.dtype
    else:
        proj_weight = getattr(getattr(dit_model, "proj_out_1", None), "weight", None)
        dtype = proj_weight.dtype if proj_weight is not None else torch.float32

    timestep_proj = timestep_encoder.time_proj(timestep).to(dtype=dtype)
    return timestep_embedder(timestep_proj)


def _predict_velocity(
    policy,
    *,
    head: nn.Module,
    action_head_inputs: FlowmatchingBackboneContext,
    actions_t: torch.Tensor,
    state_t: Optional[torch.Tensor],
    t_bucket_index: torch.Tensor,
) -> torch.Tensor:
    """Predict flow velocity for PI/GR00T-style action heads."""
    velocity_mode = action_head_inputs.velocity_mode
    if velocity_mode == "pi":
        if action_head_inputs.vl_embs_list is None:
            raise RuntimeError("Missing vl_embs_list for PI velocity prediction.")
        state_features = None
        if state_t is not None and getattr(head, "state_encoder", None) is not None:
            state_t = state_utils.prepare_state_tensor(
                state_t,
                starvla_model=policy.starvla_model,
                default_state_adapter_name=policy.state_adapter_type,
                warned_keys=policy._warned_once,
                head=head,
                device=actions_t.device,
                dtype=actions_t.dtype,
                context="predict_pi_velocity",
            )
            state_features = head.state_encoder(state_t)

        action_features = head.action_encoder(actions_t, t_bucket_index)
        if getattr(head.config, "add_pos_embed", False):
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions_t.device
            )
            pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(
            actions_t.shape[0], -1, -1
        )
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        temb = _encode_timestep_safe(head.model, t_bucket_index.long())
        model_output = sa_embs
        for layer_idx, layer in enumerate(head.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=action_head_inputs.vl_embs_list[layer_idx],
                temb=temb,
            )
        pred = head.action_decoder(model_output)
        action_horizon = int(getattr(head, "action_horizon", actions_t.shape[1]))
        return pred[:, -action_horizon:]

    if velocity_mode == "gr00t":
        if action_head_inputs.vl_embs is None:
            raise RuntimeError("Missing vl_embs for GR00T-style velocity prediction.")
        state_features = None
        if state_t is not None and getattr(head, "state_encoder", None) is not None:
            state_t = state_utils.prepare_state_tensor(
                state_t,
                starvla_model=policy.starvla_model,
                default_state_adapter_name=policy.state_adapter_type,
                warned_keys=policy._warned_once,
                head=head,
                device=actions_t.device,
                dtype=actions_t.dtype,
                context="predict_gr00t_velocity",
            )
            state_features = head.state_encoder(state_t)

        action_features = head.action_encoder(actions_t, t_bucket_index)
        if getattr(head.config, "add_pos_embed", False):
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions_t.device
            )
            pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(
            actions_t.shape[0], -1, -1
        )
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        temb = _encode_timestep_safe(head.model, t_bucket_index.long())
        model_output = sa_embs
        for idx, layer in enumerate(head.model.transformer_blocks):
            if idx % 2 == 1 and bool(
                getattr(head.model.config, "interleave_self_attention", False)
            ):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=action_head_inputs.vl_embs,
                    encoder_attention_mask=None,
                    temb=temb,
                )

        shift, scale = head.model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        model_output = (
            head.model.norm_out(model_output) * (1 + scale[:, None]) + shift[:, None]
        )
        model_output = head.model.proj_out_2(model_output)
        pred = head.action_decoder(model_output)
        action_horizon = int(getattr(head, "action_horizon", actions_t.shape[1]))
        return pred[:, -action_horizon:]

    raise RuntimeError(
        f"Unsupported velocity_mode={velocity_mode!r} in flowmatching backbone context."
    )


def run_flowmatching_rollout_action_stage(
    policy,
    *,
    head: nn.Module,
    action_head_inputs: FlowmatchingBackboneContext,
    state_t: Optional[torch.Tensor],
    action_horizon: int,
    action_dim: int,
    num_steps: int,
    sample_actions: bool,
    calculate_logprobs: bool,
) -> dict[str, Any]:
    """Roll out flowmatching transitions and collect trajectory caches."""
    rollout_hidden = action_head_inputs.rollout_hidden
    bsz = int(rollout_hidden.shape[0])
    dt = 1.0 / float(max(1, num_steps))

    actions_t = torch.randn(
        (bsz, action_horizon, action_dim),
        dtype=rollout_hidden.dtype,
        device=rollout_hidden.device,
    )
    actions_t = action_space_utils.clip_actions_for_env(actions_t)

    step_std: Optional[torch.Tensor] = None
    if sample_actions:
        step_std = (
            torch.exp(policy.actor_logstd)
            .to(
                device=actions_t.device,
                dtype=actions_t.dtype,
            )
            .view(1, 1, -1)
        )
        if step_std.shape[-1] != action_dim:
            step_std = torch.full(
                (1, 1, action_dim),
                0.1,
                device=actions_t.device,
                dtype=actions_t.dtype,
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
            (bsz,), t_bucket, device=actions_t.device, dtype=torch.long
        )
        t_bucket_indices.append(t_bucket_index)

        pred_velocity = _predict_velocity(
            policy,
            head=head,
            action_head_inputs=action_head_inputs,
            actions_t=actions_t,
            state_t=state_t,
            t_bucket_index=t_bucket_index,
        )
        mean_next = actions_t + dt * pred_velocity

        if sample_actions:
            if step_std is None:
                raise RuntimeError(
                    "Internal error: missing step_std for sampled transition."
                )
            dist_step = Normal(mean_next, step_std.expand_as(mean_next))
            next_actions = action_space_utils.clip_actions_for_env(dist_step.rsample())
            if calculate_logprobs:
                step_logprobs.append(dist_step.log_prob(next_actions))
        else:
            next_actions = action_space_utils.clip_actions_for_env(mean_next)

        actions_t = next_actions
        chain_actions.append(actions_t)

    prev_logprobs: Optional[torch.Tensor] = None
    if calculate_logprobs:
        if step_logprobs:
            prev_logprobs = (
                torch.stack(step_logprobs, dim=1).sum(dim=1).to(dtype=torch.float32)
            )
        else:
            prev_logprobs = torch.zeros_like(actions_t, dtype=torch.float32)

    return {
        "actions_t": actions_t,
        "chain_actions": torch.stack(chain_actions, dim=1),
        "t_bucket_indices": torch.stack(t_bucket_indices, dim=1),
        "num_steps": int(num_steps),
        "sample_actions": bool(sample_actions),
        "step_std": step_std,
        "prev_logprobs": prev_logprobs,
    }
