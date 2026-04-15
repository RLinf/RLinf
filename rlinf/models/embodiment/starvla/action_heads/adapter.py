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

"""Shared Adapter handlers for rollout and default_forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.distributions.normal import Normal

from ..utils import data_pipeline as data_pipeline_utils
from ..utils import state as state_utils
from ..utils import vlm_preprocess as vlm_input_utils
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE, resolve_action_chunk_len

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


# add prefix for adpater action query tokens
def _build_adapter_vlm_inputs(
    policy: StarVLAForRLActionPrediction,
    *,
    examples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Build adapter prompt inputs with dummy action-query placeholders."""
    starvla_model = policy.starvla_model
    chunk_len = resolve_action_chunk_len(
        starvla_model,
        policy.num_action_chunks,
        action_head_name="adapter",
    )

    dummy_action_prompts = str(getattr(starvla_model, "dummy_action_prompt", ""))
    if not dummy_action_prompts:
        action_token = str(getattr(starvla_model, "dummy_action_token", ""))
        action_query_num = int(getattr(starvla_model, "action_query_num", chunk_len))
        dummy_action_prompts = action_query_num * action_token

    prompt_suffix = (
        f" Please predict the next {chunk_len} robot actions: "
        f"<action>{dummy_action_prompts}<action>."
    )
    prompt_examples = [
        {
            **example,
            "lang": f"{example['lang']}{prompt_suffix}",
        }
        for example in examples
    ]
    return vlm_input_utils.build_base_vlm_inputs(
        starvla_model,
        examples=prompt_examples,
    )


def _run_adapter_pipeline(
    policy: StarVLAForRLActionPrediction,
    *,
    model_inputs: dict[str, torch.Tensor],
    state: Optional[torch.Tensor],
    use_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build adapter context, run backbone, and invoke the adapter head."""
    model = policy.starvla_model
    input_ids = model_inputs["input_ids"]
    batch_size = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])
    device = input_ids.device

    # 1) Find the adapter query placeholder positions in each sample so we can
    # inject learned action queries into the VLM embedding stream.
    action_query_num = int(getattr(model, "action_query_num"))
    action_mask = input_ids == model.dummy_action_token_id
    action_positions_tensor = torch.full(
        (batch_size, action_query_num), 0, dtype=torch.long, device=device
    )
    valid_counts = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for batch_idx in range(batch_size):
        act_pos = torch.where(action_mask[batch_idx])[0]
        if act_pos.numel() == action_query_num:
            action_positions_tensor[batch_idx] = act_pos
            valid_counts[batch_idx] = True

    def inject_query_hook(_module, _inputs, output):
        # 2) Replace the placeholder token embeddings with the learned adapter
        # query embeddings right before the backbone forward pass.
        valid = valid_counts.to(device=output.device)
        if not bool(valid.any()):
            return output

        query_embed = model.action_query.to(dtype=output.dtype, device=output.device)
        batch_indices = (
            torch.arange(batch_size, device=output.device)
            .unsqueeze(1)
            .expand(-1, action_query_num)
        )
        output[
            batch_indices[valid],
            action_positions_tensor.to(device=output.device)[valid],
            :,
        ] = query_embed.unsqueeze(0)
        return output

    # 3) Run the shared VLM backbone once with query injection enabled.
    backbone_output = run_backbone_pipeline(
        policy,
        action_head_name="adapter",
        model_inputs=model_inputs,
        use_cache=use_cache,
        input_embedding_hook=inject_query_hook,
    )
    hidden_states = backbone_output["hidden_layers"]

    # 4) Resolve the image token id and recover the contiguous visual token span
    # for each sample from the prompt sequence.
    image_token_id = getattr(model, "_rlinf_image_token_id", None)
    if image_token_id is None:
        token_id = model_inputs.get("image_token_id")
        if isinstance(token_id, torch.Tensor) and token_id.numel() > 0:
            image_token_id = int(token_id.reshape(-1)[0].item())
        else:
            model_cfg = getattr(
                getattr(getattr(model, "qwen_vl_interface", None), "model", None),
                "config",
                None,
            )
            token_id = getattr(model_cfg, "image_token_id", None)
            if token_id is None:
                token_id = getattr(model_cfg, "vision_token_id", None)
            image_token_id = token_id
        if token_id is None:
            raise RuntimeError(
                "Cannot resolve image_token_id for adapter action head. "
                "Expected 'qwen_vl_interface.model.config.image_token_id' "
                "(or 'vision_token_id') on the loaded VLM."
            )
        image_token_id = int(image_token_id)
        # Cache the resolved image token id as a model attribute to avoid repeated lookups.
        setattr(model, "_rlinf_image_token_id", image_token_id)
    else:
        image_token_id = int(image_token_id)

    image_mask = input_ids == image_token_id
    # Check that every sample has at least one image token to avoid invalid gather indices later
    num_tokens_per_sample = image_mask.sum(dim=1)
    if bool((num_tokens_per_sample <= 0).any()):
        bad_idx = (
            torch.nonzero(num_tokens_per_sample <= 0, as_tuple=False).flatten().tolist()
        )
        raise RuntimeError(
            "Adapter action head requires image tokens in 'input_ids', "
            f"but none were found for batch indices {bad_idx} with image_token_id={image_token_id}."
        )
    # Official QwenAdapter.get_image_token_counts computes first_index via
    # cumsum().argmin() which always yields 0 — the pretrained model was trained
    # with text tokens preceding images included in the "vision" feature span.
    # We replicate that here for eval alignment.
    first_index_per_sample = torch.zeros(batch_size, dtype=torch.long, device=device)
    seq_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
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

    # 5) Build batch-aligned gather indices so each hidden layer can be sliced
    # into [vision tokens] + [action query tokens].
    vision_batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_patch_len)
    )
    vision_offsets = (
        torch.arange(max_patch_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    vision_seq_indices = vision_offsets + first_index_per_sample.unsqueeze(1)
    vision_seq_indices = torch.minimum(
        vision_seq_indices, last_index_per_sample.unsqueeze(1)
    )
    vision_padding_mask = vision_offsets >= vision_patch_lengths.unsqueeze(1)
    action_batch_indices = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, action_query_num)
    )

    # 6) For every hidden layer, gather the visual span and the action-query
    # states, then stack them into the adapter head input layout.
    multi_layer_hidden_states = []
    for layer_hidden in hidden_states:
        batch_vision_states = layer_hidden[vision_batch_indices, vision_seq_indices, :]
        batch_vision_states = batch_vision_states.masked_fill(
            vision_padding_mask.unsqueeze(-1), 0.0
        )
        action_query_states = layer_hidden[
            action_batch_indices, action_positions_tensor, :
        ]
        multi_layer_hidden_states.append(
            torch.cat(
                [
                    batch_vision_states.unsqueeze(1),
                    action_query_states.unsqueeze(1),
                ],
                dim=2,
            )
        )

    multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

    # 7) Optionally project proprio/state inputs into the same feature space
    # used by the adapter action head.
    state_projected = None
    state_t = state_utils.prepare_state_tensor(
        state,
        starvla_model=policy.starvla_model,
        default_state_adapter_name=policy.state_adapter_type,
        state_adapter_name="adapter",
        device=multi_layer_hidden_states.device,
        dtype=multi_layer_hidden_states.dtype,
        context="run_adapter_action_head",
    )
    if state_t is not None and getattr(model, "proprio_projector", None) is not None:
        proprio = state_t.squeeze(1)
        state_projected = model.proprio_projector(proprio=proprio)

    # 8) Predict actions from the fused multi-layer vision/query/state features.
    with torch.autocast("cuda", dtype=torch.float32):
        mean_actions = model.action_model.predict_action(
            multi_layer_hidden_states,
            vision_hidden_len=max_patch_len,
            state_projected=state_projected,
            phase=getattr(model, "phase", "Training"),
        )

    # 9) Reuse the fused representation as critic features, blending in the
    # projected state when it has the same feature width.
    critic_features = multi_layer_hidden_states.mean(dim=1).mean(dim=1)
    if (
        state_projected is not None
        and state_projected.shape[-1] == critic_features.shape[-1]
    ):
        critic_features = 0.5 * (
            critic_features + state_projected.to(dtype=critic_features.dtype)
        )
    return mean_actions, critic_features


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
    data_pipeline_utils.forward_input_check(data)

    state = None
    if policy.uses_state_input:
        state = data.get("state")
        if state is None:
            state = data.get("states")

    model_inputs = data_pipeline_utils.collect_default_forward_model_inputs(
        data,
        skip_keys={"action", "action_tokens"},
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )

    mean_actions, critic_features = _run_adapter_pipeline(
        policy,
        model_inputs=model_inputs,
        state=state,
        use_cache=use_cache,
    )
    action_for_logprob = (
        data_pipeline_utils.fetch_action_for_logprob_for_default_forward(
            policy,
            data=data,
            reference=mean_actions,
        )
    )
    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))

    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }
    if compute_logprobs:
        result["logprobs"] = dist.log_prob(action_for_logprob).to(dtype=torch.float32)
    if compute_entropy:
        result["entropy"] = dist.entropy().to(dtype=torch.float32)
    if compute_values:
        if policy.value_head is None:
            result["values"] = torch.zeros(
                (critic_features.shape[0], 1),
                device=critic_features.device,
                dtype=torch.float32,
            )
        else:
            vh_dtype = next(policy.value_head.parameters()).dtype
            result["values"] = policy.value_head(critic_features.to(dtype=vh_dtype)).to(
                dtype=torch.float32
            )
    return result


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
    state = None
    if policy.uses_state_input:
        state = state_utils.prepare_state_tensor(
            env_obs.get("states"),
            starvla_model=policy.starvla_model,
            default_state_adapter_name=policy.state_adapter_type,
            state_adapter_name="adapter",
            context="predict_action_batch_state",
        )

    model_inputs = _build_adapter_vlm_inputs(
        policy,
        examples=examples,
    )

    mean_actions, critic_features = _run_adapter_pipeline(
        policy,
        model_inputs=model_inputs,
        state=state,
        use_cache=False,
    )
    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))

    sample_actions = bool(sampling_kwargs.get("do_sample")) and mode == "train"
    actions_for_logprob = dist.sample() if sample_actions else mean_actions

    prev_logprobs: Optional[torch.Tensor] = None
    prev_values: Optional[torch.Tensor] = None
    if calculate_logprobs:
        prev_logprobs = dist.log_prob(actions_for_logprob).to(dtype=torch.float32)
    if calculate_values:
        if policy.value_head is None:
            prev_values = torch.zeros(
                (critic_features.shape[0], 1),
                device=critic_features.device,
                dtype=torch.float32,
            )
        else:
            vh_dtype = next(policy.value_head.parameters()).dtype
            prev_values = policy.value_head(critic_features.to(dtype=vh_dtype)).to(
                dtype=torch.float32
            )

    return {
        "output": {
            "normalized_actions": data_pipeline_utils.tensor_to_numpy_compatible(
                mean_actions
            )
        },
        "model_inputs": model_inputs,
        "prev_logprobs": prev_logprobs,
        "prev_values": prev_values,
        "extra_forward_inputs": {
            "action_for_logprob": actions_for_logprob.to(dtype=torch.float32)
        },
        "state": state,
    }
