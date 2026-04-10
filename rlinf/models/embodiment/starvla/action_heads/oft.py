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

"""Shared OFT handlers for rollout and default_forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import (
    resize_images as starvla_resize_images,
)
from torch.distributions.normal import Normal

from ..utils import data_pipeline as data_pipeline_utils
from ..utils.backbone_pipeline import compute_values_from_hidden, run_backbone_pipeline
from ..utils.profile import (
    RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    resolve_action_chunk_len,
    resolve_vlm_interface,
)

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def _build_oft_vlm_inputs(
    starvla_model,
    *,
    num_action_chunks: int,
    examples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Build OFT prompt format by appending action-token placeholders."""
    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]

    from ..utils.vlm_preprocess import get_train_image_size

    train_obs_image_size = get_train_image_size(starvla_model)
    if train_obs_image_size:
        batch_images = starvla_resize_images(
            batch_images, target_size=train_obs_image_size
        )

    chunk_len = resolve_action_chunk_len(
        starvla_model,
        num_action_chunks,
        action_head_name="oft",
    )
    action_token = str(getattr(starvla_model, "action_token", ""))
    action_tokens = action_token * chunk_len
    prompt_suffix = (
        f" Please predict the next {chunk_len} robot actions: "
        f"<action>{action_tokens}<action>."
    )
    instructions = [instruction + prompt_suffix for instruction in instructions]

    qwen_vl_interface = getattr(starvla_model, "qwen_vl_interface", None)
    build_inputs = getattr(qwen_vl_interface, "build_qwenvl_inputs", None)
    # TODO: Whether this fallback is necessary. need further test
    if not callable(build_inputs):
        vlm_interface = resolve_vlm_interface(starvla_model)
        build_inputs = getattr(vlm_interface, "build_qwenvl_inputs", None)
    if not callable(build_inputs):
        raise RuntimeError("VLM interface does not provide 'build_qwenvl_inputs(...)'.")
    return build_inputs(images=batch_images, instructions=instructions)


def _run_oft_backbone_and_head(
    policy: StarVLAForRLActionPrediction,
    *,
    model_inputs: dict[str, torch.Tensor],
    use_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, Normal]:
    """Run the shared OFT backbone/action-head path."""
    backbone_output = run_backbone_pipeline(
        policy,
        action_head_name="oft",
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    model = policy.starvla_model
    last_hidden = backbone_output["last_hidden"]
    with torch.autocast("cuda", dtype=torch.float32):
        input_ids = model_inputs["input_ids"]
        action_queries = model._gather_action_token_embeddings(
            last_hidden,
            input_ids,
            action_token_id=getattr(model, "action_token_id", None),
        )
        mean_actions = model.action_model.predict_action(action_queries)

    dist = Normal(mean_actions, torch.exp(policy.actor_logstd).view(1, 1, -1))
    return mean_actions, last_hidden, dist


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
    data_pipeline_utils.forward_input_check(data)

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

    mean_actions, last_hidden, dist = _run_oft_backbone_and_head(
        policy,
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    action_for_logprob = (
        data_pipeline_utils.fetch_action_for_logprob_for_default_forward(
            policy,
            data=data,
            reference=mean_actions,
        )
    )

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
        result["values"] = compute_values_from_hidden(
            value_head=policy.value_head,
            hidden=last_hidden,
            attention_mask=model_inputs.get("attention_mask"),
        )
    return result


def run_rollout_oft(
    policy: StarVLAForRLActionPrediction,
    *,
    examples: list[dict[str, Any]],
    mode: str,
    calculate_logprobs: bool,
    calculate_values: bool,
    sampling_kwargs: dict[str, Any],
    env_obs: dict[str, Any],
) -> dict[str, Any]:
    """Roll out the OFT action head and pack replay caches for training."""
    del env_obs

    model_inputs = _build_oft_vlm_inputs(
        starvla_model=policy.starvla_model,
        num_action_chunks=policy.num_action_chunks,
        examples=examples,
    )
    mean_actions, last_hidden, dist = _run_oft_backbone_and_head(
        policy,
        model_inputs=model_inputs,
        use_cache=False,
    )
    sample_actions = bool(sampling_kwargs.get("do_sample")) and mode == "train"
    actions_for_logprob = dist.sample() if sample_actions else mean_actions

    prev_logprobs = None
    prev_values = None
    if calculate_logprobs:
        prev_logprobs = dist.log_prob(actions_for_logprob).to(dtype=torch.float32)
    if calculate_values:
        prev_values = compute_values_from_hidden(
            value_head=policy.value_head,
            hidden=last_hidden,
            attention_mask=model_inputs.get("attention_mask"),
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
        "state": None,
    }
