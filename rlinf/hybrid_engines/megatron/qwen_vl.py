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

"""Bridge Qwen-VL output adapter for response-token scoring."""

import math
from functools import partial

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from .lm_head import chunked_vocab_parallel_log_probs


def validate_qwen_vl_chunked_cfg(cfg: DictConfig) -> None:
    """Validate the experimental Qwen3.5 Actor mode before model initialization."""
    size = cfg.actor.model.get("lm_head_chunk_size", 0)
    if type(size) is not int or size < 0:
        raise ValueError(
            f"actor.model.lm_head_chunk_size must be a nonnegative integer, got {size!r}"
        )
    if size == 0:
        return
    if cfg.actor.get("training_backend") != "megatron" or not OmegaConf.select(
        cfg, "agentloop.is_dynamic_rollout_batch", default=False
    ):
        raise ValueError(
            "Chunked LM-head scoring requires a dynamic-batch Megatron Actor"
        )
    if (
        not cfg.actor.megatron.get("mbridge", False)
        or cfg.actor.model.model_type != "qwen3_5"
    ):
        raise ValueError("Qwen-VL chunked scoring requires MBridge Qwen3.5")
    if (
        cfg.actor.model.pipeline_model_parallel_size != 1
        or cfg.actor.model.get("context_parallel_size", 1) != 1
    ):
        raise ValueError("Qwen-VL chunked scoring currently requires PP1 and CP1")
    if cfg.algorithm.get("entropy_bonus", 0) != 0:
        raise ValueError(
            "Qwen-VL chunked scoring supports entropy metrics only; entropy_bonus must be zero"
        )
    if cfg.actor.model.get("load_mtp", True):
        raise ValueError("Qwen-VL chunked scoring requires actor.model.load_mtp=False")
    for option in (
        "fp8",
        "mtp_num_layers",
        "defer_embedding_wgrad_compute",
        "use_mup",
        "enable_autocast",
    ):
        if cfg.actor.model.get(option, None):
            raise ValueError(
                f"Qwen-VL chunked scoring does not support actor.model.{option}"
            )
    temperature = cfg.algorithm.sampling_params.temperature
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Qwen-VL chunked scoring requires a finite positive temperature"
        )


def forward_qwen_vl_with_chunked_log_probs(
    model: torch.nn.Module,
    unwrapped_model: torch.nn.Module,
    *,
    label: torch.Tensor,
    label_mask: torch.Tensor,
    chunk_size: int,
    temperature: float,
    calculate_entropy: bool = False,
    **model_inputs,
) -> dict[str, torch.Tensor]:
    """Score Qwen-VL decoder output while preserving its multimodal forward.

    This adapter binds Core 0.18's output processor to Bridge 0.5 Qwen3VLGPTModel.
    Inputs and labels must already follow the caller's
    packing transform. It supports training scoring on the last pipeline stage;
    labels, inference contexts and MTP in the wrapped forward are unsupported.
    The caller must keep output-position hooks disabled during this forward.
    """
    language_model = unwrapped_model.language_model
    if not (unwrapped_model.post_process and language_model.post_process):
        raise ValueError("Qwen-VL chunked scoring requires the last pipeline stage")
    if language_model.mtp_process or language_model.config.mtp_num_layers:
        raise ValueError("Qwen-VL chunked scoring does not support MTP")
    if (
        language_model.config.use_mup
        or language_model.config.defer_embedding_wgrad_compute
    ):
        raise ValueError(
            "Qwen-VL chunked scoring does not support MuP or deferred embedding gradients"
        )
    if language_model.config.fp8:
        raise ValueError("Qwen-VL chunked scoring does not support FP8")
    # Preserve an existing instance override as well as the inherited method.
    had_override = "_postprocess" in language_model.__dict__
    original_override = language_model.__dict__.get("_postprocess")
    language_model._postprocess = partial(
        language_model._postprocess,
        output_processor=partial(
            _qwen_vl_output_processor,
            label=label,
            label_mask=label_mask,
            chunk_size=chunk_size,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            tp_group=language_model.pg_collection.tp,
        ),
    )
    try:
        return model(**model_inputs)
    finally:
        if had_override:
            language_model._postprocess = original_override
        else:
            del language_model._postprocess


def _qwen_vl_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    inference_context,
    label,
    label_mask,
    chunk_size,
    temperature,
    calculate_entropy,
    tp_group,
    **kwargs,
):
    if labels is not None or inference_context is not None:
        raise ValueError(
            "Qwen-VL chunked scoring requires labels=None and no inference context"
        )
    weight = output_layer.weight if output_weight is None else output_weight
    hidden = hidden_states.to(weight.dtype)
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    if tp_size > 1:
        from megatron.core import tensor_parallel

        if output_layer.sequence_parallel:
            hidden = tensor_parallel.gather_from_sequence_parallel_region(
                hidden, tensor_parallel_output_grad=True, group=tp_group
            )
        else:
            hidden = tensor_parallel.copy_to_tensor_model_parallel_region(
                hidden, group=tp_group
            )
    return chunked_vocab_parallel_log_probs(
        hidden.transpose(0, 1),
        weight,
        label,
        label_mask,
        chunk_size=chunk_size,
        temperature=temperature,
        bias=output_layer.bias,
        tp_group=tp_group,
        calculate_entropy=calculate_entropy,
        gradient_accumulation_fusion=output_layer.gradient_accumulation_fusion,
    )
