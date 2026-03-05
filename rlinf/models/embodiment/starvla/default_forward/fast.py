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

"""Training-time forward pass for starVLA Qwen FAST action heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..utils import data_pipeline as data_pipeline_utils
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.profile import (
    RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    resolve_fast_action_token_range,
    resolve_fast_max_action_tokens,
)

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def run_default_forward_fast(
    policy: StarVLAForRLActionPrediction,
    *,
    data: dict[str, torch.Tensor],
    compute_logprobs: bool,
    compute_entropy: bool,
    compute_values: bool,
    use_cache: bool,
) -> dict[str, torch.Tensor | None]:
    """Compute training-time PPO terms for the FAST token action head."""
    # 1) Fast path: skip all computations if no RL terms are requested.
    if not compute_logprobs and not compute_entropy and not compute_values:
        return {
            "logprobs": None,
            "entropy": None,
            "values": None,
        }

    # 2) Validate required rollout caches.
    if "action_tokens" not in data:
        raise KeyError(
            "Missing 'action_tokens' in training batch. "
            "Rollout must store forward_inputs['action_tokens']."
        )
    if "input_ids" not in data or "attention_mask" not in data:
        raise KeyError(
            "Missing prompt inputs ('input_ids'/'attention_mask') in training batch. "
            "Rollout must store VLM prompt tensors in forward_inputs."
        )

    # 3) Load prompt/action token tensors.
    prompt_input_ids = data["input_ids"].to(dtype=torch.long)
    prompt_attention_mask = data["attention_mask"]
    action_tokens = data["action_tokens"].to(dtype=torch.long)

    # 4) Resolve FAST max token length used in token-level mode.
    qwenfast_max_action_tokens = resolve_fast_max_action_tokens(policy)

    # 5) Build concatenated prompt+action sequence for two supported action layouts:
    #    - chunk layout: [B, T, D]
    #    - token layout: [B, Lmax]
    token_level_mode = False
    action_token_seq: torch.Tensor
    token_mask_bool: torch.Tensor
    action_tokens_flat: torch.Tensor | None = None
    if action_tokens.ndim == 3:
        bsz, t, d = action_tokens.shape
        if d != policy.action_dim:
            raise ValueError(
                f"action_dim mismatch: got {d} from action_tokens, expected {policy.action_dim}"
            )
        action_tokens_flat = action_tokens.reshape(bsz, -1)
        action_token_seq = action_tokens_flat
        token_mask_bool = torch.ones(
            action_token_seq.shape,
            device=prompt_attention_mask.device,
            dtype=torch.bool,
        )
    elif action_tokens.ndim == 2:
        token_level_mode = True
        bsz, num_action_tokens = action_tokens.shape
        if num_action_tokens != qwenfast_max_action_tokens:
            raise ValueError(
                "FAST expected padded action_tokens length "
                f"Lmax={qwenfast_max_action_tokens} but got {num_action_tokens}. "
                "Ensure rollout and training use the same RLINF_QWENFAST_MAX_ACTION_TOKENS."
            )
        token_mask = data.get("action_token_mask")
        if isinstance(token_mask, torch.Tensor):
            token_mask_bool = token_mask.to(dtype=torch.bool)
        else:
            act_min, act_max = resolve_fast_action_token_range(policy.starvla_model)
            token_mask_bool = (action_tokens >= act_min) & (action_tokens <= act_max)
        action_token_seq = action_tokens
    else:
        raise ValueError(f"Unsupported action_tokens shape: {action_tokens.shape}")

    num_action_tokens = int(action_token_seq.shape[1])
    input_ids = torch.cat([prompt_input_ids, action_token_seq], dim=-1)
    action_mask = token_mask_bool.to(
        device=prompt_attention_mask.device,
        dtype=prompt_attention_mask.dtype,
    )
    attention_mask = torch.cat([prompt_attention_mask, action_mask], dim=-1)

    # 6) Rebuild model inputs and run backbone once.
    model_inputs = data_pipeline_utils.collect_tensor_inputs(
        data,
        skip_keys={
            "action_tokens",
            "action_token_mask",
            "action",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
        },
        ignored_keys=RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    )
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = attention_mask
    model_inputs = data_pipeline_utils.restore_pixel_values_for_forward(model_inputs)

    backbone_output = run_backbone_pipeline(
        policy,
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    action_logits = backbone_output.extras.get("logits")
    if not isinstance(action_logits, torch.Tensor):
        raise RuntimeError(
            "Fast default_forward requires 'logits' from backbone pipeline output."
        )

    action_logits = action_logits[:, -(num_action_tokens + 1) : -1, :].float()
    action_logits = data_pipeline_utils.apply_sampling_filters(
        logits=action_logits, data=data
    )

    # 7) Prepare result container and (if needed) token log-probabilities.
    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }
    logp_all: torch.Tensor | None = None
    if compute_logprobs or compute_entropy:
        logp_all = torch.log_softmax(action_logits, dim=-1)

    n_chunks = int(policy.num_action_chunks)
    act_dim = int(policy.action_dim)
    token_level_denom = float(n_chunks * act_dim)

    # 8) Compute requested RL terms.
    if compute_logprobs:
        if logp_all is None:
            raise RuntimeError(
                "Internal error: expected token log-probs tensor when compute_logprobs is enabled."
            )
        if not token_level_mode:
            if action_tokens_flat is None:
                raise RuntimeError(
                    "Internal error: expected flattened action tokens in chunk layout."
                )
            logprobs = logp_all.gather(
                dim=-1, index=action_tokens_flat.unsqueeze(-1)
            ).squeeze(-1)
            result["logprobs"] = logprobs.view(bsz, t, d).to(dtype=torch.float32)
        else:
            logprobs = logp_all.gather(
                dim=-1, index=action_tokens.unsqueeze(-1)
            ).squeeze(-1)
            logprobs = torch.where(
                token_mask_bool, logprobs, torch.zeros_like(logprobs)
            )
            total_logprob = logprobs.sum(dim=-1).to(dtype=torch.float32)
            result["logprobs"] = (
                (total_logprob / token_level_denom)
                .view(bsz, 1, 1)
                .expand(bsz, n_chunks, act_dim)
                .contiguous()
            )
    if compute_entropy:
        if logp_all is None:
            raise RuntimeError(
                "Internal error: expected token log-probs tensor when compute_entropy is enabled."
            )
        probs = torch.exp(logp_all)
        entropy = -(probs * logp_all).sum(dim=-1)
        if not token_level_mode:
            result["entropy"] = entropy.view(bsz, t, d).to(dtype=torch.float32)
        else:
            entropy = torch.where(token_mask_bool, entropy, torch.zeros_like(entropy))
            total_entropy = entropy.sum(dim=-1).to(dtype=torch.float32)
            result["entropy"] = (
                (total_entropy / token_level_denom)
                .view(bsz, 1, 1)
                .expand(bsz, n_chunks, act_dim)
                .contiguous()
            )
    if compute_values:
        if policy.value_head is None:
            result["values"] = torch.zeros(
                (bsz, 1), device=action_logits.device, dtype=torch.float32
            )
        else:
            hidden = backbone_output.last_hidden
            feat = hidden[:, -(num_action_tokens + 1), :].float()
            result["values"] = policy.value_head(feat).to(dtype=torch.float32)
    return result
