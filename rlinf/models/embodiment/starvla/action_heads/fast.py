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

"""Shared FAST handlers for rollout and default_forward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from ..utils import data_pipeline as data_pipeline_utils
from ..utils import vlm_preprocess as vlm_input_utils
from ..utils.backbone_pipeline import compute_values_from_hidden, run_backbone_pipeline
from ..utils.profile import (
    RL_BATCH_TENSOR_KEYS_TO_IGNORE,
    resolve_action_chunk_len,
    resolve_vlm_interface,
)

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


def _apply_sampling_filters(
    logits: torch.Tensor,
    *,
    data: dict[str, Any],
) -> torch.Tensor:
    """Apply FAST sampling filters to replay logits before PPO replay."""
    filtered = logits

    temp = data_pipeline_utils.get_scalar(
        data.get("temperature"), default=1.0, cast=float
    )
    if temp <= 0:
        temp = 1.0
    filtered = filtered / temp

    top_k = data_pipeline_utils.get_scalar(data.get("top_k"), default=0, cast=int)
    if top_k > 0:
        k = min(top_k, filtered.size(-1))
        kth = torch.topk(filtered, k=k, dim=-1).values[..., -1]
        neg_inf = filtered.new_full((), float("-inf"))
        filtered = torch.where(filtered < kth.unsqueeze(-1), neg_inf, filtered)

    top_p = data_pipeline_utils.get_scalar(data.get("top_p"), default=1.0, cast=float)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumprobs > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return filtered


# Fetch the pad token id from the VLM config if available, with a safe fallback to 0
def _resolve_vlm_pad_token_id(starvla_model: torch.nn.Module, default: int = 0) -> int:
    """Resolve FAST pad token id from VLM config with safe fallback."""
    iface = resolve_vlm_interface(starvla_model)
    model_cfg = getattr(getattr(iface, "model", None), "config", None)
    pad_id = (
        default if model_cfg is None else getattr(model_cfg, "pad_token_id", default)
    )
    try:
        return int(pad_id or default)
    except (TypeError, ValueError):
        return int(default)


def _run_fast_pipeline(
    policy: StarVLAForRLActionPrediction,
) -> tuple[int, int, float, int]:
    """Resolve shared FAST context for default_forward or rollout entry."""
    # Resolve shared FAST action geometry and padded token budget once for both paths.
    n_chunks = int(policy.num_action_chunks)
    act_dim = int(policy.action_dim)
    token_level_denom = float(n_chunks * act_dim)

    configured = getattr(policy, "qwenfast_max_action_tokens", None)
    try:
        max_action_tokens = int(configured or 0)
    except (TypeError, ValueError):
        max_action_tokens = 0
    if max_action_tokens <= 0:
        max_action_tokens = 256

    return n_chunks, act_dim, token_level_denom, max_action_tokens


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
    n_chunks, act_dim, token_level_denom, max_action_tokens = _run_fast_pipeline(policy)

    # default_forward branch:
    # rebuild the prompt+action-token sequence exactly as rollout stored it in replay.
    data_pipeline_utils.forward_input_check(data)

    prompt_input_ids = data["input_ids"].to(dtype=torch.long)
    prompt_attention_mask = data["attention_mask"]
    action_tokens = data["action_tokens"].to(dtype=torch.long)
    if action_tokens.ndim != 2:
        raise ValueError(
            f"FAST expected action_tokens [B, Lmax], got {tuple(action_tokens.shape)}"
        )
    bsz, num_action_tokens = action_tokens.shape
    if num_action_tokens != max_action_tokens:
        raise ValueError(
            "FAST expected padded action_tokens length "
            f"Lmax={max_action_tokens} but got {num_action_tokens}. "
            "Ensure rollout and training use the same RLINF_QWENFAST_MAX_ACTION_TOKENS."
        )

    token_mask = data.get("action_token_mask")
    if not isinstance(token_mask, torch.Tensor):
        raise KeyError(
            "Missing 'action_token_mask' in training batch. "
            "FAST rollout must store forward_inputs['action_token_mask']."
        )
    token_mask_bool = token_mask.to(dtype=torch.bool)
    action_mask = token_mask_bool.to(
        device=prompt_attention_mask.device,
        dtype=prompt_attention_mask.dtype,
    )

    model_inputs = data_pipeline_utils.collect_default_forward_model_inputs(
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
    model_inputs["input_ids"] = torch.cat(
        [prompt_input_ids, action_tokens],
        dim=-1,
    )
    model_inputs["attention_mask"] = torch.cat(
        [prompt_attention_mask, action_mask],
        dim=-1,
    )

    # Re-run the backbone on the reconstructed sequence and keep only the action suffix logits.
    backbone_output = run_backbone_pipeline(
        policy,
        action_head_name="fast",
        model_inputs=model_inputs,
        use_cache=use_cache,
    )
    action_logits = backbone_output["extras"].get("logits")
    if not isinstance(action_logits, torch.Tensor):
        raise RuntimeError(
            "FAST handler requires 'logits' in backbone extras for sampling/logprob."
        )
    action_logits = action_logits[:, -(num_action_tokens + 1) : -1, :].float()

    # Training can optionally replay the rollout sampling filters before PPO term computation.
    do_sample = bool(
        data_pipeline_utils.get_scalar(
            data.get("do_sample"),
            default=0,
            cast=int,
        )
    )
    if do_sample:
        action_logits = _apply_sampling_filters(
            logits=action_logits,
            data=data,
        )

    result: dict[str, torch.Tensor | None] = {
        "logprobs": None,
        "entropy": None,
        "values": None,
    }
    logp_all: Optional[torch.Tensor] = None
    if compute_logprobs or compute_entropy:
        logp_all = torch.log_softmax(action_logits, dim=-1)

    if compute_logprobs:
        if logp_all is None:
            raise RuntimeError(
                "Internal error: expected token log-probs tensor when compute_logprobs is enabled."
            )
        logprobs = logp_all.gather(dim=-1, index=action_tokens.unsqueeze(-1)).squeeze(
            -1
        )
        logprobs = torch.where(token_mask_bool, logprobs, torch.zeros_like(logprobs))
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
            feat = backbone_output["last_hidden"][:, -(num_action_tokens + 1), :]
            vh_dtype = next(policy.value_head.parameters()).dtype
            result["values"] = policy.value_head(feat.to(dtype=vh_dtype)).to(
                dtype=torch.float32
            )

    return result


def run_rollout_fast(
    policy: StarVLAForRLActionPrediction,
    *,
    examples: list[dict[str, Any]],
    mode: str,
    calculate_logprobs: bool,
    calculate_values: bool,
    sampling_kwargs: dict[str, Any],
    env_obs: dict[str, Any],
) -> dict[str, Any]:
    """Roll out the FAST token action head and pack replay caches for training."""
    del mode, env_obs

    backbone_output = None
    if calculate_values:
        backbone_output = run_backbone_pipeline(
            policy,
            action_head_name="fast",
            examples=examples,
            use_cache=False,
        )
        model_inputs = dict(backbone_output["model_inputs"])
    else:
        starvla_model = policy.starvla_model
        model_inputs = vlm_input_utils.build_base_vlm_inputs(
            starvla_model,
            examples=examples,
            vlm_type=policy.vlm_type,
        )

    prompt_inputs = dict(model_inputs)
    n_chunks, act_dim, token_level_denom, max_action_tokens = _run_fast_pipeline(policy)

    with torch.no_grad():
        # rollout branch:
        # generate FAST action tokens directly from the VLM prompt using the requested sampling setup.
        do_sample = bool(sampling_kwargs["do_sample"])
        temperature = float(sampling_kwargs["temperature"])
        top_k = int(sampling_kwargs["top_k"])
        top_p = float(sampling_kwargs["top_p"])
        max_new_tokens = sampling_kwargs["max_new_tokens"]
        max_length = sampling_kwargs["max_length"]

        if max_new_tokens is None and max_length is None:
            max_new_tokens = 256

        vlm_interface = resolve_vlm_interface(policy.starvla_model)
        gen_kwargs: dict[str, Any] = {
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                }
            )
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        elif max_length is not None:
            gen_kwargs["max_length"] = int(max_length)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_out = vlm_interface.model.generate(
                **prompt_inputs,
                **gen_kwargs,
            )

        sequences = getattr(gen_out, "sequences", gen_out)
        scores = getattr(gen_out, "scores", None)
        if sequences is None:
            raise RuntimeError(
                "QwenFast generate did not return 'sequences'. Ensure "
                "'return_dict_in_generate=True' and 'output_scores=True'."
            )
        if scores is None or len(scores) == 0:
            raise RuntimeError(
                "QwenFast generate did not return non-empty 'scores'. Ensure "
                "'return_dict_in_generate=True' and 'output_scores=True', and the "
                "generated sequence length is > 0."
            )

        # Convert step scores into token-level logprobs for the actually generated suffix.
        gen_len = len(scores)
        gen_token_ids = sequences[:, -gen_len:]
        bsz = int(gen_token_ids.size(0))
        gen_logprobs = torch.empty(
            (bsz, gen_len), device=gen_token_ids.device, dtype=torch.float32
        )
        for t, step_scores in enumerate(scores):
            logp = torch.log_softmax(step_scores.float(), dim=-1)
            gen_logprobs[:, t] = logp.gather(
                dim=-1, index=gen_token_ids[:, t].unsqueeze(-1)
            ).squeeze(-1)
        # Identify which generated tokens correspond to FAST action tokens.
        act_min = getattr(vlm_interface, "_ACTION_TOKEN_MIN", None)
        act_max = getattr(vlm_interface, "_ACTION_TOKEN_MAX", None)
        if act_min is None or act_max is None:
            raise RuntimeError(
                "FAST action head requires '_ACTION_TOKEN_MIN/_ACTION_TOKEN_MAX' on the VLM interface."
            )
        action_mask = (gen_token_ids >= int(act_min)) & (gen_token_ids <= int(act_max))

        # Ensure the checkpoint horizon matches the PPO action horizon expected by the policy.
        qwenfast_chunks_test = resolve_action_chunk_len(
            policy.starvla_model,
            policy.num_action_chunks,
            action_head_name="fast",
        )
        if int(qwenfast_chunks_test) != int(policy.num_action_chunks):
            raise RuntimeError(
                "QwenFast action horizon mismatch: "
                f"FAST time_horizon={int(qwenfast_chunks_test)} but policy.num_action_chunks={int(policy.num_action_chunks)}. "
                "Set actor.model.num_action_chunks to match the checkpoint's FAST time_horizon."
            )

        expected_coeffs = n_chunks * act_dim
        pad_id = _resolve_vlm_pad_token_id(policy.starvla_model, default=0)
        fast_processor = policy.starvla_model.action_model.fast_tokenizer

        # Allocate the replay tensors that training will later consume verbatim.
        action_tokens = torch.full(
            (bsz, max_action_tokens),
            fill_value=pad_id,
            device=gen_token_ids.device,
            dtype=torch.long,
        )
        action_token_mask = torch.zeros(
            (bsz, max_action_tokens),
            device=gen_token_ids.device,
            dtype=torch.bool,
        )
        token_logprob_sums = torch.zeros(
            (bsz,),
            device=gen_token_ids.device,
            dtype=torch.float32,
        )
        normalized_actions = np.zeros((bsz, n_chunks, act_dim), dtype=np.float32)

        native_extract = getattr(
            policy.starvla_model, "_extract_action_token_ids", None
        )
        native_decode = getattr(policy.starvla_model, "_decode_action_tokens", None)
        if not callable(native_extract) or not callable(native_decode):
            raise RuntimeError(
                "QwenFast native decode unavailable: starvla_model must expose "
                "_extract_action_token_ids and _decode_action_tokens."
            )
        try:
            # Native helpers define the authoritative mapping from generated VLM tokens
            # to FAST decoder tokens and continuous actions.
            batch_vlm_ids = native_extract(gen_token_ids)
            batch_fast_ids = native_decode(batch_vlm_ids)
        except Exception as exc:
            raise RuntimeError(
                "QwenFast native decode failed when calling starVLA helpers: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Check the batch-level output shapes before per-sample validation to catch gross mismatches early.
        if not (
            isinstance(batch_vlm_ids, list)
            and isinstance(batch_fast_ids, list)
            and len(batch_vlm_ids) == bsz
            and len(batch_fast_ids) == bsz
        ):
            raise RuntimeError(
                "QwenFast native decode returned invalid batch shapes: "
                f"len(vlm_ids)={len(batch_vlm_ids) if isinstance(batch_vlm_ids, list) else 'N/A'}, "
                f"len(fast_ids)={len(batch_fast_ids) if isinstance(batch_fast_ids, list) else 'N/A'}, "
                f"expected={bsz}."
            )

        # Validate each sample and pack the action suffix into replay-friendly tensors.
        for b in range(bsz):
            vlm_ids = [int(t) for t in batch_vlm_ids[b]]
            fast_ids = [int(t) for t in batch_fast_ids[b]]
            if not vlm_ids or not fast_ids:
                raise RuntimeError(
                    f"QwenFast native decode empty action tokens at sample {b}."
                )
            if len(vlm_ids) != len(fast_ids):
                raise RuntimeError(
                    f"QwenFast native decode token length mismatch at sample {b}: "
                    f"{len(vlm_ids)} vs {len(fast_ids)}."
                )
            if len(vlm_ids) > max_action_tokens:
                raise RuntimeError(
                    f"QwenFast action token length exceeds max_action_tokens at sample {b}: "
                    f"len={len(vlm_ids)} > max_action_tokens={max_action_tokens}. "
                    "Increase RLINF_QWENFAST_MAX_ACTION_TOKENS."
                )
            # Decode the FAST action tokens into the continuous action coefficients, validating the expected shape.
            arr = np.asarray(fast_processor.decode([fast_ids]))
            if arr.dtype == object:
                arr = np.asarray(arr[0], dtype=np.float32)
            else:
                arr = arr.astype(np.float32)
                if arr.ndim >= 1:
                    arr = arr[0]
            if arr.ndim == 1 and arr.size == expected_coeffs:
                arr = arr.reshape(n_chunks, act_dim)
            if arr.shape != (n_chunks, act_dim):
                raise RuntimeError(
                    f"QwenFast decode shape mismatch: expected ({n_chunks}, {act_dim}), got {arr.shape}."
                )

            idx = action_mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                raise RuntimeError(
                    f"QwenFast no action token span found at sample {b}."
                )
            if int(idx.numel()) != len(vlm_ids):
                raise RuntimeError(
                    "QwenFast action-token span mismatch at sample "
                    f"{b}: action_mask count={int(idx.numel())} but native_extract returned {len(vlm_ids)} tokens."
                )
            expected_idx = torch.arange(
                gen_len,
                device=gen_token_ids.device,
                dtype=idx.dtype,
            )[-len(vlm_ids) :]
            if not torch.equal(idx, expected_idx):
                raise RuntimeError(
                    "QwenFast action tokens must form a contiguous suffix of the generated sequence "
                    f"for stable PPO replay, but sample {b} has indices={idx.tolist()} "
                    f"(expected suffix indices={expected_idx.tolist()})."
                )
            expected_tokens = torch.as_tensor(
                vlm_ids,
                device=gen_token_ids.device,
                dtype=torch.long,
            )
            if not torch.equal(gen_token_ids[b, idx], expected_tokens):
                raise RuntimeError(
                    f"QwenFast action token IDs mismatch at sample {b}: native_extract tokens do not match "
                    "generated token IDs at masked positions."
                )

            prefix_len = len(vlm_ids)
            action_tokens[b, :prefix_len] = expected_tokens
            action_token_mask[b, :prefix_len] = True
            token_logprob_sums[b] = gen_logprobs[b, idx].sum()
            normalized_actions[b] = arr

        # Normalize token logprobs back to the PPO action tensor shape used by the rest of the stack.
        action_logprobs = (
            (token_logprob_sums / token_level_denom)
            .view(bsz, 1, 1)
            .expand(bsz, n_chunks, act_dim)
            .contiguous()
        )
        output = {
            "normalized_actions": normalized_actions,
            "action_tokens": action_tokens,
            "action_token_mask": action_token_mask,
            "action_logprobs": action_logprobs,
            "model_inputs": dict(prompt_inputs),
        }

    prev_logprobs: Optional[torch.Tensor]
    if calculate_logprobs:
        prev_logprobs = output["action_logprobs"].to(dtype=torch.float32)
    else:
        prev_logprobs = None

    prev_values: Optional[torch.Tensor] = None
    if calculate_values:
        if backbone_output is None:
            raise RuntimeError(
                "Internal error: calculate_values=True requires cached backbone_output."
            )
        if policy.value_head is None:
            prev_values = torch.zeros((len(examples), 1), dtype=torch.float32)
        else:
            prev_values = compute_values_from_hidden(
                value_head=policy.value_head,
                hidden=backbone_output["last_hidden"],
                attention_mask=backbone_output["attention_mask"],
            )

    extra_forward_inputs: dict[str, torch.Tensor] = {
        "action_tokens": output["action_tokens"]
    }
    action_token_mask = output.get("action_token_mask")
    if isinstance(action_token_mask, torch.Tensor):
        extra_forward_inputs["action_token_mask"] = action_token_mask

    return {
        "output": output,
        "model_inputs": model_inputs,
        "prev_logprobs": prev_logprobs,
        "prev_values": prev_values,
        "extra_forward_inputs": extra_forward_inputs,
        "state": None,
    }
