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

"""Rollout-time action sampling for starVLA Qwen FAST action heads."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from ..utils import vlm_preprocess as vlm_input_utils
from ..utils.backbone_pipeline import run_backbone_pipeline
from ..utils.profile import (
    infer_vlm_type,
    resolve_action_chunk_len,
    resolve_fast_action_token_range,
    resolve_fast_max_action_tokens,
    resolve_vlm_interface,
    resolve_vlm_pad_token_id,
)

if TYPE_CHECKING:
    from ..starvla_action_model import StarVLAForRLActionPrediction


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

    # 1) Build rollout prompt inputs (with backbone cache only when value is requested).
    backbone_output = None
    if calculate_values:
        backbone_output = run_backbone_pipeline(
            policy,
            examples=examples,
            use_cache=False,
        )
        model_inputs = dict(backbone_output.model_inputs)
    else:
        starvla_model = policy.starvla_model
        model_inputs = vlm_input_utils.build_base_vlm_inputs(
            starvla_model,
            examples=examples,
            vlm_type=infer_vlm_type(starvla_model),
        )

    prompt_inputs = dict(model_inputs)
    do_sample = bool(sampling_kwargs["do_sample"])
    temperature = float(sampling_kwargs["temperature"])
    top_k = int(sampling_kwargs["top_k"])
    top_p = float(sampling_kwargs["top_p"])
    max_new_tokens = sampling_kwargs["max_new_tokens"]
    max_length = sampling_kwargs["max_length"]

    # 2) Run one generation pass and collect per-step token logprobs.
    with torch.no_grad():
        if max_new_tokens is None and max_length is None:
            max_new_tokens = int(os.environ.get("RLINF_QWENFAST_MAX_NEW_TOKENS", "256"))
        vlm_interface = resolve_vlm_interface(policy.starvla_model)
        gen_kwargs: dict[str, Any] = {
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        elif max_length is not None:
            gen_kwargs["max_length"] = int(max_length)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with autocast_ctx:
            gen_out = vlm_interface.model.generate(
                **prompt_inputs,
                **gen_kwargs,
            )

        sequences = getattr(gen_out, "sequences", gen_out)
        scores = getattr(gen_out, "scores", None)
        if scores is None:
            raise RuntimeError(
                "QwenFast generate did not return 'scores'. Ensure "
                "'return_dict_in_generate=True' and 'output_scores=True'."
            )

        gen_len = len(scores)
        if gen_len == 0:
            raise RuntimeError("QwenFast generate returned empty scores.")

        gen_token_ids = sequences[:, -gen_len:]

        step_logprobs: list[torch.Tensor] = []
        for t in range(gen_len):
            step_scores = scores[t].float()
            step_logp = torch.log_softmax(step_scores, dim=-1).gather(
                dim=-1,
                index=gen_token_ids[:, t].unsqueeze(-1),
            )
            step_logprobs.append(step_logp.squeeze(-1))
        gen_logprobs = torch.stack(step_logprobs, dim=1)

        # 3) Find FAST-action token span and verify rollout horizon contract.
        act_min, act_max = resolve_fast_action_token_range(policy.starvla_model)
        action_mask = (gen_token_ids >= act_min) & (gen_token_ids <= act_max)

        qwenfast_chunks = resolve_action_chunk_len(
            policy.starvla_model,
            policy.num_action_chunks,
            action_head_name="fast",
        )
        if int(qwenfast_chunks) != int(policy.num_action_chunks):
            raise RuntimeError(
                "QwenFast action horizon mismatch: "
                f"FAST time_horizon={int(qwenfast_chunks)} but policy.num_action_chunks={int(policy.num_action_chunks)}. "
                "Set actor.model.num_action_chunks to match the checkpoint's FAST time_horizon."
            )

        # 4) Resolve static dimensions / tokenizer handles for FAST decoding.
        max_action_tokens = resolve_fast_max_action_tokens(policy)
        pad_id = resolve_vlm_pad_token_id(policy.starvla_model, default=0)

        bsz = int(gen_token_ids.size(0))
        n_chunks = int(policy.num_action_chunks)
        act_dim = int(policy.action_dim)
        expected_coeffs = n_chunks * act_dim

        fast_processor = policy.starvla_model.action_model.fast_tokenizer
        fast_bpe_tokenizer = getattr(fast_processor, "bpe_tokenizer", None)
        if fast_bpe_tokenizer is None:
            fast_bpe_tokenizer = getattr(fast_processor, "tokenizer", None)

        def decode_len(tokenizer: Any, token_ids: list[int]) -> int:
            try:
                return len(tokenizer.decode(token_ids))
            except TypeError:
                return len(tokenizer.decode(token_ids, skip_special_tokens=False))

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

        valid_indices: list[int] = []
        valid_fast_token_ids: list[list[int]] = []
        fail_reason_by_sample: dict[int, str] = {}

        # 5) Convert generated tokens into fixed-length storage tensors and per-sample sums.
        for b in range(bsz):
            idx = action_mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() > 1:
                diffs = idx[1:] - idx[:-1]
                break_pos = (diffs != 1).nonzero(as_tuple=False)
                if break_pos.numel() > 0:
                    idx = idx[: int(break_pos[0].item()) + 1]
            if idx.numel() == 0:
                fail_reason_by_sample.setdefault(b, "no_action_token_span")
                continue

            tok_b = gen_token_ids[b, idx]
            lp_b = gen_logprobs[b, idx]
            if tok_b.numel() == 0 or tok_b.numel() > max_action_tokens:
                fail_reason_by_sample.setdefault(b, "invalid_action_token_length")
                continue

            vlm_ids_full = tok_b.tolist()
            fast_ids_full = [t - act_min for t in vlm_ids_full]

            prefix_len: Optional[int]
            if fast_bpe_tokenizer is not None:
                full_len = decode_len(fast_bpe_tokenizer, fast_ids_full)
                if full_len == expected_coeffs:
                    prefix_len = len(fast_ids_full)
                elif full_len < expected_coeffs:
                    prefix_len = None
                else:
                    prefix_len = None
                    for k in range(1, len(fast_ids_full) + 1):
                        n = decode_len(fast_bpe_tokenizer, fast_ids_full[:k])
                        if n == expected_coeffs:
                            prefix_len = k
                            break
                        if n > expected_coeffs:
                            break
            else:
                prefix_len = len(fast_ids_full)
            if prefix_len is None or prefix_len <= 0:
                fail_reason_by_sample.setdefault(b, "prefix_length_mismatch")
                continue

            vlm_ids = vlm_ids_full[:prefix_len]
            fast_ids = fast_ids_full[:prefix_len]
            lp_sel = lp_b[:prefix_len]

            action_tokens[b, :prefix_len] = torch.as_tensor(
                vlm_ids,
                device=gen_token_ids.device,
                dtype=torch.long,
            )
            action_token_mask[b, :prefix_len] = True
            token_logprob_sums[b] = lp_sel.sum().to(dtype=torch.float32)

            valid_indices.append(b)
            valid_fast_token_ids.append(fast_ids)

        decoded_valid_indices: list[int] = []
        if valid_fast_token_ids:
            # Decode each sample and keep only valid [T, D] FAST actions.
            for b, fast_ids in zip(valid_indices, valid_fast_token_ids, strict=True):
                try:
                    decoded_single = fast_processor.decode([fast_ids])
                    arr = np.asarray(decoded_single)
                    if arr.dtype == object:
                        arr = np.asarray(arr[0], dtype=np.float32)
                    else:
                        arr = arr.astype(np.float32)
                        if arr.ndim >= 1:
                            arr = arr[0]

                    if arr.ndim == 1 and arr.size == expected_coeffs:
                        arr = arr.reshape(n_chunks, act_dim)
                    if arr.shape == (n_chunks, act_dim):
                        normalized_actions[b] = arr
                        decoded_valid_indices.append(b)
                        continue
                    fail_reason_by_sample.setdefault(b, "decode_shape_mismatch")
                except Exception:
                    fail_reason_by_sample.setdefault(b, "decode_exception")

                action_tokens[b].fill_(pad_id)
                action_token_mask[b].fill_(False)
                token_logprob_sums[b] = 0.0

        # 6) Strict mode: fail fast when any sample cannot be decoded to valid FAST action.
        decoded_valid_set = set(decoded_valid_indices)
        if len(decoded_valid_indices) < bsz:
            failed_indices = [b for b in range(bsz) if b not in decoded_valid_set]
            reason_counts: dict[str, int] = {}
            for b in failed_indices:
                reason = fail_reason_by_sample.get(b, "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            reason_text = ", ".join(
                f"{k}={v}" for k, v in sorted(reason_counts.items())
            )
            raise RuntimeError(
                "QwenFast strict decode failure: unable to produce valid FAST actions for "
                f"{len(failed_indices)}/{bsz} samples (failed_indices={failed_indices}, reasons={reason_text}). "
                "Suggested fixes: (1) ensure checkpoint and FAST tokenizer are paired; "
                "(2) reduce sampling stochasticity (do_sample=False or lower temperature/top_p); "
                "(3) increase RLINF_QWENFAST_MAX_NEW_TOKENS and qwenfast_max_action_tokens if truncation occurs; "
                "(4) verify actor.model.num_action_chunks matches FAST time_horizon."
            )

        # 7) Map sequence-level FAST logprob sums into RLinf [B, T, D] convention.
        denom = float(n_chunks * act_dim)
        action_logprobs = (
            (token_logprob_sums / denom)
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

    # 8) Build optional PPO baselines and return rollout caches for training replay.
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
            prev_values = policy._compute_values_from_hidden(
                hidden=backbone_output.last_hidden,
                attention_mask=backbone_output.attention_mask,
            )

    action_tokens = output["action_tokens"]
    action_token_mask = output.get("action_token_mask")
    extra_forward_inputs = {"action_tokens": action_tokens}
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
