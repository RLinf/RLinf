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
from ..utils.backbone_pipeline import compute_values_from_hidden

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
        def decode_fast_ids_to_action(fast_ids: list[int]) -> np.ndarray:
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
            if arr.shape != (n_chunks, act_dim):
                raise RuntimeError(
                    f"QwenFast decode shape mismatch: expected ({n_chunks}, {act_dim}), got {arr.shape}."
                )
            return arr

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
        # 5) Decode generated FAST action tokens using native starVLA helpers only.
        native_extract = getattr(policy.starvla_model, "_extract_action_token_ids", None)
        native_decode = getattr(policy.starvla_model, "_decode_action_tokens", None)
        if not callable(native_extract) or not callable(native_decode):
            raise RuntimeError(
                "QwenFast native decode unavailable: starvla_model must expose "
                "_extract_action_token_ids and _decode_action_tokens."
            )
        try:
            batch_vlm_ids = native_extract(gen_token_ids)
            batch_fast_ids = native_decode(batch_vlm_ids)
        except Exception as exc:
            raise RuntimeError(
                "QwenFast native decode failed when calling starVLA helpers: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

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

        for b in range(bsz):
            vlm_ids = [int(t) for t in batch_vlm_ids[b]][:max_action_tokens]
            fast_ids = [int(t) for t in batch_fast_ids[b]][:max_action_tokens]
            if not vlm_ids or not fast_ids:
                raise RuntimeError(f"QwenFast native decode empty action tokens at sample {b}.")
            if len(vlm_ids) != len(fast_ids):
                raise RuntimeError(
                    f"QwenFast native decode token length mismatch at sample {b}: "
                    f"{len(vlm_ids)} vs {len(fast_ids)}."
                )
            arr = decode_fast_ids_to_action(fast_ids)

            idx = action_mask[b].nonzero(as_tuple=False).flatten()
            prefix_len = len(vlm_ids)
            if idx.numel() == 0:
                raise RuntimeError(f"QwenFast no action token span found at sample {b}.")
            idx = idx[:prefix_len]
            prefix_len = int(idx.numel())

            action_tokens[b, :prefix_len] = torch.as_tensor(
                vlm_ids[:prefix_len],
                device=gen_token_ids.device,
                dtype=torch.long,
            )
            action_token_mask[b, :prefix_len] = True
            token_logprob_sums[b] = gen_logprobs[b, idx[:prefix_len]].sum().to(
                dtype=torch.float32
            )
            normalized_actions[b] = arr

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
            prev_values = compute_values_from_hidden(
                value_head=policy.value_head,
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
