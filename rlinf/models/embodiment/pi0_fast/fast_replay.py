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

from __future__ import annotations

from typing import Any

import torch


def _language_token_keys() -> tuple[str, str]:
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    return OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


def compute_token_logprobs(
    logits: torch.Tensor,
    action_tokens: torch.Tensor,
    action_token_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute selected-token log probabilities and optional policy entropy.

    Args:
        logits: Policy logits shaped ``[batch, tokens, vocabulary]``.
        action_tokens: Sampled token ids shaped ``[batch, tokens]``.
        action_token_mask: Boolean mask selecting policy-objective tokens.
        temperature: Temperature used to sample the action tokens.
        compute_entropy: Whether to materialize the full-vocabulary entropy.

    Returns:
        Per-token log probabilities and optional per-token entropy.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,V], got {tuple(logits.shape)}")
    if logits.shape[:2] != action_tokens.shape:
        raise ValueError(
            "logits and action_tokens must share [B,T] shape, got "
            f"{tuple(logits.shape[:2])} vs {tuple(action_tokens.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    action_tokens = action_tokens.to(device=logits.device)
    scaled_logits = logits.float() / temperature
    selected_logits = scaled_logits.gather(
        dim=-1, index=action_tokens.long().unsqueeze(-1)
    ).squeeze(-1)
    logprobs = selected_logits - torch.logsumexp(scaled_logits, dim=-1)
    mask = action_token_mask.to(dtype=torch.bool, device=logprobs.device)
    logprobs = torch.where(mask, logprobs, torch.zeros_like(logprobs))
    if not compute_entropy:
        return logprobs, None

    logp_all = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(logp_all)
    entropy = -(probs * logp_all).sum(dim=-1)
    entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
    return logprobs, entropy


def _first_tensor_device(batch: dict[str, Any]) -> torch.device | None:
    for value in batch.values():
        if torch.is_tensor(value):
            return value.device
        if isinstance(value, dict):
            device = _first_tensor_device(value)
            if device is not None:
                return device
    return None


def _policy_device(policy, batch: dict[str, Any] | None = None) -> torch.device:
    first_param = next(policy.parameters(), None)
    if first_param is not None:
        return first_param.device

    if batch is not None:
        batch_device = _first_tensor_device(batch)
        if batch_device is not None:
            return batch_device

    model = getattr(policy, "model", None)
    if model is not None:
        first_model_param = next(model.parameters(), None)
        if first_model_param is not None:
            return first_model_param.device

    return torch.device("cpu")


def _policy_floating_dtype(policy) -> torch.dtype | None:
    for param in policy.parameters():
        if param.is_floating_point():
            return param.dtype

    model = getattr(policy, "model", None)
    if model is not None:
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype

    return None


def _preprocess_images(
    policy, batch: dict[str, Any]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    images = []
    img_masks = []
    device = _policy_device(policy, batch)
    target_dtype = _policy_floating_dtype(policy)

    present_img_keys = [key for key in policy.config.image_features if key in batch]
    missing_img_keys = [key for key in policy.config.image_features if key not in batch]

    if len(present_img_keys) == 0:
        raise ValueError(
            "All image features are missing from the batch. At least one expected. "
            f"(batch: {batch.keys()}) (image_features: {policy.config.image_features})"
        )

    img = None
    mask = None
    for key in present_img_keys:
        img = batch[key]
        if img.device != device:
            img = img.to(device)
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        is_channels_first = img.shape[1] == 3
        if is_channels_first:
            img = img.permute(0, 2, 3, 1)
        if img.shape[1:3] != policy.config.image_resolution:
            from lerobot.policies.pi0_fast.modeling_pi0_fast import (
                resize_with_pad_torch,
            )

            img = resize_with_pad_torch(img, *policy.config.image_resolution)
        img = img * 2.0 - 1.0
        if is_channels_first:
            img = img.permute(0, 3, 1, 2)
        if target_dtype is not None and img.dtype != target_dtype:
            img = img.to(dtype=target_dtype)

        images.append(img)
        mask = torch.ones(img.shape[0], dtype=torch.bool, device=device)
        img_masks.append(mask)

    if img is None or mask is None:
        raise ValueError("pi0_fast image preprocessing requires at least one image.")

    for _ in range(len(missing_img_keys)):
        images.append(torch.ones_like(img) * -1)
        img_masks.append(torch.zeros_like(mask))

    return images, img_masks


def _ensure_prefix_precision(model, prefix_embs: torch.Tensor) -> torch.Tensor:
    layer = model.paligemma_with_expert.paligemma.language_model.layers[0]
    if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
        return prefix_embs.to(dtype=torch.bfloat16)
    return prefix_embs


def _condition_prefix(policy, batch: dict[str, Any]):
    token_key, mask_key = _language_token_keys()
    device = _policy_device(policy, batch)
    images, img_masks = _preprocess_images(policy, batch)
    tokens = batch[token_key].to(device=device, dtype=torch.long)
    masks = batch[mask_key].to(device=device, dtype=torch.bool)
    bos_token = torch.full(
        (tokens.shape[0], 1),
        policy._paligemma_tokenizer.bos_token_id,
        dtype=torch.long,
        device=device,
    )
    tokens = torch.cat([tokens, bos_token], dim=1)
    masks = torch.cat(
        [masks, torch.ones((masks.shape[0], 1), dtype=torch.bool, device=device)],
        dim=1,
    )
    return images, img_masks, tokens, masks


def _forward_embeds(model, prefix_embs, prefix_pad_masks, prefix_att_masks):
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = model._prepare_attention_masks_4d(
        prefix_att_masks, dtype=prefix_embs.dtype
    )
    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        adarms_cond=[None, None],
    )
    return prefix_out


def _sample_next_token(
    logits: torch.Tensor, *, temperature: float, do_sample: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    if do_sample and temperature > 0:
        sampling_logits = logits / temperature
        probs = torch.softmax(sampling_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        sampling_logits = logits
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    logprob = torch.log_softmax(sampling_logits.float(), dim=-1).gather(
        dim=-1, index=next_token
    )
    return next_token, logprob.squeeze(-1)


def _action_prefix_token_tensor(policy, device: torch.device) -> torch.Tensor:
    prefix_ids = policy._paligemma_tokenizer.encode(
        "Action: ", add_special_tokens=False
    )
    if len(prefix_ids) == 0:
        raise ValueError("pi0_fast tokenizer produced an empty Action prefix.")
    return torch.tensor(prefix_ids, dtype=torch.long, device=device)


def _end_marker_token_tensor(policy, device: torch.device) -> torch.Tensor:
    end_ids = policy._paligemma_tokenizer.encode("|", add_special_tokens=False)
    if len(end_ids) == 0:
        raise ValueError("pi0_fast tokenizer produced an empty action end marker.")
    return torch.tensor(end_ids, dtype=torch.long, device=device)


def _init_action_token_buffers(
    *,
    batch_size: int,
    max_action_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_tokens = torch.zeros(
        (batch_size, max_action_tokens), dtype=torch.long, device=device
    )
    action_token_mask = torch.ones(
        (batch_size, max_action_tokens), dtype=torch.bool, device=device
    )
    return action_tokens, action_token_mask


def _find_first_subsequence(values: list[int], needle: list[int], *, start: int) -> int:
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def _bpe_vocab_size(policy) -> int:
    tokenizer = policy.action_tokenizer.bpe_tokenizer
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    if hasattr(tokenizer, "get_vocab"):
        return len(tokenizer.get_vocab())
    raise AttributeError("FAST BPE tokenizer does not expose its vocabulary size.")


def build_action_sequence_metadata(
    policy: Any,
    action_tokens: torch.Tensor,
    generation_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Validate native FAST output and build the PPO mask through the first `|`."""
    device = action_tokens.device
    prefix_ids = _action_prefix_token_tensor(policy, device).tolist()
    end_ids = _end_marker_token_tensor(policy, device).tolist()
    bpe_vocab_size = _bpe_vocab_size(policy)
    paligemma_vocab_size = int(policy._paligemma_tokenizer.vocab_size)
    fast_skip_tokens = int(policy.config.fast_skip_tokens)

    batch_size, sequence_length = action_tokens.shape
    if generation_mask is None:
        generation_mask = torch.ones_like(action_tokens, dtype=torch.bool)
    generation_mask = generation_mask.to(device=device, dtype=torch.bool)
    if generation_mask.shape != action_tokens.shape:
        raise ValueError(
            "generation_mask must match action_tokens, got "
            f"{tuple(generation_mask.shape)} vs {tuple(action_tokens.shape)}"
        )
    prefix_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    end_marker_present = torch.zeros(batch_size, dtype=torch.bool, device=device)
    body_decode_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    action_logprob_mask = generation_mask.clone()

    for row in range(batch_size):
        generated_count = int(generation_mask[row].sum().item())
        token_ids = action_tokens[row, :generated_count].detach().cpu().tolist()
        prefix_valid[row] = token_ids[: len(prefix_ids)] == prefix_ids

        end_index = _find_first_subsequence(token_ids, end_ids, start=len(prefix_ids))
        if end_index < 0:
            continue

        end_marker_present[row] = True
        mask_end = min(end_index + len(end_ids), sequence_length)
        action_logprob_mask[row, mask_end:] = False

        body_token_ids = token_ids[len(prefix_ids) : end_index]
        action_ids = [
            paligemma_vocab_size - 1 - fast_skip_tokens - token_id
            for token_id in body_token_ids
        ]
        if not action_ids or any(
            action_id < 0 or action_id >= bpe_vocab_size for action_id in action_ids
        ):
            continue

        try:
            policy.action_tokenizer.bpe_tokenizer.decode(action_ids)
        except Exception:
            continue
        body_decode_valid[row] = True

    return {
        "prefix_valid": prefix_valid,
        "end_marker_present": end_marker_present,
        "body_decode_valid": body_decode_valid,
        "action_logprob_mask": action_logprob_mask,
    }


def safe_detokenize_actions(
    policy: Any,
    action_tokens: torch.Tensor,
    *,
    action_horizon: int,
    action_dim: int,
    generation_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decode valid rows and leave invalid rows as normalized zero actions."""
    metadata = build_action_sequence_metadata(
        policy, action_tokens, generation_mask=generation_mask
    )
    decode_valid = (
        metadata["prefix_valid"]
        & metadata["end_marker_present"]
        & metadata["body_decode_valid"]
    )
    actions = torch.zeros(
        (action_tokens.shape[0], action_horizon, action_dim),
        dtype=torch.float32,
        device=action_tokens.device,
    )

    for row in torch.nonzero(decode_valid, as_tuple=False).flatten().tolist():
        try:
            decoded = policy.detokenize_actions(
                action_tokens[row : row + 1],
                action_horizon=action_horizon,
                action_dim=action_dim,
            )
            decoded = decoded[0, :action_horizon, :action_dim].to(
                device=actions.device, dtype=actions.dtype
            )
            if decoded.shape != actions[row].shape or not torch.isfinite(decoded).all():
                decode_valid[row] = False
                continue
            actions[row].copy_(decoded)
        except Exception:
            decode_valid[row] = False

    metadata["decode_valid"] = decode_valid
    return actions, metadata


def _append_action_token(model, prefix_embs, prefix_pad_masks, prefix_att_masks, token):
    token_emb = model.paligemma_with_expert.embed_language_tokens(token)
    token_emb = _ensure_prefix_precision(model, token_emb)
    prefix_embs = torch.cat([prefix_embs, token_emb], dim=1)

    bsz = prefix_pad_masks.shape[0]
    device = prefix_pad_masks.device
    prefix_pad_masks = torch.cat(
        [prefix_pad_masks, torch.ones((bsz, 1), dtype=torch.bool, device=device)],
        dim=1,
    )

    old_len = prefix_att_masks.shape[1]
    new_len = old_len + 1
    new_att_masks = torch.zeros(
        (bsz, new_len, new_len), dtype=torch.bool, device=device
    )
    new_att_masks[:, :old_len, :old_len] = prefix_att_masks
    new_att_masks[:, -1, :] = prefix_pad_masks
    return prefix_embs, prefix_pad_masks, new_att_masks


def _advance_kv_cache(
    model,
    lm_head,
    past_key_values,
    current_pad_mask,
    token,
    *,
    embedding_dtype: torch.dtype,
):
    token_emb = model.paligemma_with_expert.embed_language_tokens(token)
    if embedding_dtype == torch.bfloat16:
        token_emb = token_emb.to(dtype=torch.bfloat16)

    bsz = current_pad_mask.shape[0]
    device = current_pad_mask.device
    current_pad_mask = torch.cat(
        [
            current_pad_mask,
            torch.ones((bsz, 1), dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    current_position_ids = (current_pad_mask.sum(dim=1, keepdim=True) - 1).long()
    step_att_mask = model._prepare_attention_masks_4d(
        current_pad_mask.unsqueeze(1), dtype=token_emb.dtype
    )
    (step_out, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=step_att_mask,
        position_ids=current_position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[token_emb, None],
        use_cache=True,
        adarms_cond=[None, None],
    )
    logits = lm_head(step_out[:, -1, :])
    return logits, past_key_values, current_pad_mask


def generate_action_tokens_with_logprobs(
    policy: Any,
    batch: dict[str, Any],
    *,
    max_action_tokens: int,
    num_action_chunks: int,
    action_dim: int,
    temperature: float,
    do_sample: bool = True,
    compute_logprobs: bool = True,
) -> dict[str, torch.Tensor]:
    """Generate a native FAST action sequence and optional behavior logprobs.

    Args:
        policy: LeRobot PI0-Fast policy.
        batch: Prepared LeRobot policy inputs.
        max_action_tokens: Maximum number of autoregressive tokens to generate.
        num_action_chunks: Number of decoded action chunks returned to RLinf.
        action_dim: Number of action dimensions per chunk.
        temperature: Sampling temperature.
        do_sample: Use multinomial sampling when true, otherwise greedy decoding.
        compute_logprobs: Replay generated tokens to compute behavior logprobs.

    Returns:
        Generated tokens, masks, decoded actions, validity metadata, and optional
        behavior log probabilities.
    """
    if getattr(getattr(policy, "config", None), "use_kv_cache", True):
        model = getattr(policy, "model", None)
        restore_gradient_checkpointing = bool(
            getattr(model, "gradient_checkpointing_enabled", False)
        )
        try:
            if restore_gradient_checkpointing:
                model.gradient_checkpointing_disable()
            return _generate_action_tokens_with_logprobs_kv_cache(
                policy,
                batch,
                max_action_tokens=max_action_tokens,
                num_action_chunks=num_action_chunks,
                action_dim=action_dim,
                temperature=temperature,
                do_sample=do_sample,
                compute_logprobs=compute_logprobs,
            )
        finally:
            if restore_gradient_checkpointing:
                model.gradient_checkpointing_enable()

    model = policy.model
    images, img_masks, tokens, masks = _condition_prefix(policy, batch)
    bsz = tokens.shape[0]
    device = tokens.device
    lm_head = model.paligemma_with_expert.paligemma.lm_head

    prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = model.embed_prefix_fast(
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    )
    prefix_embs = _ensure_prefix_precision(model, prefix_embs)

    action_tokens, action_token_mask = _init_action_token_buffers(
        batch_size=bsz,
        max_action_tokens=max_action_tokens,
        device=device,
    )

    for step in range(max_action_tokens):
        prefix_out = _forward_embeds(
            model, prefix_embs, prefix_pad_masks, prefix_att_masks
        )
        logits = lm_head(prefix_out[:, -1, :])
        next_token, _ = _sample_next_token(
            logits, temperature=temperature, do_sample=do_sample
        )
        action_tokens[:, step] = next_token.squeeze(-1)
        if step < max_action_tokens - 1:
            prefix_embs, prefix_pad_masks, prefix_att_masks = _append_action_token(
                model, prefix_embs, prefix_pad_masks, prefix_att_masks, next_token
            )

    actions, metadata = safe_detokenize_actions(
        policy,
        action_tokens,
        action_horizon=num_action_chunks,
        action_dim=action_dim,
        generation_mask=action_token_mask,
    )
    result = {
        "actions": actions,
        "action_tokens": action_tokens,
        "action_token_mask": action_token_mask,
        "action_logprob_mask": metadata["action_logprob_mask"],
        "prefix_valid": metadata["prefix_valid"],
        "end_marker_present": metadata["end_marker_present"],
        "decode_valid": metadata["decode_valid"],
    }
    if compute_logprobs:
        replay_logits, _ = replay_action_logits(
            policy, batch, action_tokens, action_token_mask
        )
        token_logprobs, _ = compute_token_logprobs(
            replay_logits,
            action_tokens,
            metadata["action_logprob_mask"],
            temperature=temperature,
            compute_entropy=False,
        )
        result["token_logprobs"] = token_logprobs
    return result


def _generate_action_tokens_with_logprobs_kv_cache(
    policy,
    batch: dict[str, Any],
    *,
    max_action_tokens: int,
    num_action_chunks: int,
    action_dim: int,
    temperature: float,
    do_sample: bool = True,
    compute_logprobs: bool = True,
) -> dict[str, torch.Tensor]:
    model = policy.model
    images, img_masks, tokens, masks = _condition_prefix(policy, batch)
    bsz = tokens.shape[0]
    device = tokens.device
    lm_head = model.paligemma_with_expert.paligemma.lm_head

    prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = model.embed_prefix_fast(
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    )
    prefix_embs = _ensure_prefix_precision(model, prefix_embs)

    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = model._prepare_attention_masks_4d(
        prefix_att_masks, dtype=prefix_embs.dtype
    )
    (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        adarms_cond=[None, None],
    )

    logits = lm_head(prefix_out[:, -1, :])
    action_tokens, action_token_mask = _init_action_token_buffers(
        batch_size=bsz,
        max_action_tokens=max_action_tokens,
        device=device,
    )
    current_pad_mask = prefix_pad_masks

    for step in range(max_action_tokens):
        next_token, _ = _sample_next_token(
            logits, temperature=temperature, do_sample=do_sample
        )
        action_tokens[:, step] = next_token.squeeze(-1)
        if step < max_action_tokens - 1:
            logits, past_key_values, current_pad_mask = _advance_kv_cache(
                model,
                lm_head,
                past_key_values,
                current_pad_mask,
                next_token,
                embedding_dtype=prefix_embs.dtype,
            )

    actions, metadata = safe_detokenize_actions(
        policy,
        action_tokens,
        action_horizon=num_action_chunks,
        action_dim=action_dim,
        generation_mask=action_token_mask,
    )
    result = {
        "actions": actions,
        "action_tokens": action_tokens,
        "action_token_mask": action_token_mask,
        "action_logprob_mask": metadata["action_logprob_mask"],
        "prefix_valid": metadata["prefix_valid"],
        "end_marker_present": metadata["end_marker_present"],
        "decode_valid": metadata["decode_valid"],
    }
    if compute_logprobs:
        replay_logits, _ = replay_action_logits(
            policy, batch, action_tokens, action_token_mask
        )
        token_logprobs, _ = compute_token_logprobs(
            replay_logits,
            action_tokens,
            metadata["action_logprob_mask"],
            temperature=temperature,
            compute_entropy=False,
        )
        result["token_logprobs"] = token_logprobs
    return result


def replay_action_logits(
    policy: Any,
    forward_inputs: dict[str, torch.Tensor],
    action_tokens: torch.Tensor,
    action_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay sampled FAST tokens with one teacher-forcing policy forward.

    Args:
        policy: LeRobot PI0-Fast policy.
        forward_inputs: Cached rollout observations and language inputs.
        action_tokens: Tokens sampled during rollout.
        action_token_mask: Mask identifying generated token positions.

    Returns:
        Per-token logits and the final hidden state used for diagnostics.
    """
    model = policy.model
    images, img_masks, tokens, masks = _condition_prefix(policy, forward_inputs)
    lm_head = model.paligemma_with_expert.paligemma.lm_head
    action_tokens = action_tokens.to(device=tokens.device, dtype=torch.long)
    action_token_mask = action_token_mask.to(device=tokens.device, dtype=torch.bool)

    single_token = action_tokens.shape[1] == 1
    prefix_embs, prefix_pad_masks, prefix_att_masks, _, num_fast_embs = (
        model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None if single_token else action_tokens[:, :-1],
            fast_action_masks=None if single_token else action_token_mask[:, :-1],
        )
    )
    prefix_embs = _ensure_prefix_precision(model, prefix_embs)
    prefix_out = _forward_embeds(model, prefix_embs, prefix_pad_masks, prefix_att_masks)

    if single_token:
        return lm_head(prefix_out[:, -1:, :]), prefix_out[:, -1, :]

    logits = lm_head(prefix_out[:, -num_fast_embs - 1 :, :])
    return logits, prefix_out[:, -1, :]
