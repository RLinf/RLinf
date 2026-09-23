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


# LeRobot 8a74e0ac (the documented pi0_fast pin) keeps prepare_attention_masks_4d
# helper on PI0FastPytorch. Later LeRobot extracted it to lerobot.policies.common,
# which does not exist at that commit.
_OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def _prepare_attention_masks_4d(
    att_2d_masks: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Expand boolean 2D attention masks to the additive 4D transformer layout."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    result = torch.where(att_2d_masks_4d, 0.0, _OPENPI_ATTENTION_MASK_VALUE)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def _prepare_sample_inputs(policy, batch: dict[str, Any]):
    """Prepare LeRobot sample_actions_fast* inputs without appending BOS.

    ``PI0FastPytorch.sample_actions_fast`` / ``sample_actions_fast_kv_cache``
    append the BOS token internally. Replay still has to add it itself because
    ``embed_prefix_fast`` does not.
    """
    images, img_masks = policy._preprocess_images(batch)
    device = next(policy.parameters()).device
    tokens = batch["observation.language.tokens"].to(device=device, dtype=torch.long)
    masks = batch["observation.language.attention_mask"].to(
        device=device, dtype=torch.bool
    )
    return images, img_masks, tokens, masks


def _generation_mask_from_native_tokens(
    action_tokens: torch.Tensor, *, greedy: bool
) -> torch.Tensor:
    """Mark tokens actually emitted by LeRobot sampling.

    Greedy KV-cache decoding may stop once every row has emitted ``|`` and leave
    a shared zero-filled suffix. Stochastic decoding always fills the buffer, so
    every position is treated as generated.
    """
    if not greedy:
        return torch.ones_like(action_tokens, dtype=torch.bool)
    trailing_pad_columns = (action_tokens == 0).all(dim=0)
    generated = torch.ones_like(action_tokens, dtype=torch.bool)
    if bool(trailing_pad_columns.all()):
        return torch.zeros_like(action_tokens, dtype=torch.bool)
    last_generated = int((~trailing_pad_columns).nonzero(as_tuple=False)[-1].item())
    generated[:, last_generated + 1 :] = False
    return generated


def _find_first_subsequence(values: list[int], needle: list[int], *, start: int) -> int:
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def build_action_sequence_metadata(
    policy: Any,
    action_tokens: torch.Tensor,
    generation_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Validate native FAST output and build the PPO mask through the first `|`."""
    device = action_tokens.device
    tokenizer = policy._paligemma_tokenizer
    prefix_ids = tokenizer.encode("Action: ", add_special_tokens=False)
    if not prefix_ids:
        raise ValueError("pi0_fast tokenizer produced an empty Action prefix.")
    end_ids = tokenizer.encode("|", add_special_tokens=False)
    if not end_ids:
        raise ValueError("pi0_fast tokenizer produced an empty action end marker.")
    bpe = policy.action_tokenizer.bpe_tokenizer
    if hasattr(bpe, "get_vocab_size"):
        bpe_vocab_size = int(bpe.get_vocab_size())
    elif hasattr(bpe, "get_vocab"):
        bpe_vocab_size = len(bpe.get_vocab())
    else:
        raise AttributeError("FAST BPE tokenizer does not expose its vocabulary size.")
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

    Sampling is delegated to LeRobot's ``sample_actions_fast`` /
    ``sample_actions_fast_kv_cache``. RLinf only detokenizes, builds the GRPO
    mask, and teacher-forces the sampled tokens for behavior logprobs.

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
    sample_temperature = temperature if do_sample and temperature > 0 else 0.0
    images, img_masks, tokens, masks = _prepare_sample_inputs(policy, batch)
    model = policy.model
    use_kv_cache = bool(getattr(policy.config, "use_kv_cache", True))
    sample_fn = (
        model.sample_actions_fast_kv_cache
        if use_kv_cache
        else model.sample_actions_fast
    )
    restore_gradient_checkpointing = use_kv_cache and bool(
        getattr(model, "gradient_checkpointing_enabled", False)
    )
    try:
        if restore_gradient_checkpointing:
            model.gradient_checkpointing_disable()
        action_tokens = sample_fn(
            images,
            img_masks,
            tokens,
            masks,
            max_decoding_steps=max_action_tokens,
            temperature=sample_temperature,
        )
    finally:
        if restore_gradient_checkpointing:
            model.gradient_checkpointing_enable()
    action_token_mask = _generation_mask_from_native_tokens(
        action_tokens, greedy=sample_temperature <= 0
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
            temperature=temperature if temperature > 0 else 1.0,
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
    images, img_masks, tokens, masks = _prepare_sample_inputs(policy, forward_inputs)
    bos_token = torch.full(
        (tokens.shape[0], 1),
        policy._paligemma_tokenizer.bos_token_id,
        dtype=torch.long,
        device=tokens.device,
    )
    tokens = torch.cat([tokens, bos_token], dim=1)
    masks = torch.cat(
        [
            masks,
            torch.ones((masks.shape[0], 1), dtype=torch.bool, device=tokens.device),
        ],
        dim=1,
    )
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
    q_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj
    if q_proj.weight.dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = _prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        adarms_cond=[None, None],
    )

    if single_token:
        return lm_head(prefix_out[:, -1:, :]), prefix_out[:, -1, :]

    logits = lm_head(prefix_out[:, -num_fast_embs - 1 :, :])
    return logits, prefix_out[:, -1, :]
