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

"""Helpers to infer starVLA policy wiring from checkpoint metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn

ACTION_HEAD_TYPES = {"fast", "oft", "adapter", "pi", "gr00t", "dual"}
STATE_ADAPTER_TYPES = {"none", "adapter", "pi", "gr00t", "dual"}
CONTINUOUS_ACTION_HEAD_TYPES = set(ACTION_HEAD_TYPES)
_VLM_TYPE_BY_TOKEN: dict[str, str] = {
    "qwen": "qwen",
    "florence": "florence",
    "cosmos": "cosmos",
}
_ACTION_HEAD_BY_TOKEN: dict[str, str] = {
    "fast": "fast",
    "oft": "oft",
    "adapter": "adapter",
    "pi": "pi",
    "gr00t": "gr00t",
    "groot": "gr00t",
    "dual": "dual",
}


@dataclass(frozen=True)
class PolicyProfile:
    """Resolved policy wiring: VLM family, action head, and state-adapter mode."""

    vlm_type: str
    action_head_type: str
    state_adapter_type: str
    is_continuous_action: bool


RL_BATCH_TENSOR_KEYS_TO_IGNORE: set[str] = {
    "action",
    "action_tokens",
    "prev_logprobs",
    "prev_values",
    "advantages",
    "returns",
    "loss_mask",
    "loss_mask_sum",
    "rewards",
    "dones",
    "terminations",
    "truncations",
    "ref_logprobs",
    "recompute_prev_logprobs",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "max_new_tokens",
    "max_length",
    "obs",
    "next_obs",
    "transitions",
    "env_info",
    "state",
    "states",
    "pi_chain_actions",
    "pi_t_bucket_indices",
    "pi_num_steps",
    "pi_sample_actions",
    "pi_step_std",
    "gr00t_chain_actions",
    "gr00t_t_bucket_indices",
    "gr00t_num_steps",
    "gr00t_sample_actions",
    "gr00t_step_std",
    "dual_chain_actions",
    "dual_t_bucket_indices",
    "dual_num_steps",
    "dual_sample_actions",
    "dual_step_std",
}


def resolve_vlm_interface(starvla_model: nn.Module) -> Any:
    """Return the starVLA VLM interface object expected by the wrapper."""
    iface = getattr(starvla_model, "qwen_vl_interface", None)
    if iface is not None:
        return iface
    raise RuntimeError(
        "Cannot find VLM interface on starVLA model: expected 'qwen_vl_interface'."
    )


def _resolve_framework_name(starvla_model: nn.Module) -> Optional[str]:
    """Fetch configured framework name from model/config candidates."""
    candidates = [getattr(starvla_model, "framework_name", None)]
    cfg = getattr(starvla_model, "config", None)
    if cfg is not None:
        candidates.append(getattr(cfg, "framework_name", None))
        framework_cfg = getattr(cfg, "framework", None)
        if framework_cfg is not None:
            candidates.append(getattr(framework_cfg, "framework_name", None))

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def _infer_from_framework_name(
    starvla_model: nn.Module,
) -> tuple[Optional[str], Optional[str]]:
    """Parse framework name tokens into '(vlm_type, action_head_type)'."""
    framework_name = _resolve_framework_name(starvla_model)
    if framework_name is None:
        return None, None
    norm = "".join(ch for ch in str(framework_name).lower() if ch.isalnum())

    vlm_matches = [vlm for token, vlm in _VLM_TYPE_BY_TOKEN.items() if token in norm]
    vlm_matches = list(dict.fromkeys(vlm_matches))
    if len(vlm_matches) > 1:
        raise RuntimeError(
            f"Ambiguous framework_name={framework_name!r}: multiple VLM tokens matched {vlm_matches}."
        )
    vlm_type = vlm_matches[0] if vlm_matches else None

    head_matches = [
        head for token, head in _ACTION_HEAD_BY_TOKEN.items() if token in norm
    ]
    head_matches = list(dict.fromkeys(head_matches))
    if len(head_matches) > 1:
        raise RuntimeError(
            f"Ambiguous framework_name={framework_name!r}: multiple action-head tokens matched {head_matches}."
        )
    action_head = head_matches[0] if head_matches else None
    return vlm_type, action_head


def infer_vlm_type(starvla_model: nn.Module) -> str:
    """Infer VLM family used by the checkpoint."""
    framework_vlm, _ = _infer_from_framework_name(starvla_model)
    if framework_vlm is not None:
        return framework_vlm

    cfg = getattr(starvla_model, "config", None)
    framework_cfg = getattr(cfg, "framework", None) if cfg is not None else None
    qwenvl_cfg = (
        getattr(framework_cfg, "qwenvl", None) if framework_cfg is not None else None
    )
    base_vlm = getattr(qwenvl_cfg, "base_vlm", None) if qwenvl_cfg is not None else None
    if base_vlm is not None:
        norm = "".join(ch for ch in str(base_vlm).lower() if ch.isalnum())
        matches = [vlm for token, vlm in _VLM_TYPE_BY_TOKEN.items() if token in norm]
        matches = list(dict.fromkeys(matches))
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous base_vlm={base_vlm!r}: multiple VLM tokens matched {matches}."
            )
        if len(matches) == 1:
            return matches[0]

    framework_name = _resolve_framework_name(starvla_model)
    raise RuntimeError(
        "Unable to infer VLM type for starVLA model. "
        "Set 'framework_name' (e.g. QwenDual/FlorenceGR00T/CosmosPI) or "
        "'config.framework.qwenvl.base_vlm' explicitly. "
        f"framework_name={framework_name!r}, base_vlm={base_vlm!r}."
    )


def resolve_fast_action_token_range(starvla_model: nn.Module) -> tuple[int, int]:
    """Read FAST action-token id range from VLM interface."""
    iface = resolve_vlm_interface(starvla_model)
    token_min = getattr(iface, "_ACTION_TOKEN_MIN", None)
    token_max = getattr(iface, "_ACTION_TOKEN_MAX", None)
    if token_min is None or token_max is None:
        raise RuntimeError(
            "FAST action head requires '_ACTION_TOKEN_MIN/_ACTION_TOKEN_MAX' on the VLM interface."
        )
    return int(token_min), int(token_max)


def resolve_fast_max_action_tokens(
    policy: Any,
    *,
    default: int = 256,
    env_key: str = "RLINF_QWENFAST_MAX_ACTION_TOKENS",
) -> int:
    """Resolve FAST max generated token length from policy/env settings."""
    configured = getattr(policy, "qwenfast_max_action_tokens", None)
    try:
        max_action_tokens = int(configured or 0)
    except (TypeError, ValueError):
        max_action_tokens = 0

    if max_action_tokens <= 0:
        env_raw = os.environ.get(env_key, str(default))
        try:
            max_action_tokens = int(env_raw)
        except (TypeError, ValueError):
            max_action_tokens = 0

    if max_action_tokens <= 0:
        raise ValueError(
            f"Invalid FAST max action token setting: qwenfast_max_action_tokens={configured!r}, "
            f"{env_key}={os.environ.get(env_key)!r}."
        )
    return max_action_tokens


def resolve_vlm_pad_token_id(starvla_model: nn.Module, default: int = 0) -> int:
    """Resolve pad token id from VLM config with safe fallback."""
    iface = resolve_vlm_interface(starvla_model)
    model_cfg = getattr(getattr(iface, "model", None), "config", None)
    if model_cfg is None:
        return int(default)
    try:
        return int(getattr(model_cfg, "pad_token_id", default) or default)
    except (TypeError, ValueError):
        return int(default)


def infer_action_head_type(starvla_model: nn.Module) -> str:
    """Infer action-head type used by the checkpoint."""
    _, framework_action_head = _infer_from_framework_name(starvla_model)
    if framework_action_head is not None:
        return framework_action_head

    cfg = getattr(starvla_model, "config", None)
    framework_cfg = getattr(cfg, "framework", None) if cfg is not None else None
    action_cfg = (
        getattr(framework_cfg, "action_model", None)
        if framework_cfg is not None
        else None
    )
    cfg_head_type = (
        getattr(action_cfg, "action_head_type", None)
        if action_cfg is not None
        else None
    )
    if cfg_head_type is not None:
        cfg_head_type = str(cfg_head_type).strip().lower()
        if cfg_head_type in ACTION_HEAD_TYPES:
            return cfg_head_type
        raise RuntimeError(
            "Invalid 'config.framework.action_model.action_head_type'. "
            f"Got {cfg_head_type!r}, expected one of {sorted(ACTION_HEAD_TYPES)}."
        )

    raise RuntimeError(
        "Unable to infer action_head_type for starVLA model. "
        "Set 'framework_name' (e.g. QwenDual/FlorenceGR00T/CosmosPI) or "
        "'config.framework.action_model.action_head_type' explicitly to one of: "
        "fast/oft/adapter/pi/gr00t/dual."
    )


def infer_state_adapter_type(action_head_type: str) -> str:
    """Map action-head type to required state-adapter mode."""
    if action_head_type not in ACTION_HEAD_TYPES:
        raise ValueError(
            f"Unknown action_head_type={action_head_type!r}. Expected one of {sorted(ACTION_HEAD_TYPES)}."
        )
    if action_head_type in {"adapter", "pi", "gr00t", "dual"}:
        return action_head_type
    return "none"


def infer_policy_profile(starvla_model: nn.Module) -> PolicyProfile:
    """Build full policy profile used by dispatch and runtime checks."""
    action_head_type = infer_action_head_type(starvla_model)
    return PolicyProfile(
        vlm_type=infer_vlm_type(starvla_model),
        action_head_type=action_head_type,
        state_adapter_type=infer_state_adapter_type(action_head_type),
        is_continuous_action=(action_head_type in CONTINUOUS_ACTION_HEAD_TYPES),
    )


def infer_hidden_size(starvla_model: nn.Module) -> int:
    """Infer hidden width for value-head construction."""
    vlm_iface = resolve_vlm_interface(starvla_model)
    hf_model = getattr(vlm_iface, "model", None)
    if hf_model is None:
        raise RuntimeError("VLM interface has no .model; cannot build value head.")
    cfg = getattr(hf_model, "config", None)
    if cfg is None:
        raise RuntimeError("HF model has no config; cannot infer hidden size.")

    for key in ("hidden_size", "d_model", "n_embd"):
        if hasattr(cfg, key):
            val = getattr(cfg, key)
            if isinstance(val, int) and val > 0:
                return val
    raise RuntimeError(f"Cannot infer hidden size from HF config: {type(cfg)}")


def resolve_action_chunk_len(
    starvla_model: nn.Module,
    num_action_chunks: int,
    *,
    action_head_name: Optional[str] = None,
) -> int:
    """Resolve rollout/train action horizon for a specific action head."""
    head = action_head_name or infer_action_head_type(starvla_model)
    if head not in ACTION_HEAD_TYPES:
        raise ValueError(
            f"Unknown action_head_name={head!r}. Expected one of {sorted(ACTION_HEAD_TYPES)}."
        )

    cfg = getattr(starvla_model, "config", None)
    framework_cfg = getattr(cfg, "framework", None) if cfg is not None else None
    action_cfg = (
        getattr(framework_cfg, "action_model", None)
        if framework_cfg is not None
        else None
    )

    def cfg_int(key: str) -> Optional[int]:
        if action_cfg is None:
            return None
        value = getattr(action_cfg, key, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if head == "fast":
        tokenizer = getattr(
            getattr(starvla_model, "action_model", None), "fast_tokenizer", None
        )
        time_horizon = getattr(tokenizer, "time_horizon", None)
        if time_horizon is not None:
            return int(time_horizon)
        future = cfg_int("future_action_window_size")
        if future is not None:
            return future + 1

    if head in {"oft", "pi", "gr00t", "dual"}:
        past = cfg_int("past_action_window_size")
        future = cfg_int("future_action_window_size")
        if past is not None and future is not None:
            return past + 1 + future

    if head == "adapter":
        num_actions_chunk = cfg_int("num_actions_chunk")
        if num_actions_chunk is not None:
            return num_actions_chunk

    chunk_len = getattr(starvla_model, "chunk_len", None)
    if chunk_len is not None:
        try:
            return int(chunk_len)
        except (TypeError, ValueError):
            pass

    return int(num_action_chunks)


def iter_gradient_checkpointing_targets(starvla_model: nn.Module) -> list[nn.Module]:
    """List modules that may expose gradient-checkpointing APIs."""
    targets: list[nn.Module] = []

    def add(module: Optional[nn.Module]) -> None:
        if module is None or not isinstance(module, nn.Module):
            return
        if module in targets:
            return
        targets.append(module)

    add(starvla_model)
    vlm_iface = resolve_vlm_interface(starvla_model)
    add(vlm_iface)
    add(getattr(vlm_iface, "model", None) if vlm_iface is not None else None)
    add(getattr(starvla_model, "action_model", None))
    return targets
