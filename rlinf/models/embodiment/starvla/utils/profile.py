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

from typing import Any, Optional

import torch.nn as nn

ACTION_HEAD_TYPES = {"fast", "oft", "adapter", "pi", "gr00t", "dual"}
STATE_ADAPTER_TYPES = {"none", "adapter", "pi", "gr00t", "dual"}
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
    "pi_chain_actions_preclip",
    "pi_t_bucket_indices",
    "pi_actions_pre",
    "pi_actions_next_preclip",
    "pi_t_bucket_index",
    "pi_denoise_inds",
    "pi_num_steps",
    "pi_sample_actions",
    "pi_step_std",
    "gr00t_chain_actions",
    "gr00t_chain_actions_preclip",
    "gr00t_t_bucket_indices",
    "gr00t_actions_pre",
    "gr00t_actions_next_preclip",
    "gr00t_t_bucket_index",
    "gr00t_denoise_inds",
    "gr00t_num_steps",
    "gr00t_sample_actions",
    "gr00t_step_std",
    "dual_chain_actions",
    "dual_chain_actions_preclip",
    "dual_t_bucket_indices",
    "dual_actions_pre",
    "dual_actions_next_preclip",
    "dual_t_bucket_index",
    "dual_denoise_inds",
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


def infer_policy_profile(starvla_model: nn.Module) -> dict[str, str]:
    """Build full policy profile used by dispatch and runtime checks."""
    cfg = getattr(starvla_model, "config", None)
    framework_name = None
    framework_vlm = None
    framework_action_head = None

    for candidate in (
        getattr(starvla_model, "framework_name", None),
        getattr(cfg, "framework_name", None) if cfg is not None else None,
        getattr(getattr(cfg, "framework", None), "framework_name", None)
        if cfg is not None
        else None,
    ):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            framework_name = text
            break

    if framework_name is not None:
        framework_norm = "".join(
            ch for ch in str(framework_name).lower() if ch.isalnum()
        )

        vlm_matches = [
            vlm for token, vlm in _VLM_TYPE_BY_TOKEN.items() if token in framework_norm
        ]
        vlm_matches = list(dict.fromkeys(vlm_matches))
        if len(vlm_matches) > 1 or len(vlm_matches) == 0:
            raise RuntimeError(f"Ambiguous framework_name={framework_name!r}.")
        if len(vlm_matches) == 1:
            framework_vlm = vlm_matches[0]

        head_matches = [
            head
            for token, head in _ACTION_HEAD_BY_TOKEN.items()
            if token in framework_norm
        ]
        head_matches = list(dict.fromkeys(head_matches))
        if len(head_matches) > 1 or len(head_matches) == 0:
            raise RuntimeError(f"Ambiguous framework_name={framework_name!r}.")
        if len(head_matches) == 1:
            framework_action_head = head_matches[0]

    framework_cfg = getattr(cfg, "framework", None) if cfg is not None else None
    qwenvl_cfg = (
        getattr(framework_cfg, "qwenvl", None) if framework_cfg is not None else None
    )
    base_vlm = getattr(qwenvl_cfg, "base_vlm", None) if qwenvl_cfg is not None else None
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

    if framework_vlm is None:
        raise RuntimeError(
            "Unable to infer VLM type for starVLA model. "
            "Set 'framework_name' (e.g. QwenDual/FlorenceGR00T/CosmosPI) or "
            "'config.framework.qwenvl.base_vlm' explicitly. "
            f"framework_name={framework_name!r}, base_vlm={base_vlm!r}."
        )

    if framework_action_head is None:
        raise RuntimeError(
            "Unable to infer action_head_type for starVLA model. "
            f"cfg_head_type is {cfg_head_type!r}."
            "Set 'framework_name' (e.g. QwenDual/FlorenceGR00T/CosmosPI) or "
            "'config.framework.action_model.action_head_type' explicitly to one of: "
            "fast/oft/adapter/pi/gr00t/dual."
        )

    if framework_action_head in {"adapter", "pi", "gr00t", "dual"}:
        state_adapter_type = framework_action_head
    else:
        state_adapter_type = "none"
    return {
        "vlm_type": framework_vlm,
        "action_head_type": framework_action_head,
        "state_adapter_type": state_adapter_type,
    }


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
    if action_head_name is None:
        action_head_name = infer_policy_profile(starvla_model)["action_head_type"]
    if action_head_name not in ACTION_HEAD_TYPES:
        raise ValueError(
            f"Unknown action_head_name={action_head_name!r}. Expected one of {sorted(ACTION_HEAD_TYPES)}."
        )

    cfg = getattr(starvla_model, "config", None)
    framework_cfg = getattr(cfg, "framework", None) if cfg is not None else None
    action_cfg = (
        getattr(framework_cfg, "action_model", None)
        if framework_cfg is not None
        else None
    )

    def cfg_int(key: str) -> Optional[int]:
        value = getattr(action_cfg, key, None) if action_cfg is not None else None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if action_head_name == "fast":
        tokenizer = getattr(
            getattr(starvla_model, "action_model", None), "fast_tokenizer", None
        )
        time_horizon = getattr(tokenizer, "time_horizon", None)
        if time_horizon is not None:
            return int(time_horizon)
        future = cfg_int("future_action_window_size")
        if future is not None:
            return future + 1

    if action_head_name in {"oft", "pi", "gr00t", "dual"}:
        past = cfg_int("past_action_window_size")
        future = cfg_int("future_action_window_size")
        if past is not None and future is not None:
            return past + 1 + future

    if action_head_name == "adapter":
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
