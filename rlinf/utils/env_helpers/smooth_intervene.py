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

"""Realworld smooth-intervene helpers for EnvWorker orchestration.

Bypasses policy inference across action-chunk boundaries while human teleop
continues. Requires PICO (``env.train.use_pico=True``); SpaceMouse is not
supported. Env only supplies hold actions; this module owns PolicyOutput dummy
construction and per-stage continue/skip state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.schema.embodied_types import PolicyOutput
from rlinf.envs.utils import get_env_attr

# Maps forward_inputs observation keys → env_obs keys (from obs_processor).
OBS_KEY_FROM_ENV_OBS: dict[str, str] = {
    "observation/image": "main_images",
    "observation/state": "states",
    "observation/extra_view_image": "extra_view_images",
    "observation/wrist_image": "wrist_images",
}


def should_continue_smooth_intervene(
    intervene_flags: torch.Tensor | None, dones: torch.Tensor
) -> bool:
    if intervene_flags is None:
        return False
    return bool(intervene_flags[:, -1].any().item()) and not bool(dones.any().item())


def build_smooth_intervene_policy_output(
    policy_output: PolicyOutput,
    curr_obs: dict[str, Any],
    hold_actions: np.ndarray | torch.Tensor,
) -> PolicyOutput:
    """Build a shape-compatible policy output without running inference.

    Policy statistics are zeroed. Executed actions use ``hold_actions`` so
    inactive arms stay put when the intervention wrapper does not auto-hold.

    The hold is a per-chunk baseline (typically measured TCP at chunk build).
    Mid-chunk release is handled by the intervention wrapper: it keeps the last
    intervened action until the next action chunk begins, rather than snapping
    back to this baseline pose.

    Current observation tensors and prompt tokens are retained for collectors.
    """

    def _zero_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return torch.zeros_like(tensor) if tensor is not None else None

    def _dummy_forward_input(key: str, value: torch.Tensor) -> torch.Tensor | None:
        if key.startswith("observation/"):
            env_obs_key = OBS_KEY_FROM_ENV_OBS.get(key)
            if env_obs_key is None or env_obs_key not in curr_obs:
                raise KeyError(
                    f"Cannot map smooth-intervene forward input {key!r} "
                    "to the current environment observation"
                )
            env_value = curr_obs[env_obs_key]
            if not isinstance(env_value, torch.Tensor):
                raise TypeError(
                    f"Expected curr_obs[{env_obs_key!r}] to be a tensor, "
                    f"got {type(env_value).__name__}"
                )
            return env_value.cpu().contiguous()
        if key in ("tokenized_prompt", "tokenized_prompt_mask"):
            return value.clone()
        return _zero_tensor(value)

    if policy_output.actions is None:
        raise ValueError(
            "smooth_intervene requires action tensors from a real policy output"
        )

    ref_actions = policy_output.actions
    if ref_actions.ndim == 2:
        fallback_shape = ref_actions.shape
    elif ref_actions.ndim == 3:
        fallback_shape = (ref_actions.shape[0], ref_actions.shape[2])
    else:
        raise ValueError(
            "smooth_intervene expects policy actions with shape "
            f"[B, action_dim] or [B, T, action_dim], got {tuple(ref_actions.shape)}"
        )

    hold_actions = torch.as_tensor(
        hold_actions, dtype=ref_actions.dtype, device=ref_actions.device
    )
    if tuple(hold_actions.shape) != tuple(fallback_shape):
        raise ValueError(
            "hold_actions has incompatible shape "
            f"{tuple(hold_actions.shape)}, expected {tuple(fallback_shape)}"
        )
    if ref_actions.ndim == 2:
        dummy_actions = hold_actions
    else:
        dummy_actions = hold_actions.unsqueeze(1).expand_as(ref_actions).contiguous()

    dummy_forward_inputs = {
        key: _dummy_forward_input(key, value)
        for key, value in policy_output.forward_inputs.items()
        if value is not None
    }
    dummy_forward_inputs["action"] = dummy_actions.reshape(
        dummy_actions.shape[0], -1
    ).contiguous()

    return PolicyOutput(
        actions=dummy_actions,
        prev_logprobs=_zero_tensor(policy_output.prev_logprobs),
        prev_values=_zero_tensor(policy_output.prev_values),
        bootstrap_values=_zero_tensor(policy_output.bootstrap_values),
        intervene_flags=_zero_tensor(policy_output.intervene_flags),
        forward_inputs=dummy_forward_inputs,
        versions=_zero_tensor(policy_output.versions),
    )


class SmoothInterveneController:
    """Per-stage state for realworld smooth intervention."""

    def __init__(self, stage_num: int, enabled: bool = False):
        self.enabled = bool(enabled)
        self.stage_num = int(stage_num)
        self.next_intervene_flags = [False for _ in range(self.stage_num)]
        self.last_policy_outputs: list[PolicyOutput | None] = [
            None for _ in range(self.stage_num)
        ]

    @classmethod
    def from_cfg(
        cls,
        cfg: DictConfig,
        *,
        stage_num: int,
        enable_train: bool,
        train_num_envs_per_stage: int,
    ) -> SmoothInterveneController:
        enabled = bool(
            enable_train
            and OmegaConf.select(cfg, "env.train.smooth_intervene", default=False)
        )
        if enabled:
            if OmegaConf.select(cfg, "env.train.env_type") != "realworld":
                raise ValueError(
                    "smooth_intervene requires env.train.env_type to be 'realworld'"
                )
            if train_num_envs_per_stage != 1:
                raise ValueError(
                    "smooth_intervene requires exactly one env per EnvWorker stage"
                )
            if not bool(OmegaConf.select(cfg, "env.train.use_pico", default=False)):
                raise ValueError(
                    "smooth_intervene requires env.train.use_pico=True "
                    "(PICO-only; SpaceMouse is not supported)"
                )
            if bool(OmegaConf.select(cfg, "env.train.use_spacemouse", default=False)):
                raise ValueError(
                    "smooth_intervene does not support SpaceMouse; "
                    "set env.train.use_spacemouse=False and use_pico=True"
                )
        return cls(stage_num=stage_num, enabled=enabled)

    def active_stage_ids(self) -> set[int]:
        if not self.enabled:
            return set()
        return {
            stage_id
            for stage_id, intervene in enumerate(self.next_intervene_flags)
            if intervene
        }

    def is_active(self, stage_id: int) -> bool:
        return self.enabled and self.next_intervene_flags[stage_id]

    def remember_policy_output(
        self, stage_id: int, policy_output: PolicyOutput
    ) -> None:
        self.last_policy_outputs[stage_id] = policy_output

    def build_dummy_policy_output(
        self,
        stage_id: int,
        *,
        env: Any,
        curr_obs: dict[str, Any] | None,
    ) -> PolicyOutput:
        policy_output = self.last_policy_outputs[stage_id]
        if policy_output is None:
            raise ValueError(
                "smooth_intervene requires one real policy output before dummy chunks"
            )
        if curr_obs is None:
            raise ValueError("smooth_intervene requires the current observation")
        if policy_output.actions is None:
            raise ValueError(
                "smooth_intervene requires action tensors from a real policy output"
            )

        ref_actions = policy_output.actions
        if ref_actions.ndim == 2:
            fallback_actions = ref_actions
        elif ref_actions.ndim == 3:
            fallback_actions = ref_actions[:, -1, :]
        else:
            raise ValueError(
                "smooth_intervene expects policy actions with shape "
                f"[B, action_dim] or [B, T, action_dim], got {tuple(ref_actions.shape)}"
            )

        get_hold_actions = get_env_attr(env, "get_hold_actions")
        if not callable(get_hold_actions):
            raise ValueError(
                "smooth_intervene requires the env to expose get_hold_actions()"
            )
        hold_np = get_hold_actions(fallback_actions.detach().cpu().numpy())
        return build_smooth_intervene_policy_output(
            policy_output, curr_obs=curr_obs, hold_actions=hold_np
        )

    def on_chunk_done(
        self,
        stage_id: int,
        intervene_flags: torch.Tensor | None,
        dones: torch.Tensor,
    ) -> bool:
        """Update continue flag. Returns True when rollout send should be skipped."""
        if not self.enabled:
            self.next_intervene_flags[stage_id] = False
            return False
        continue_smooth = should_continue_smooth_intervene(intervene_flags, dones)
        self.next_intervene_flags[stage_id] = continue_smooth
        return continue_smooth
