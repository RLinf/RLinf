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

from typing import Any, Optional

import torch

from rlinf.envs import SupportedEnvType
from rlinf.utils.nested_dict_process import copy_dict_tensor

RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")
RLT_TRANSITION_PREFIX = "rlt_transition_"


def _train_env_type(cfg: Any) -> Optional[SupportedEnvType]:
    train_env_cfg = cfg.env.get("train", None)
    if train_env_cfg is None:
        return None
    try:
        return SupportedEnvType(train_env_cfg.get("env_type", ""))
    except ValueError:
        return None


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True when RLT stores one replay row per env step.

    Default: enabled for ``maniskill_rlt``. Override with
    ``algorithm.rlt_transition_replay`` (used by realworld Stage2 to match
    simulator critical-phase filtering).
    """
    algo_cfg = cfg.get("algorithm", None)
    if algo_cfg is not None and "rlt_transition_replay" in algo_cfg:
        return bool(algo_cfg.get("rlt_transition_replay"))
    return _train_env_type(cfg) == SupportedEnvType.MANISKILL_RLT


def use_maniskill_rlt_env(cfg: Any) -> bool:
    """Return True when the train env is ManiSkill RLT (expert route path)."""
    return _train_env_type(cfg) == SupportedEnvType.MANISKILL_RLT


def extract_rlt_obs_from_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    transition: bool = False,
) -> dict[str, Any]:
    prefix = RLT_TRANSITION_PREFIX if transition else ""
    missing = [
        f"{prefix}{key}"
        for key in RLT_OBS_KEYS
        if f"{prefix}{key}" not in forward_inputs
    ]
    if missing:
        raise ValueError(
            f"Missing RLT forward_inputs keys: {missing}. Ensure "
            "rollout.rlt_feature_model is configured and the rollout worker "
            "populates RLT features."
        )
    return copy_dict_tensor(
        {key: forward_inputs[f"{prefix}{key}"] for key in RLT_OBS_KEYS}
    )


def update_rlt_transitions(
    stage_id: int,
    pending_obs: list[dict[str, Any] | None],
    trajectory_builders: list[Any],
    policy_output: Any,
    *,
    cache_current: bool,
    intervene_actions: torch.Tensor | None = None,
    intervene_flags: torch.Tensor | None = None,
) -> None:
    if pending_obs[stage_id] is not None:
        if intervene_actions is not None and intervene_flags is not None:
            current_obs = pending_obs[stage_id]
            ref_chunk = current_obs["ref_chunk"]
            batch_size = ref_chunk.shape[0]
            flags = intervene_flags.reshape(batch_size, -1, 1).to(
                device=ref_chunk.device, dtype=torch.bool
            )
            human_actions = intervene_actions.reshape(batch_size, flags.shape[1], -1)
            action_dim = human_actions.shape[-1]
            ref_actions = ref_chunk.reshape(batch_size, -1, action_dim).clone()
            ref_actions[:, : flags.shape[1]] = torch.where(
                flags,
                human_actions.to(device=ref_chunk.device, dtype=ref_chunk.dtype),
                ref_actions[:, : flags.shape[1]],
            )
            current_obs["ref_chunk"] = ref_actions.reshape_as(ref_chunk)
        next_obs = extract_rlt_obs_from_forward_inputs(
            policy_output.forward_inputs,
            transition=True,
        )
        trajectory_builders[stage_id].append_transitions(
            pending_obs[stage_id],
            next_obs,
        )
        pending_obs[stage_id] = None

    if cache_current:
        pending_obs[stage_id] = extract_rlt_obs_from_forward_inputs(
            policy_output.forward_inputs
        )
