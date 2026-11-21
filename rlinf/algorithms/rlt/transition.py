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

import os
from typing import Any

import torch

from rlinf.envs import SupportedEnvType
from rlinf.envs.layout import EnvLayout
from rlinf.utils.nested_dict_process import copy_dict_tensor

RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")
# Carried when the feature model emits them (rtj joint Stage2): the critic's
# TCP-space proprio, and the reference chunk's TCP rows (stored, never an input).
RLT_OPTIONAL_OBS_KEYS = (
    "critic_proprio",
    "ref_chunk_tcp",
    "tactile_images",
    "tactile_valid",
    "tactile_age_ms",
)
_OVERWRITE_REF_WITH_HUMAN = os.environ.get("RLT_OVERWRITE_REF_WITH_HUMAN", "0") == "1"
_REF_OFFSET = int(os.environ.get("RLT_REF_OFFSET", "0"))
RLT_TRANSITION_PREFIX = "rlt_transition_"


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True for envs that store one replay row per env step.

    Raises:
        ValueError: If only some of the train envs store a row per env step;
            the actor keeps one replay layout.
    """
    env_cfgs = EnvLayout.env_cfgs(cfg.env.get("train", None), "env.train")
    per_step = {path: _is_maniskill_rlt(env_cfg) for path, env_cfg in env_cfgs.items()}
    if len(set(per_step.values())) > 1:
        raise ValueError(
            "RLT replay needs every train env to be maniskill_rlt or none of "
            f"them, got {per_step}."
        )
    return any(per_step.values())


def _is_maniskill_rlt(env_cfg: Any) -> bool:
    try:
        return (
            SupportedEnvType(env_cfg.get("env_type", ""))
            == SupportedEnvType.MANISKILL_RLT
        )
    except ValueError:
        return False


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
    keys = RLT_OBS_KEYS + tuple(
        k for k in RLT_OPTIONAL_OBS_KEYS if f"{prefix}{k}" in forward_inputs
    )
    return copy_dict_tensor({key: forward_inputs[f"{prefix}{key}"] for key in keys})


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
        if (
            _OVERWRITE_REF_WITH_HUMAN
            and intervene_actions is not None
            and intervene_flags is not None
        ):
            current_obs = pending_obs[stage_id]
            ref_chunk = current_obs["ref_chunk"]
            batch_size = ref_chunk.shape[0]
            action_dim = int(ref_chunk.shape[-1]) if ref_chunk.dim() == 3 else 14
            human_actions = intervene_actions.reshape(batch_size, -1, action_dim)
            chunk_len = human_actions.shape[1]
            flags = (
                intervene_flags.reshape(batch_size, chunk_len, -1)
                .to(device=ref_chunk.device)
                .bool()
                .any(dim=-1, keepdim=True)
            )
            ref_actions = ref_chunk.reshape(batch_size, -1, action_dim).clone()
            o = _REF_OFFSET
            if o + chunk_len > ref_actions.shape[1]:
                raise ValueError(
                    "RLT_REF_OFFSET=%d + chunk_len %d exceeds ref_chunk len %d"
                    % (o, chunk_len, ref_actions.shape[1])
                )
            ref_actions[:, o : o + chunk_len] = torch.where(
                flags,
                human_actions.to(device=ref_chunk.device, dtype=ref_chunk.dtype),
                ref_actions[:, o : o + chunk_len],
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


def apply_rlt_interventions(
    obs: dict[str, Any],
    actions: torch.Tensor | None,
    flags: torch.Tensor | None,
) -> None:
    """Replace reference actions with interventions executed by the environment."""
    if actions is None or flags is None:
        return

    ref_chunk = obs["ref_chunk"]
    batch_size = ref_chunk.shape[0]
    flags = flags.reshape(batch_size, -1, 1).to(
        device=ref_chunk.device, dtype=torch.bool
    )
    actions = actions.reshape(batch_size, flags.shape[1], -1).to(
        device=ref_chunk.device, dtype=ref_chunk.dtype
    )
    ref_actions = ref_chunk.reshape(batch_size, -1, actions.shape[-1]).clone()
    ref_actions[:, : flags.shape[1]] = torch.where(
        flags,
        actions,
        ref_actions[:, : flags.shape[1]],
    )
    obs["ref_chunk"] = ref_actions.reshape_as(ref_chunk)
