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

from typing import Any

import torch

from rlinf.envs import SupportedEnvType
from rlinf.envs.layout import EnvLayout
from rlinf.models.embodiment.prefix_features import PREFIX_OBS_KEYS
from rlinf.utils.nested_dict_process import copy_dict_tensor

PREFIX_TRANSITION_PREFIX = "rlt_transition_"


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True for envs that store one replay row per env step.

    Raises:
        ValueError: If only some of the train envs store a row per env step;
            the actor keeps one replay layout.
    """
    algo_cfg = cfg.get("algorithm", None)
    if algo_cfg is not None and "rlt_transition_replay" in algo_cfg:
        return bool(algo_cfg.get("rlt_transition_replay"))
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


def use_maniskill_rlt_env(cfg: Any) -> bool:
    """Return whether every environment in the active layout is ManiSkill RLT."""
    mode = "eval" if cfg.runner.get("only_eval", False) else "train"
    env_cfgs = EnvLayout.env_cfgs(cfg.env.get(mode, None), f"env.{mode}")
    return bool(env_cfgs) and all(
        _is_maniskill_rlt(env_cfg) for env_cfg in env_cfgs.values()
    )


def extract_prefix_obs_from_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    transition: bool = False,
) -> dict[str, Any]:
    prefix = PREFIX_TRANSITION_PREFIX if transition else ""
    missing = [
        f"{prefix}{key}"
        for key in PREFIX_OBS_KEYS
        if f"{prefix}{key}" not in forward_inputs
    ]
    if missing:
        raise ValueError(
            f"Missing Prefix forward_inputs keys: {missing}. Ensure "
            "rollout.prefix_feature_model is configured and the rollout worker "
            "populates Prefix features."
        )
    return copy_dict_tensor(
        {key: forward_inputs[f"{prefix}{key}"] for key in PREFIX_OBS_KEYS}
    )


def apply_prefix_interventions(
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
