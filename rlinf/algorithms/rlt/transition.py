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
from rlinf.utils.nested_dict_process import copy_dict_tensor

RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")
RLT_TRANSITION_PREFIX = "rlt_transition_"


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True for envs that store one replay row per env step."""
    train_env_cfg = cfg.env.get("train", None)
    if train_env_cfg is None:
        return False
    try:
        return (
            SupportedEnvType(train_env_cfg.get("env_type", ""))
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
    return copy_dict_tensor(
        {key: forward_inputs[f"{prefix}{key}"] for key in RLT_OBS_KEYS}
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
