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


def update_rlt_transitions(
    stage_id: int,
    pending_obs: list[dict[str, Any] | None],
    rollout_results: list[Any],
    rollout_result: Any,
    *,
    cache_current: bool,
) -> None:
    if pending_obs[stage_id] is not None:
        next_obs = extract_rlt_obs_from_forward_inputs(
            rollout_result.forward_inputs,
            transition=True,
        )
        rollout_results[stage_id].append_transitions(
            pending_obs[stage_id],
            next_obs,
        )
        pending_obs[stage_id] = None

    if cache_current:
        pending_obs[stage_id] = extract_rlt_obs_from_forward_inputs(
            rollout_result.forward_inputs
        )
