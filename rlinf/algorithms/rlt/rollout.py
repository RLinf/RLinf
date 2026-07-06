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

from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX


def _normalize_rlt_switch_flags(
    actions: torch.Tensor,
    rlt_switch_flags: torch.Tensor | None,
) -> torch.Tensor:
    if rlt_switch_flags is None:
        rlt_switch_flags = torch.zeros(
            (actions.shape[0], actions.shape[1]),
            dtype=torch.bool,
            device=actions.device,
        )
    else:
        rlt_switch_flags = torch.as_tensor(
            rlt_switch_flags, device=actions.device
        ).bool()
    if rlt_switch_flags.dim() == 1:
        rlt_switch_flags = rlt_switch_flags[:, None]
    if rlt_switch_flags.shape[1] > 1:
        rlt_switch_flags = rlt_switch_flags[:, -1:]
    if actions.shape[1] > 1:
        rlt_switch_flags = rlt_switch_flags.expand(-1, actions.shape[1])
    return rlt_switch_flags.reshape(actions.shape[0], actions.shape[1], 1)


def predict_rlt_stage2_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    rlt_switch_flags: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_stage2_obs(env_obs)
        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        rlt_switch_flags = _normalize_rlt_switch_flags(actions, rlt_switch_flags)
        ref_actions = result["forward_inputs"]["ref_chunk"].to(
            device=actions.device, dtype=actions.dtype
        )
        actions = torch.where(
            rlt_switch_flags,
            actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
        ).contiguous()
        result["forward_inputs"]["action"] = actions.reshape(
            actions.shape[0], -1
        ).contiguous()

        transition_obs = rlt_obs
        if final_obs is not None:
            transition_obs = feature_model.extract_rlt_stage2_obs(final_obs)
        for key in RLT_OBS_KEYS:
            result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[
                key
            ]

    result["expert_label_flag"] = False
    return actions, result
