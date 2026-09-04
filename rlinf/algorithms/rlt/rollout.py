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

from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX
from rlinf.models.embodiment.prefix_ft.history import StateHistoryBuffer
from rlinf.models.embodiment.prefix_ft.protocol import extract_prefix_obs


def _dones_mask(dones: Any) -> torch.Tensor | None:
    if dones is None:
        return None
    if not torch.is_tensor(dones):
        dones = torch.as_tensor(dones)
    return dones.to(dtype=torch.bool).reshape(-1)


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
    history: StateHistoryBuffer | None = None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = extract_prefix_obs(feature_model, final_obs)
        if history is not None:
            # Peek: next_obs should include the post-chunk proprio without
            # double-pushing it before the next decision's curr extract.
            transition_obs = history.fuse(transition_obs, commit=False)
    for key in RLT_OBS_KEYS:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]


def predict_rlt_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
    history: StateHistoryBuffer | None = None,
    dones: Any | None = None,
    update_history: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        if history is not None and update_history:
            done_mask = _dones_mask(dones)
            if done_mask is not None and done_mask.any():
                history.reset(mask=done_mask)

        rlt_obs = extract_prefix_obs(feature_model, env_obs)
        if history is not None:
            rlt_obs = history.fuse(rlt_obs, commit=update_history)
        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=expert_model,
                version=version,
            )
        )
        actions = route_output.actions
        result = route_output.result

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
            history=history if update_history else None,
        )

    return actions, result


predict_prefix_actions = predict_rlt_actions
