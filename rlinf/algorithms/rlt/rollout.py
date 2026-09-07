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

from rlinf.algorithms.rlt.rlt_steam_phase_head import RLT_PHASE_FEATURE_KEY
from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = feature_model.extract_rlt_obs(final_obs)
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
    critical_phase_gate: Any | None = None,
    stage_id: int = 0,
    reset_mask: torch.Tensor | None = None,
    update_gate: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        route_switch_flags = rlt_switch_flags
        route_intervene_requested = intervene_requested
        if critical_phase_gate is not None and update_gate:
            gate_decision = critical_phase_gate.step(
                env_obs,
                mode=mode,
                stage_id=stage_id,
                reset_mask=reset_mask,
                external_actor_switch=rlt_switch_flags,
                actor_routing_enabled=rlt_route.actor_routing_enabled(version),
                expert_routing_enabled=rlt_route.expert_routing_enabled(
                    version,
                    mode=mode,
                    expert_model=expert_model,
                ),
            )
            if bool(getattr(critical_phase_gate, "controls_actor_routing", True)):
                route_switch_flags = gate_decision.actor_switch
            route_intervene_requested = gate_decision.expert_requested
            result["forward_inputs"].update(
                {
                    key: value.detach()
                    for key, value in gate_decision.diagnostics.items()
                }
            )
            if gate_decision.phase_features is not None:
                result["forward_inputs"][RLT_PHASE_FEATURE_KEY] = (
                    gate_decision.phase_features
                )
        elif critical_phase_gate is not None:
            result["forward_inputs"].update(
                critical_phase_gate.empty_diagnostics(actions.shape[0])
            )
            if bool(getattr(critical_phase_gate, "emit_phase_features", False)):
                result["forward_inputs"][RLT_PHASE_FEATURE_KEY] = (
                    critical_phase_gate.empty_phase_features(actions.shape[0])
                )

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=route_switch_flags,
                intervene_requested=route_intervene_requested,
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
        )

    return actions, result
