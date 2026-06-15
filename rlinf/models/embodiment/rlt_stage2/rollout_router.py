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

"""Rollout-side routing helpers for RLT Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .action_router import RLTActionRouteInputs, route_rlt_stage2_actions
from .rollout_schema import require_rlt_stage2_forward_inputs


@dataclass(frozen=True)
class RLTStage2RolloutRouteConfig:
    ready_for_online: bool
    online_gate_step: int
    intervention_enabled: bool
    allow_expert: bool
    chunk_length: int
    action_dim: int


@dataclass(frozen=True)
class RLTStage2RolloutRouteResult:
    actions: torch.Tensor
    result: dict[str, Any]
    expert_label_flag: bool


def route_rlt_stage2_rollout(
    *,
    env_obs: dict[str, Any],
    policy_info: dict[str, torch.Tensor] | None,
    student_model: Any,
    expert_model_getter: Callable[[], Any],
    model_kwargs: dict[str, Any],
    cfg: RLTStage2RolloutRouteConfig,
) -> RLTStage2RolloutRouteResult:
    """Run student/expert inference and build canonical RLT forward inputs."""
    student_actions, result = student_model.predict_action_batch(
        env_obs=env_obs,
        **model_kwargs,
    )
    if "forward_inputs" not in result:
        raise RuntimeError(
            "RLT Stage2 rollout requires result['forward_inputs']; "
            "model.predict_action_batch must expose cached rollout features."
        )
    forward_inputs = result["forward_inputs"]
    if "a_tilde" not in forward_inputs:
        raise RuntimeError(
            "RLT Stage2 rollout requires forward_inputs['a_tilde']; "
            "the rollout policy must expose the base/reference action chunk."
        )
    base_flat = forward_inputs["a_tilde"].detach()
    batch_size = int(student_actions.shape[0])

    expert_takeover = _bool_policy_info(
        policy_info,
        "expert_takeover",
        batch_size=batch_size,
        device=student_actions.device,
        default=False,
    )
    requested_expert_takeover = expert_takeover
    expert_takeover = expert_takeover & cfg.ready_for_online & cfg.allow_expert
    in_critical_phase = _bool_policy_info(
        policy_info,
        "in_critical_phase",
        batch_size=batch_size,
        device=student_actions.device,
        default=True,
    )
    record_transition = _bool_policy_info(
        policy_info,
        "record_transition",
        batch_size=batch_size,
        device=student_actions.device,
        default=True,
    )
    intervention_phase = _float_policy_info(
        policy_info,
        "intervention_phase",
        batch_size=batch_size,
        device=student_actions.device,
        default=0.0,
    )

    expert_actions = None
    expert_label_flag = False
    if cfg.allow_expert and expert_takeover.any():
        expert_model = expert_model_getter()
        if getattr(expert_model, "act_as_vla_reference", False) and hasattr(
            expert_model,
            "predict_vla_reference_action_batch",
        ):
            expert_actions, _ = expert_model.predict_vla_reference_action_batch(
                env_obs=env_obs,
                **model_kwargs,
            )
        else:
            expert_actions, _ = expert_model.predict_action_batch(
                env_obs=env_obs,
                **model_kwargs,
            )
        expert_label_flag = True

    route = route_rlt_stage2_actions(
        RLTActionRouteInputs(
            student_actions=student_actions,
            base_flat=base_flat,
            expert_actions=expert_actions,
            expert_takeover=expert_takeover,
            requested_expert_takeover=requested_expert_takeover,
            intervention_phase=intervention_phase,
            in_critical_phase=in_critical_phase,
            record_transition=record_transition,
            ready_for_online=cfg.ready_for_online,
            online_gate_step=cfg.online_gate_step,
            chunk_length=cfg.chunk_length,
            action_dim=cfg.action_dim,
        )
    )
    actions = route.actions
    forward_inputs.update(route.to_forward_input_updates())
    if policy_info is not None and "deviation" in policy_info:
        forward_inputs["deviation"] = policy_info["deviation"].to(
            actions.device,
            dtype=torch.bool,
        )
    if policy_info is not None and "takeover_left" in policy_info:
        forward_inputs["takeover_left"] = policy_info["takeover_left"].to(
            actions.device,
            dtype=torch.float32,
        )
    forward_inputs["intervention_enabled"] = torch.full(
        (actions.shape[0], 1),
        cfg.intervention_enabled,
        dtype=torch.bool,
        device=actions.device,
    )
    require_rlt_stage2_forward_inputs(
        forward_inputs,
        batch_size=actions.shape[0],
        chunk_length=cfg.chunk_length,
        action_dim=cfg.action_dim,
        context="predict",
    )
    return RLTStage2RolloutRouteResult(
        actions=actions,
        result=result,
        expert_label_flag=expert_label_flag,
    )


def _bool_policy_info(
    policy_info: dict[str, torch.Tensor] | None,
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if policy_info is None or key not in policy_info:
        return torch.full(
            (batch_size,),
            bool(default),
            dtype=torch.bool,
            device=device,
        )
    value = torch.as_tensor(policy_info[key], device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(value.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return value.reshape(batch_size, -1).to(torch.bool).any(dim=1)


def _float_policy_info(
    policy_info: dict[str, torch.Tensor] | None,
    key: str,
    *,
    batch_size: int,
    device: torch.device,
    default: float,
) -> torch.Tensor:
    if policy_info is None or key not in policy_info:
        return torch.full(
            (batch_size,),
            float(default),
            dtype=torch.float32,
            device=device,
        )
    value = torch.as_tensor(policy_info[key], device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            float(value.reshape(-1)[0].item()),
            dtype=torch.float32,
            device=device,
        )
    return value.reshape(batch_size, -1)[:, -1].to(torch.float32)
