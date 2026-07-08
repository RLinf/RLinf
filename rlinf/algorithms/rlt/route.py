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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch

from rlinf.algorithms.rlt.expert import predict_expert_actions
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay


@dataclass(kw_only=True)
class RLTRouteContext:
    env_obs: dict[str, Any]
    rlt_obs: dict[str, torch.Tensor]
    student_actions: torch.Tensor
    result: dict[str, Any]
    mode: Literal["train", "eval"]
    rlt_switch_flags: torch.Tensor | None = None
    intervene_requested: torch.Tensor | None = None
    expert_model: Any | None = None
    version: int = 0
    default_actor_switch: bool = False


@dataclass(kw_only=True)
class RLTRouteOutput:
    actions: torch.Tensor
    result: dict[str, Any]


def _info_bool(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), bool(default), dtype=torch.bool, device=device)
    value = torch.as_tensor(value, device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(value.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return value.reshape(batch_size, -1).to(torch.bool).any(dim=1)


def _last_info_bool(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), bool(default), dtype=torch.bool, device=device)
    value = torch.as_tensor(value, device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(value.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return value.reshape(batch_size, -1).to(torch.bool)[:, -1]


def _flatten_action_chunk(actions: torch.Tensor) -> torch.Tensor:
    if actions.dim() <= 2:
        return actions
    return actions.reshape(actions.shape[0], -1)


def _normalize_rlt_switch_flags(
    actions: torch.Tensor,
    rlt_switch_flags: torch.Tensor | None,
    *,
    default: bool,
) -> torch.Tensor:
    if rlt_switch_flags is None:
        rlt_switch_flags = torch.full(
            (actions.shape[0], actions.shape[1]),
            bool(default),
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


def _base_ref_actions(
    ref_chunk: torch.Tensor,
    *,
    chunk_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if ref_chunk.dim() == 2:
        ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
    return ref_chunk[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)


class RLTRoute(ABC):
    @abstractmethod
    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        """Route student actions among actor, reference, and optional expert."""


class RealworldRLTRoute(RLTRoute):
    """Actor/ref routing for realworld RLT (keyboard or env switch flags)."""

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        actions = ctx.student_actions
        result = ctx.result
        rlt_switch_flags = _normalize_rlt_switch_flags(
            actions,
            ctx.rlt_switch_flags,
            default=ctx.default_actor_switch,
        )
        ref_actions = result["forward_inputs"]["ref_chunk"].to(
            device=actions.device, dtype=actions.dtype
        )
        routed_actions = torch.where(
            rlt_switch_flags,
            actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
        ).contiguous()
        result["forward_inputs"]["action"] = routed_actions.reshape(
            routed_actions.shape[0], -1
        ).contiguous()
        result["forward_inputs"]["record_transition"] = rlt_switch_flags.reshape(
            actions.shape[0], -1
        )[:, :1].to(torch.bool)
        result["forward_inputs"]["actor_switch"] = result["forward_inputs"][
            "record_transition"
        ]
        return RLTRouteOutput(actions=routed_actions, result=result)


class SimulatorRLTRoute(RLTRoute):
    """Actor/ref/expert routing for ManiSkill RLT with schedule warmup."""

    def __init__(self, *, use_schedule: bool, warmup_updates: int):
        self.use_schedule = use_schedule
        self.warmup_updates = warmup_updates

    def _ready_for_online(self, version: int) -> bool:
        return not self.use_schedule or int(version) >= self.warmup_updates

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        actions = ctx.student_actions
        result = ctx.result
        batch_size, chunk_len, action_dim = actions.shape
        ready_for_online = self._ready_for_online(ctx.version)

        critical_phase = _last_info_bool(
            ctx.rlt_switch_flags,
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        actor_switch = critical_phase
        if self.use_schedule:
            actor_switch = actor_switch & torch.full(
                (batch_size,),
                ready_for_online,
                dtype=torch.bool,
                device=actions.device,
            )

        requested_expert_takeover = _last_info_bool(
            ctx.intervene_requested,
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        expert_takeover = (
            requested_expert_takeover & ready_for_online & (ctx.mode == "train")
        )

        base_actions = _base_ref_actions(
            ctx.rlt_obs["ref_chunk"],
            chunk_len=chunk_len,
            action_dim=action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        routed_actions = torch.where(
            actor_switch[:, None, None],
            actions,
            base_actions,
        ).contiguous()

        intervene_flags = torch.zeros(
            (batch_size, chunk_len),
            dtype=torch.bool,
            device=actions.device,
        )
        if expert_takeover.any():
            if ctx.expert_model is None:
                raise RuntimeError(
                    "ManiSkill RLT expert takeover was requested, but expert_model "
                    "is not configured."
                )
            expert_actions = predict_expert_actions(
                ctx.expert_model,
                ctx.env_obs,
                chunk_len=chunk_len,
                action_dim=action_dim,
                device=actions.device,
                dtype=actions.dtype,
            )
            routed_actions = torch.where(
                expert_takeover[:, None, None],
                expert_actions,
                routed_actions,
            ).contiguous()
            intervene_flags[expert_takeover] = True

        forward_inputs = result["forward_inputs"]
        forward_inputs["action"] = _flatten_action_chunk(routed_actions).detach()
        forward_inputs["record_transition"] = critical_phase[:, None]
        forward_inputs["actor_switch"] = (actor_switch & ~expert_takeover)[:, None]
        forward_inputs["intervention_requested"] = requested_expert_takeover[:, None]
        result["intervene_flags"] = intervene_flags
        return RLTRouteOutput(actions=routed_actions, result=result)


def build_rlt_route(cfg: Any) -> RLTRoute:
    if use_simulator_transition_replay(cfg):
        schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        return SimulatorRLTRoute(
            use_schedule=bool(schedule_cfg.get("enable", False)),
            warmup_updates=int(schedule_cfg.get("warmup_post_collect_updates", 0)),
        )
    return RealworldRLTRoute()
