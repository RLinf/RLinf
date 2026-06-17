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

"""Rollout adapter for RLT Stage 2 TD3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np
import torch

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import EmbodiedRolloutResult

from .schedule import resolve_warmup_required_updates


class TransitionSource(IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


def resolve_collection_phase_id(phase: str | int | None) -> int:
    if phase is None:
        return COLLECTION_PHASE_UNKNOWN
    if isinstance(phase, int):
        return int(phase)
    phase_name = str(phase).split(":", 1)[0].lower()
    if phase_name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if phase_name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def resolve_chunk_source(source_chunk: np.ndarray) -> int:
    values = {int(value) for value in np.asarray(source_chunk).reshape(-1)}
    if not values:
        return int(TransitionSource.RL)
    if int(TransitionSource.MIXED) in values or len(values) > 1:
        return int(TransitionSource.MIXED)
    return next(iter(values))


def human_source_mask(source_chunk) -> np.ndarray:
    source_array = np.asarray(source_chunk)
    return np.logical_or(
        source_array == int(TransitionSource.HUMAN),
        source_array == int(TransitionSource.MIXED),
    )


REQUIRED_RLT_STAGE2_FORWARD_INPUTS = (
    "x",
    "a_tilde",
    "base_a_tilde",
    "ref_chunk",
    "action",
    "action_chunk",
    "student_control",
    "intervention_flags",
    "source_chunk",
    "source",
    "collection_phase_id",
    "intervention_requested",
    "intervention_phase",
    "in_critical_phase",
    "record_transition",
    "ready_for_online",
    "online_gate_step",
)


def resolve_source_from_source_chunk(source_chunk: torch.Tensor) -> torch.Tensor:
    """Collapse per-step source labels to one chunk-level source label."""
    if not isinstance(source_chunk, torch.Tensor) or source_chunk.ndim != 2:
        raise ValueError(
            "RLT Stage2 source_chunk must have shape [B, T], got "
            f"{_shape_str(source_chunk)}."
        )
    return torch.where(
        source_chunk.eq(source_chunk[:, :1]).all(dim=1, keepdim=True),
        source_chunk[:, :1],
        torch.full(
            (source_chunk.shape[0], 1),
            int(TransitionSource.MIXED),
            dtype=torch.uint8,
            device=source_chunk.device,
        ),
    )


def require_rlt_stage2_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    batch_size: int,
    chunk_length: int,
    action_dim: int,
    context: str,
) -> dict[str, Any]:
    """Validate that rollout emitted the canonical RLT Stage 2 fields."""
    missing = [
        key for key in REQUIRED_RLT_STAGE2_FORWARD_INPUTS if key not in forward_inputs
    ]
    if missing:
        raise RuntimeError(
            f"RLT Stage2 {context} forward_inputs missing required keys: {missing}. "
            "Build them in the rollout RLT path instead of relying on silent "
            "fallback defaults."
        )

    action_chunk_dim = int(chunk_length) * int(action_dim)
    expected_shapes = {
        "x": (batch_size, None),
        "a_tilde": (batch_size, action_chunk_dim),
        "base_a_tilde": (batch_size, action_chunk_dim),
        "ref_chunk": (batch_size, action_chunk_dim),
        "action": (batch_size, action_chunk_dim),
        "action_chunk": (batch_size, action_chunk_dim),
        "student_control": (batch_size, 1),
        "intervention_flags": (batch_size, chunk_length),
        "source_chunk": (batch_size, chunk_length),
        "source": (batch_size, 1),
        "collection_phase_id": (batch_size, 1),
        "intervention_requested": (batch_size, 1),
        "intervention_phase": (batch_size, 1),
        "in_critical_phase": (batch_size, 1),
        "record_transition": (batch_size, 1),
        "ready_for_online": (batch_size, 1),
        "online_gate_step": (batch_size, 1),
    }
    for key, expected_shape in expected_shapes.items():
        _require_tensor_shape(
            forward_inputs[key],
            expected_shape,
            field_name=key,
            context=context,
        )
    return forward_inputs


def _require_tensor_shape(
    value: Any,
    expected_shape: tuple[int | None, ...],
    *,
    field_name: str,
    context: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"RLT Stage2 {context} forward_inputs[{field_name!r}] must be a "
            f"torch.Tensor, got {type(value).__name__}."
        )
    if value.ndim != len(expected_shape):
        raise ValueError(
            f"RLT Stage2 {context} forward_inputs[{field_name!r}] must have "
            f"{len(expected_shape)} dims, got shape {_shape_str(value)}."
        )
    for dim, expected in enumerate(expected_shape):
        if expected is not None and int(value.shape[dim]) != int(expected):
            raise ValueError(
                f"RLT Stage2 {context} forward_inputs[{field_name!r}] shape "
                f"mismatch: expected {expected_shape}, got {_shape_str(value)}."
            )


def _shape_str(value: Any) -> str:
    return "None" if value is None else str(tuple(getattr(value, "shape", ())))


@dataclass(frozen=True)
class RLTActionRouteInputs:
    student_actions: torch.Tensor
    base_flat: torch.Tensor
    expert_actions: torch.Tensor | None
    expert_takeover: torch.Tensor
    requested_expert_takeover: torch.Tensor
    intervention_phase: torch.Tensor
    in_critical_phase: torch.Tensor
    record_transition: torch.Tensor
    ready_for_online: bool
    online_gate_step: int
    chunk_length: int
    action_dim: int


@dataclass(frozen=True)
class RLTActionRouteResult:
    actions: torch.Tensor
    action_flat: torch.Tensor
    base_flat: torch.Tensor
    student_control: torch.Tensor
    intervention_flags: torch.Tensor
    source_chunk: torch.Tensor
    source: torch.Tensor
    collection_phase_id: torch.Tensor
    intervention_requested: torch.Tensor
    intervention_phase: torch.Tensor
    in_critical_phase: torch.Tensor
    record_transition: torch.Tensor
    ready_for_online: torch.Tensor
    online_gate_step: torch.Tensor

    def to_forward_input_updates(self) -> dict[str, torch.Tensor]:
        return {
            "base_a_tilde": self.base_flat,
            "ref_chunk": self.base_flat.detach(),
            "action": self.action_flat.detach(),
            "action_chunk": self.action_flat.detach(),
            "student_control": self.student_control[:, None],
            "intervention_flags": self.intervention_flags,
            "source_chunk": self.source_chunk,
            "source": self.source,
            "collection_phase_id": self.collection_phase_id,
            "intervention_requested": self.intervention_requested[:, None],
            "intervention_phase": self.intervention_phase[:, None],
            "in_critical_phase": self.in_critical_phase[:, None],
            "record_transition": self.record_transition[:, None],
            "ready_for_online": self.ready_for_online,
            "online_gate_step": self.online_gate_step,
        }


def route_rlt_stage2_actions(inputs: RLTActionRouteInputs) -> RLTActionRouteResult:
    """Route execution between base VLA, RLT actor, and expert correction."""
    _validate_route_inputs(inputs)

    base_actions = inputs.base_flat.reshape(
        inputs.base_flat.shape[0],
        inputs.chunk_length,
        inputs.action_dim,
    )
    actor_control = (
        torch.full(
            (inputs.student_actions.shape[0],),
            bool(inputs.ready_for_online),
            dtype=torch.bool,
            device=inputs.student_actions.device,
        )
        & inputs.in_critical_phase
    )
    actions = torch.where(
        actor_control[:, None, None],
        inputs.student_actions,
        base_actions,
    )

    intervention_flags = torch.zeros(
        (actions.shape[0], inputs.chunk_length),
        dtype=torch.bool,
        device=actions.device,
    )
    source_chunk = torch.full(
        (actions.shape[0], inputs.chunk_length),
        int(TransitionSource.BASE),
        dtype=torch.uint8,
        device=actions.device,
    )
    source_chunk[actor_control] = int(TransitionSource.RL)

    if inputs.expert_actions is not None:
        expert_mask = inputs.expert_takeover[:, None, None].to(actions.device)
        actions = torch.where(expert_mask, inputs.expert_actions, actions)
        intervention_flags[inputs.expert_takeover] = True
        source_chunk[inputs.expert_takeover] = int(TransitionSource.HUMAN)

    action_flat = actions.reshape(actions.shape[0], -1)
    return RLTActionRouteResult(
        actions=actions,
        action_flat=action_flat,
        base_flat=inputs.base_flat,
        student_control=actor_control,
        intervention_flags=intervention_flags,
        source_chunk=source_chunk,
        source=resolve_source_from_source_chunk(source_chunk),
        collection_phase_id=torch.full(
            (actions.shape[0], 1),
            COLLECTION_PHASE_ONLINE
            if inputs.ready_for_online
            else COLLECTION_PHASE_WARMUP,
            dtype=torch.uint8,
            device=actions.device,
        ),
        intervention_requested=inputs.requested_expert_takeover.to(
            actions.device,
            dtype=torch.bool,
        ),
        intervention_phase=inputs.intervention_phase.to(
            actions.device,
            dtype=torch.float32,
        ),
        in_critical_phase=inputs.in_critical_phase.to(actions.device, dtype=torch.bool),
        record_transition=inputs.record_transition.to(actions.device, dtype=torch.bool),
        ready_for_online=torch.full(
            (actions.shape[0], 1),
            bool(inputs.ready_for_online),
            dtype=torch.bool,
            device=actions.device,
        ),
        online_gate_step=torch.full(
            (actions.shape[0], 1),
            float(inputs.online_gate_step),
            dtype=torch.float32,
            device=actions.device,
        ),
    )


def _validate_route_inputs(inputs: RLTActionRouteInputs) -> None:
    if inputs.student_actions.ndim != 3:
        raise ValueError(
            "RLTActionRouter student_actions must have shape [B, T, A], got "
            f"{_shape_str(inputs.student_actions)}."
        )
    batch_size = int(inputs.student_actions.shape[0])
    expected_action_shape = (batch_size, inputs.chunk_length, inputs.action_dim)
    if tuple(inputs.student_actions.shape) != expected_action_shape:
        raise ValueError(
            "RLTActionRouter student_actions shape mismatch: expected "
            f"{expected_action_shape}, got {_shape_str(inputs.student_actions)}."
        )
    expected_flat_shape = (batch_size, inputs.chunk_length * inputs.action_dim)
    if tuple(inputs.base_flat.shape) != expected_flat_shape:
        raise ValueError(
            "RLTActionRouter base_flat shape mismatch: expected "
            f"{expected_flat_shape}, got {_shape_str(inputs.base_flat)}."
        )
    if (
        inputs.expert_actions is not None
        and tuple(inputs.expert_actions.shape) != expected_action_shape
    ):
        raise ValueError(
            "RLTActionRouter expert_actions shape mismatch: expected "
            f"{expected_action_shape}, got {_shape_str(inputs.expert_actions)}."
        )
    for name in (
        "expert_takeover",
        "requested_expert_takeover",
        "intervention_phase",
        "in_critical_phase",
        "record_transition",
    ):
        value = getattr(inputs, name)
        if tuple(value.shape) != (batch_size,):
            raise ValueError(
                f"RLTActionRouter {name} must have shape [{batch_size}], got "
                f"{_shape_str(value)}."
            )


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


def update_last_rlt_action_metadata(
    rollout_result: EmbodiedRolloutResult,
    intervene_flags: torch.Tensor,
) -> None:
    """Update RLT source metadata after an env-side action override."""
    if not rollout_result.forward_inputs:
        return

    last_forward_inputs = rollout_result.forward_inputs[-1]
    if "source_chunk" not in last_forward_inputs:
        return

    if intervene_flags.dim() == 1:
        intervene_flags = intervene_flags[:, None]
    intervene_flags = intervene_flags.to(torch.bool)

    batch_size, num_action_chunks = intervene_flags.shape[:2]
    source_chunk = last_forward_inputs["source_chunk"].clone()
    source_chunk = source_chunk.reshape(batch_size, num_action_chunks)
    source_chunk[intervene_flags.to(source_chunk.device)] = int(TransitionSource.HUMAN)
    last_forward_inputs["source_chunk"] = source_chunk.cpu().contiguous()
    last_forward_inputs["source"] = (
        resolve_source_from_source_chunk(source_chunk).cpu().contiguous()
    )
    last_forward_inputs["intervention_flag"] = (
        intervene_flags.any(dim=1, keepdim=True).cpu().contiguous()
    )
    if "intervention_flags" in last_forward_inputs:
        last_forward_inputs["intervention_flags"] = (
            intervene_flags.cpu().contiguous()
        )
    last_forward_inputs.pop("model_action", None)


class RLTStage2RolloutAdapter:
    """Encapsulates RLT Stage 2 rollout-only behavior."""

    enabled = True

    def __init__(
        self,
        *,
        cfg,
        student_model: Any,
        expert_model_getter,
        has_expert_model_config: bool,
    ) -> None:
        if not self.is_enabled_cfg(cfg):
            raise ValueError("RLTStage2RolloutAdapter requires RLT Stage2 TD3 config.")
        self.cfg = cfg
        self.student_model = student_model
        self.expert_model_getter = expert_model_getter
        self.has_expert_model_config = bool(has_expert_model_config)
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        self.intervention_mode = str(
            intervention_cfg.get("mode", "local_correction")
        )
        if (
            bool(intervention_cfg.get("enable", False))
            and self.intervention_mode != "local_correction"
        ):
            raise ValueError(
                "RLT Stage2 rollout received unsupported "
                "algorithm.intervention.mode, got "
                f"{self.intervention_mode!r}."
            )
        self.intervention_enabled = bool(intervention_cfg.get("enable", False)) and (
            self.intervention_mode == "local_correction"
        )

    @staticmethod
    def is_enabled_cfg(cfg) -> bool:
        return (
            cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and SupportedModel(cfg.actor.model.model_type) == SupportedModel.RLT_STAGE2
        )

    def expert_model_path(self, configured_path: str, expert_ckpt_path: str | None):
        if expert_ckpt_path and os.path.isdir(str(expert_ckpt_path)):
            return expert_ckpt_path
        return configured_path

    def configure_expert_model(self, expert_model_config) -> None:
        if expert_model_config.get("rlt_stage2", None) is None:
            raise RuntimeError(
                "RLT Stage2 expert model config must include actor.model.rlt_stage2."
            )
        rollout_expert_cfg = self.cfg.rollout.expert_model
        act_as_vla_reference = rollout_expert_cfg.get("act_as_vla_reference", True)
        expert_model_config.rlt_stage2.act_as_vla_reference = act_as_vla_reference
        if act_as_vla_reference:
            expert_model_config.rlt_stage2.load_feature_backbones = True
            expert_model_config.rlt_stage2.load_rl_token_model = False

    def ready_for_online(self, update_version: int) -> tuple[bool, int]:
        warmup_required_updates = resolve_warmup_required_updates(self.cfg)
        return update_version >= warmup_required_updates, warmup_required_updates

    def predict(
        self,
        *,
        env_obs: dict[str, Any],
        policy_info: dict[str, torch.Tensor] | None,
        model_kwargs: dict[str, Any],
        mode: str,
        allow_expert: bool,
        update_version: int,
    ):
        ready_for_online, online_gate_step = self.ready_for_online(update_version)
        return route_rlt_stage2_rollout(
            env_obs=env_obs,
            policy_info=policy_info,
            student_model=self.student_model,
            expert_model_getter=self.expert_model_getter,
            model_kwargs=model_kwargs,
            cfg=RLTStage2RolloutRouteConfig(
                ready_for_online=ready_for_online,
                online_gate_step=online_gate_step,
                intervention_enabled=self.intervention_enabled,
                allow_expert=(
                    mode == "train" and allow_expert and self.has_expert_model_config
                ),
                chunk_length=self.cfg.actor.model.num_action_chunks,
                action_dim=self.cfg.actor.model.action_dim,
            ),
        )

    def rollout_state_dict(self):
        return self.student_model.rollout_state_dict()

    def use_dagger_beta(self) -> bool:
        return False

    def allow_bootstrap_values(self) -> bool:
        return False

    def final_forward_inputs(self, result: dict[str, Any]) -> dict[str, Any]:
        return result["forward_inputs"]

    def encode_step_trace(
        self,
        step_obs: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if step_obs is None:
            return {}
        if not hasattr(self.student_model, "encode_obs"):
            raise RuntimeError(
                "RLT Stage2 stride replay requires hf_model.encode_obs for step features."
            )
        first_tensor = next(
            (
                value
                for key, value in step_obs.items()
                if not key.startswith("_rlt_")
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        explicit_offsets = step_obs.get("_rlt_step_offsets", None)
        if explicit_offsets is not None:
            if (
                not isinstance(explicit_offsets, torch.Tensor)
                or explicit_offsets.dim() != 2
            ):
                raise ValueError(
                    "RLT step_obs['_rlt_step_offsets'] must have shape [A, B], "
                    f"got {type(explicit_offsets)=}."
                )
            anchor_offset_tensor = explicit_offsets.to(torch.long).contiguous()
        else:
            if first_tensor is None:
                raise ValueError("RLT step_obs must contain at least one tensor field.")
            step_count = int(first_tensor.shape[0])
            batch_size = int(first_tensor.shape[1])
            anchor_offsets = self._sparse_anchor_offsets(step_count)
            anchor_offset_tensor = (
                torch.tensor(anchor_offsets, dtype=torch.long)[:, None]
                .expand(len(anchor_offsets), batch_size)
                .contiguous()
            )
            if anchor_offsets:
                step_obs = self._slice_step_obs_offsets(step_obs, anchor_offsets)

        if anchor_offset_tensor.shape[0] == 0:
            return {"anchor_offsets": anchor_offset_tensor}

        if first_tensor is None:
            raise ValueError("RLT step_obs must contain obs tensors for sparse anchors.")
        flat_obs, step_count, batch_size = self._flatten_step_obs(step_obs)
        total = step_count * batch_size
        micro_batch_size = int(
            self.cfg.actor.model.rlt_stage2.get("replay_feature_batch_size", 32)
        )
        if micro_batch_size <= 0:
            micro_batch_size = total
        encoded_x = []
        encoded_a_tilde = []
        with torch.no_grad():
            for begin in range(0, total, micro_batch_size):
                end = min(begin + micro_batch_size, total)
                obs_chunk = self._slice_flat_obs(flat_obs, begin, end)
                x, a_tilde = self.student_model.encode_obs(obs_chunk)
                encoded_x.append(x.detach().cpu())
                encoded_a_tilde.append(a_tilde.detach().cpu())
        x_all = torch.cat(encoded_x, dim=0).reshape(step_count, batch_size, -1)
        a_tilde_all = torch.cat(encoded_a_tilde, dim=0).reshape(
            step_count,
            batch_size,
            -1,
        )
        return {
            "anchor_offsets": anchor_offset_tensor,
            "x": x_all.contiguous(),
            "a_tilde": a_tilde_all.contiguous(),
        }

    def _sparse_anchor_offsets(self, step_count: int) -> list[int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        stride = int(self.cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
        if stride <= 0 or chunk_len <= 0:
            return []

        offsets = set()
        offset = 0
        while True:
            offset = (offset + stride) % chunk_len
            if offset == 0 or offset in offsets:
                break
            if offset < step_count:
                offsets.add(offset)
        return sorted(offsets)

    @staticmethod
    def _flatten_step_obs(
        step_obs: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int]:
        first_tensor = next(
            (
                value
                for key, value in step_obs.items()
                if not key.startswith("_rlt_")
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        if first_tensor is None:
            raise ValueError("RLT step_obs must contain at least one tensor field.")
        step_count = int(first_tensor.shape[0])
        batch_size = int(first_tensor.shape[1])
        flat_obs: dict[str, Any] = {}
        for key, value in step_obs.items():
            if key.startswith("_rlt_"):
                continue
            if isinstance(value, torch.Tensor):
                flat_obs[key] = value.reshape(step_count * batch_size, *value.shape[2:])
            elif isinstance(value, list):
                flat_obs[key] = [item for step_values in value for item in step_values]
            elif value is None:
                flat_obs[key] = None
            else:
                flat_obs[key] = value
        return flat_obs, step_count, batch_size

    @staticmethod
    def _slice_step_obs_offsets(
        step_obs: dict[str, Any],
        offsets: list[int],
    ) -> dict[str, Any]:
        sliced_obs: dict[str, Any] = {}
        index_tensor = torch.as_tensor(offsets, dtype=torch.long)
        for key, value in step_obs.items():
            if key.startswith("_rlt_"):
                continue
            if isinstance(value, torch.Tensor):
                sliced_obs[key] = value.index_select(
                    0,
                    index_tensor.to(device=value.device),
                )
            elif isinstance(value, list):
                sliced_obs[key] = [value[offset] for offset in offsets]
            elif value is None:
                sliced_obs[key] = None
            else:
                sliced_obs[key] = value
        return sliced_obs

    @staticmethod
    def _slice_flat_obs(
        flat_obs: dict[str, Any],
        begin: int,
        end: int,
    ) -> dict[str, Any]:
        obs_chunk: dict[str, Any] = {}
        for key, value in flat_obs.items():
            if isinstance(value, torch.Tensor):
                obs_chunk[key] = value[begin:end]
            elif isinstance(value, list):
                obs_chunk[key] = value[begin:end]
            else:
                obs_chunk[key] = value
        return obs_chunk
