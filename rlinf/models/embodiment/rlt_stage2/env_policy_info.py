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

"""Env-side policy_info adapter for RLT Stage 2 collection."""

from __future__ import annotations

import os
from typing import Any, Literal

import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult, RolloutResult

from .schedule import (
    metric_mean,
    phase_id,
    resolve_rollout_phase,
    utc_timestamp,
    write_status_json,
)
from .rollout import update_last_rlt_action_metadata


def is_rlt_stage2_td3_cfg(cfg) -> bool:
    return (
        cfg.algorithm.get("loss_type", None) == "rlt_td3"
        and cfg.actor.model.get("model_type", None) == "rlt_stage2"
    )


def build_policy_info_adapter(
    *,
    cfg,
    train_batch_size: int | None,
    eval_batch_size: int | None,
):
    """Build the RLT Stage 2 env adapter when the cfg enables it."""
    if not is_rlt_stage2_td3_cfg(cfg):
        return None
    adapter = RLTStage2PolicyInfoAdapter(
        cfg=cfg,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    if not (adapter.enabled("train") or adapter.enabled("eval")):
        return None
    return adapter


def build_step_obs(
    cfg,
    start_obs: dict[str, Any] | None,
    obs_list,
) -> dict[str, Any] | None:
    if start_obs is None or not isinstance(obs_list, (list, tuple)) or not obs_list:
        return None

    stride = int(cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
    if stride <= 0:
        return None

    step_obs_list = [start_obs, *obs_list]
    offsets = _sparse_step_obs_offsets(cfg, len(step_obs_list))
    step_obs: dict[str, Any] = {}
    batch_size = _infer_obs_batch_size(step_obs_list[0])
    for key in step_obs_list[0].keys():
        if not offsets:
            continue
        values = [step_obs_list[offset].get(key, None) for offset in offsets]
        first_non_none = next((value for value in values if value is not None), None)
        if first_non_none is None:
            step_obs[key] = None
        elif isinstance(first_non_none, torch.Tensor):
            if any(value is None for value in values):
                raise ValueError(
                    f"Inconsistent RLT step_obs key {key!r}: "
                    "tensor values contain None."
                )
            values = [
                value.to(first_non_none.device)
                if value.device != first_non_none.device
                else value
                for value in values
            ]
            step_obs[key] = torch.stack(values, dim=0)
        elif isinstance(first_non_none, list):
            step_obs[key] = values
        else:
            step_obs[key] = values
    step_obs["_rlt_step_offsets"] = torch.tensor(
        offsets,
        dtype=torch.long,
    )[:, None].expand(len(offsets), batch_size).contiguous()
    return step_obs


def emit_rollout_status(
    *,
    cfg,
    env_metrics: dict[str, torch.Tensor],
    rank: int,
    last_logged_phase: str | None,
    log_info,
) -> str | None:
    if not is_rlt_stage2_td3_cfg(cfg):
        return last_logged_phase
    ready_value = metric_mean(env_metrics, "rlt_ready_for_online")
    if ready_value is None:
        return last_logged_phase

    ready_for_online = bool(ready_value >= 0.5)
    in_critical_phase_rate = metric_mean(
        env_metrics,
        "rlt_in_critical_phase",
        default=0.0,
    )
    record_transition_rate = metric_mean(
        env_metrics,
        "rlt_record_transition",
        default=0.0,
    )
    student_control_rate = metric_mean(
        env_metrics,
        "student_control_rate",
        default=0.0,
    )
    phase = resolve_rollout_phase(
        ready_for_online=ready_for_online,
        student_control_rate=float(student_control_rate),
    )
    phase_numeric_id = phase_id(phase)

    status_like = env_metrics["rlt_ready_for_online"].detach().float().reshape(-1)
    env_metrics["rlt_status_phase_id"] = torch.full_like(
        status_like,
        float(phase_numeric_id),
    )
    env_metrics["rlt_status_ready_for_online"] = torch.full_like(
        status_like,
        float(ready_for_online),
    )
    env_metrics["rlt_status_in_critical_phase_rate"] = torch.full_like(
        status_like,
        float(in_critical_phase_rate),
    )
    env_metrics["rlt_status_record_transition_rate"] = torch.full_like(
        status_like,
        float(record_transition_rate),
    )
    env_metrics["rlt_status_student_control_rate"] = torch.full_like(
        status_like,
        float(student_control_rate),
    )

    if rank == 0 and phase != last_logged_phase:
        log_info(
            "[RLT_STATUS][env] "
            f"phase={phase} ready={int(ready_for_online)} "
            f"critical={float(in_critical_phase_rate):.2f} "
            f"record={float(record_transition_rate):.2f} "
            f"student={float(student_control_rate):.2f}"
        )
        last_logged_phase = phase

    status_dir = os.path.join(cfg.runner.logger.log_path, "status")
    write_status_json(
        os.path.join(status_dir, f"rlt_env_status_rank{rank}.json"),
        {
            "timestamp": utc_timestamp(),
            "component": "env",
            "rank": rank,
            "phase": phase,
            "phase_id": phase_numeric_id,
            "ready_for_online": ready_for_online,
            "in_critical_phase_rate": float(in_critical_phase_rate),
            "record_transition_rate": float(record_transition_rate),
            "student_control_rate": float(student_control_rate),
        },
    )
    return last_logged_phase


def _sparse_step_obs_offsets(cfg, step_count: int) -> list[int]:
    stride = int(cfg.actor.model.rlt_stage2.get("replay_subsample_stride", 0))
    chunk_len = int(cfg.actor.model.num_action_chunks)
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


def _infer_obs_batch_size(obs: dict[str, Any]) -> int:
    for value in obs.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
    raise ValueError("Cannot infer RLT step_obs batch size from observation.")


class RLTStage2PolicyInfoAdapter:
    """Owns RLT Stage 2 env-side state exposed to rollout workers."""

    def __init__(
        self,
        *,
        cfg,
        train_batch_size: int | None,
        eval_batch_size: int | None,
    ) -> None:
        self.cfg = cfg
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_states: list[dict[str, torch.Tensor]] = []
        self.eval_states: list[dict[str, torch.Tensor]] = []
        self.train_maniskill_controllers: list[Any | None] = []
        self.eval_maniskill_controllers: list[Any | None] = []

    def init_stage(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if not self.enabled(mode):
            return None

        controller = self._init_maniskill_controller(
            stage_id=stage_id,
            mode=mode,
            batch_size=self._batch_size(mode),
            env=env,
        )
        states = self._states(mode)
        self._ensure_len(states, stage_id, {})
        states[stage_id] = controller.state
        return controller.export_policy_info(controller.state)

    def update_stage(
        self,
        *,
        infos: dict[str, Any] | None,
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        states = self._states(mode)
        if not self.enabled(mode) or infos is None or stage_id >= len(states):
            return None

        controllers = self._maniskill_controllers(mode)
        if stage_id >= len(controllers) or controllers[stage_id] is None:
            controller = self._init_maniskill_controller(
                stage_id=stage_id,
                mode=mode,
                batch_size=int(chunk_dones.shape[0]),
                env=env,
            )
        else:
            controller = controllers[stage_id]
        assert controller is not None
        policy_info = controller.update(
            infos=infos,
            chunk_dones=chunk_dones,
            intervention_enabled=self.local_correction_enabled(),
        )
        self._ensure_len(states, stage_id, {})
        states[stage_id] = controller.state
        return policy_info

    def update_last_action_metadata(
        self,
        *,
        rollout_result: Any,
        intervene_flags: torch.Tensor,
    ) -> None:
        if not self.local_correction_enabled():
            return

        update_last_rlt_action_metadata(rollout_result, intervene_flags)

    def build_step_obs(
        self,
        *,
        start_obs: dict[str, Any] | None,
        obs_list,
    ) -> dict[str, Any] | None:
        return build_step_obs(self.cfg, start_obs, obs_list)

    def append_step_trace(
        self,
        *,
        rollout_accumulator: EmbodiedRolloutResult,
        rollout_result: RolloutResult,
    ) -> None:
        if rollout_result.step_trace:
            rollout_accumulator.append_step_trace(rollout_result.step_trace)

    def final_forward_inputs(self, rollout_result: RolloutResult) -> dict[str, Any]:
        return rollout_result.forward_inputs

    def collect_rollout_metrics(
        self,
        *,
        env_metrics: dict[str, list],
        rollout_result: RolloutResult,
    ) -> None:
        forward_inputs = rollout_result.forward_inputs
        intervention_flags = forward_inputs.get("intervention_flags", None)
        if intervention_flags is not None:
            actual_intervention = intervention_flags.detach().float().reshape(-1).cpu()
            env_metrics["expert_intervention_actual_rate"].append(
                actual_intervention
            )
            env_metrics["expert_takeover_rate"].append(actual_intervention)
            intervention_phase = forward_inputs.get("intervention_phase", None)
            if intervention_phase is not None:
                phase = intervention_phase.detach().float().reshape(-1).cpu()
                if phase.numel() == actual_intervention.numel():
                    env_metrics["insert_intervention_rate"].append(
                        actual_intervention * (phase == 2).float()
                    )

        metric_names = {
            "intervention_requested": "expert_intervention_requested_rate",
            "ready_for_online": "rlt_ready_for_online",
            "in_critical_phase": "rlt_in_critical_phase",
            "record_transition": "rlt_record_transition",
            "student_control": "student_control_rate",
        }
        for source_key, metric_key in metric_names.items():
            value = forward_inputs.get(source_key, None)
            if value is not None:
                env_metrics[metric_key].append(
                    value.detach().float().reshape(-1).cpu()
                )

    def emit_status(
        self,
        *,
        env_metrics: dict[str, torch.Tensor],
        rank: int,
        last_logged_phase: str | None,
        log_info,
    ) -> str | None:
        return emit_rollout_status(
            cfg=self.cfg,
            env_metrics=env_metrics,
            rank=rank,
            last_logged_phase=last_logged_phase,
            log_info=log_info,
        )

    def enabled(self, mode: Literal["train", "eval"]) -> bool:
        if not self.td3_enabled():
            return False
        return self.local_correction_enabled() and self.env_type(mode) == "maniskill"

    def td3_enabled(self) -> bool:
        return (
            self.cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and self.cfg.actor.model.get("model_type", None) == "rlt_stage2"
        )

    def intervention_mode(self) -> str:
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        return str(intervention_cfg.get("mode", "local_correction"))

    def intervention_enabled(self) -> bool:
        intervention_cfg = self.cfg.algorithm.get("intervention", {})
        return (
            self.td3_enabled()
            and bool(intervention_cfg.get("enable", False))
            and self.intervention_mode() == "local_correction"
        )

    def local_correction_enabled(self) -> bool:
        return (
            self.intervention_enabled()
            and self.intervention_mode() == "local_correction"
        )

    def env_type(self, mode: Literal["train", "eval"]) -> str:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        return str(env_cfg.get("env_type", "")).lower()

    def _batch_size(self, mode: Literal["train", "eval"]) -> int:
        batch_size = self.train_batch_size if mode == "train" else self.eval_batch_size
        if batch_size is None:
            raise RuntimeError(f"RLT policy_info {mode} batch size is not initialized.")
        return int(batch_size)

    def _init_maniskill_controller(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        batch_size: int,
        env: Any | None,
    ) -> Any:
        from rlinf.envs.maniskill.rlt_intervention import (
            ManiSkillLocalCorrectionController,
        )

        controllers = self._maniskill_controllers(mode)
        self._ensure_len(controllers, stage_id, None)
        controller = ManiSkillLocalCorrectionController(
            cfg=self.cfg,
            batch_size=batch_size,
            mode=mode,
            hole_radii=self._get_maniskill_hole_radii(env),
        )
        controllers[stage_id] = controller
        return controller

    def _states(self, mode: Literal["train", "eval"]) -> list[dict[str, torch.Tensor]]:
        return self.train_states if mode == "train" else self.eval_states

    def _maniskill_controllers(self, mode: Literal["train", "eval"]) -> list[Any | None]:
        return (
            self.train_maniskill_controllers
            if mode == "train"
            else self.eval_maniskill_controllers
        )

    @staticmethod
    def _unwrap_env(env: Any) -> Any:
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "unwrapped", env)

    @classmethod
    def _get_maniskill_hole_radii(cls, env: Any | None) -> torch.Tensor | None:
        if env is None:
            return None
        unwrapped = cls._unwrap_env(env)
        if hasattr(unwrapped, "box_hole_radii"):
            return unwrapped.box_hole_radii
        return None

    @staticmethod
    def _ensure_len(target: list, stage_id: int, fill_value: Any) -> None:
        while len(target) <= stage_id:
            target.append(
                fill_value.copy() if isinstance(fill_value, dict) else fill_value
            )
