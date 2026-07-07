# Copyright 2025 The RLinf Authors.
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

"""ManiSkill RLT environment with automatic actor/reference and expert switching."""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from omegaconf import DictConfig, open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.envs.maniskill.peg_insertion_side_variants import (
    RLT_OPENPI_JOINT_WRAP_MODE,
    init_peg_insertion_event_state,
    is_peg_insertion_side_env_id,
    maybe_augment_peg_insertion_info,
    patch_rlt_openpi_joint_env_args,
    reset_peg_insertion_event_state,
    resolve_maniskill_task_descriptions,
    restore_peg_insertion_event_state,
    snapshot_peg_insertion_event_state,
    wrap_rlt_openpi_joint_obs,
)

__all__ = ["ManiskillRLTEnv"]


def _to_bool_tensor(value, *, num_envs, device):
    if isinstance(value, bool):
        return torch.full((num_envs,), value, dtype=torch.bool, device=device)
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.bool)
    else:
        value = torch.as_tensor(value, device=device, dtype=torch.bool)
    if value.ndim == 0:
        value = value.reshape(1).repeat(num_envs)
    return value


def _shape_str(value):
    return "None" if value is None else str(tuple(getattr(value, "shape", ())))


def _extract_termination_from_info(info, num_envs, device, fallback=None):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(
                _to_bool_tensor(info["success"], num_envs=num_envs, device=device),
                _to_bool_tensor(info["fail"], num_envs=num_envs, device=device),
            )
        else:
            terminated = _to_bool_tensor(
                info["success"], num_envs=num_envs, device=device
            )
    else:
        if "fail" in info:
            terminated = _to_bool_tensor(info["fail"], num_envs=num_envs, device=device)
        else:
            if fallback is None:
                return torch.zeros(num_envs, dtype=bool, device=device)
            terminated = _to_bool_tensor(fallback, num_envs=num_envs, device=device)
    return terminated


class ManiskillRLTEnv(ManiskillEnv):
    """ManiSkill env with peg-insertion RLT policy switching."""

    _RLT_REQUIRED_INFO_KEYS = (
        "consecutive_grasp_current",
        "success_current",
        "peg_head_hole_x",
        "peg_head_hole_abs_y",
        "peg_head_hole_abs_z",
    )

    _RLT_FULL_TASK = "full_task"
    _RLT_CRITICAL_PHASE = "critical_phase"
    _RLT_AUTO_TRIGGER = "auto"
    _RLT_ALWAYS_ON_TRIGGER = "always_on"
    _RLT_INTERVENTION_GATE_TRIGGER = "intervention_gate"
    _RLT_STALLED_PROGRESS_TRIGGER = "stalled_progress"

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self._has_seeded_reset = False
        self.task_id = getattr(cfg.init_params, "id", None)
        self._is_peg_insertion_side = is_peg_insertion_side_env_id(self.task_id)
        self._rlt_switch_cfg = getattr(cfg, "rlt_policy_switch", None)
        self._rlt_switch_state: dict[str, torch.Tensor] | None = None
        self._rlt_hole_radii: torch.Tensor | None = None

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        env_args = patch_rlt_openpi_joint_env_args(
            env_args,
            wrap_obs_mode=getattr(cfg, "wrap_obs_mode", "default"),
        )
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )
        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        if self._is_peg_insertion_side:
            self.peg_event_state = init_peg_insertion_event_state(
                num_envs=self.num_envs,
                device=self.device,
            )
        self._show_goal_site_visual()
        if self.record_metrics:
            self._init_metrics()
        self._init_persistent_done_state()
        self._init_rlt_switch()

    @property
    def instruction(self):
        return resolve_maniskill_task_descriptions(
            self.env.unwrapped,
            num_envs=self.num_envs,
            is_peg_insertion_side=self._is_peg_insertion_side,
        )

    def _rlt_switch_enabled(self) -> bool:
        return self._rlt_switch_cfg is not None and bool(
            self._rlt_switch_cfg.get("enable", False)
        )

    def _init_rlt_switch(self) -> None:
        if not self._rlt_switch_enabled():
            return
        if not self._is_peg_insertion_side:
            raise ValueError(
                "ManiSkill RLT policy switch is only supported for peg-insertion tasks."
            )
        self._rlt_hole_radii = getattr(self.env.unwrapped, "box_hole_radii", None)
        self._rlt_switch_state = self._init_rlt_switch_state(self.num_envs)

    def _init_rlt_switch_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        task_mode = str(self._rlt_switch_cfg.get("task_mode", self._RLT_FULL_TASK))
        start_active = task_mode == self._RLT_CRITICAL_PHASE
        return {
            "rlt_use_actor": torch.full(
                (batch_size,),
                start_active,
                dtype=torch.bool,
            ),
            "entered_actor_phase_once": torch.full(
                (batch_size,),
                start_active,
                dtype=torch.bool,
            ),
            "actor_switch_step": torch.zeros(batch_size, dtype=torch.float32),
            "expert_takeover_active": torch.zeros(batch_size, dtype=torch.bool),
            "expert_progress_guard": torch.zeros(batch_size, dtype=torch.bool),
            "progress_initialized": torch.zeros(batch_size, dtype=torch.bool),
            "best_progress_x": torch.zeros(batch_size, dtype=torch.float32),
            "best_progress_yz": torch.zeros(batch_size, dtype=torch.float32),
            "best_progress_score": torch.zeros(batch_size, dtype=torch.float32),
            "stalled_progress_chunks": torch.zeros(batch_size, dtype=torch.float32),
        }

    def _reset_rlt_switch(self, env_idx=None) -> None:
        if self._rlt_switch_state is None:
            return
        new_state = self._init_rlt_switch_state(self.num_envs)
        if env_idx is None:
            self._rlt_switch_state = new_state
            return
        for key, value in new_state.items():
            state_value = self._rlt_switch_state[key]
            index = env_idx
            if isinstance(index, torch.Tensor):
                index = index.to(device=state_value.device)
            value = value.to(device=state_value.device)
            self._rlt_switch_state[key][index] = value[index]

    def _update_rlt_switch(
        self,
        *,
        infos: dict[str, Any],
        chunk_dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        infos = self._select_rlt_info_source(infos)
        device = infos["peg_head_hole_x"].device
        state = self._rlt_switch_state_to(device)

        task_mode = str(self._rlt_switch_cfg.get("task_mode", self._RLT_FULL_TASK))
        trigger_mode = str(
            self._rlt_switch_cfg.get("trigger_mode", self._RLT_AUTO_TRIGGER)
        )
        if (
            task_mode == self._RLT_CRITICAL_PHASE
            or trigger_mode == self._RLT_ALWAYS_ON_TRIGGER
        ):
            enter_actor = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        elif (
            task_mode == self._RLT_FULL_TASK and trigger_mode == self._RLT_AUTO_TRIGGER
        ):
            enter_actor = self._rlt_auto_enter_actor(infos, device)
        else:
            raise ValueError(
                "rlt_policy_switch supports task_mode in "
                f"{self._RLT_FULL_TASK, self._RLT_CRITICAL_PHASE} and trigger_mode in "
                f"{self._RLT_AUTO_TRIGGER, self._RLT_ALWAYS_ON_TRIGGER}, got "
                f"{task_mode=} {trigger_mode=}."
            )

        previous_use_actor = state["rlt_use_actor"]
        latch_until_done = bool(self._rlt_switch_cfg.get("latch_until_done", True))
        if latch_until_done:
            use_actor = previous_use_actor | enter_actor
        else:
            use_actor = enter_actor

        switched_now = (~previous_use_actor) & use_actor
        elapsed_steps = self._rlt_elapsed_steps(infos, device)
        state["actor_switch_step"] = torch.where(
            switched_now,
            elapsed_steps,
            state["actor_switch_step"],
        )
        state["entered_actor_phase_once"] = (
            state["entered_actor_phase_once"] | use_actor | switched_now
        )
        state["rlt_use_actor"] = use_actor
        self._update_rlt_expert_takeover_state(infos=infos, device=device)

        return self._export_rlt_switch_info(
            device=device,
            infos=infos,
            chunk_dones=chunk_dones,
        )

    def _export_rlt_switch_info(
        self,
        *,
        device: torch.device,
        infos: dict[str, Any] | None = None,
        chunk_dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self._rlt_switch_state_to(device)
        intervene_flag = self._rlt_expert_takeover_mask(infos=infos, device=device)
        rlt_switch_flags = state["rlt_use_actor"]
        if chunk_dones is not None:
            chunk_done_mask = torch.as_tensor(
                chunk_dones,
                device=device,
                dtype=torch.bool,
            ).reshape(self.num_envs, -1)
            rlt_switch_flags = rlt_switch_flags & (~chunk_done_mask.any(dim=1))
        return {
            "rlt_switch_flags": rlt_switch_flags[:, None],
            "intervene_flag": intervene_flag[:, None],
            "entered_actor_phase_once": state["entered_actor_phase_once"][:, None],
            "actor_switch_step": state["actor_switch_step"][:, None],
            "actor_switch_step_nonzero": torch.where(
                state["entered_actor_phase_once"],
                state["actor_switch_step"],
                torch.zeros_like(state["actor_switch_step"]),
            )[:, None],
        }

    def _rlt_expert_takeover_mask(
        self,
        *,
        infos: dict[str, Any] | None,
        device: torch.device,
    ) -> torch.Tensor:
        expert_cfg = self._rlt_switch_cfg.get("expert_takeover", {})
        if not bool(expert_cfg.get("enable", False)):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        trigger_mode = str(expert_cfg.get("trigger_mode", "critical_phase"))
        state = self._rlt_switch_state_to(device)
        if trigger_mode == "critical_phase":
            return state["rlt_use_actor"].clone()
        if trigger_mode == self._RLT_INTERVENTION_GATE_TRIGGER:
            return state["rlt_use_actor"] & self._rlt_expert_intervention_gate(
                infos=infos,
                expert_cfg=expert_cfg,
                device=device,
            )
        if trigger_mode == self._RLT_STALLED_PROGRESS_TRIGGER:
            takeover = state["rlt_use_actor"] & state["expert_takeover_active"]
            if infos is not None:
                takeover = takeover & (
                    ~self._rlt_info_bool(infos, "success_current", device)
                )
            return takeover
        if trigger_mode == "always_on":
            return torch.ones(self.num_envs, dtype=torch.bool, device=device)
        raise ValueError(
            "rlt_policy_switch.expert_takeover.trigger_mode supports "
            "'critical_phase', 'intervention_gate', 'stalled_progress', and "
            f"'always_on', got {trigger_mode!r}."
        )

    def _update_rlt_expert_takeover_state(
        self,
        *,
        infos: dict[str, Any],
        device: torch.device,
    ) -> None:
        expert_cfg = self._rlt_switch_cfg.get("expert_takeover", {})
        state = self._rlt_switch_state_to(device)
        if not bool(expert_cfg.get("enable", False)):
            state["expert_takeover_active"].zero_()
            state["expert_progress_guard"].zero_()
            state["progress_initialized"].zero_()
            state["stalled_progress_chunks"].zero_()
            return

        trigger_mode = str(expert_cfg.get("trigger_mode", "critical_phase"))
        if trigger_mode == self._RLT_STALLED_PROGRESS_TRIGGER:
            self._update_rlt_stalled_progress_expert_takeover(
                infos=infos,
                expert_cfg=expert_cfg,
                device=device,
            )
            return

        state["expert_takeover_active"].zero_()
        state["expert_progress_guard"].zero_()
        state["progress_initialized"].zero_()
        state["stalled_progress_chunks"].zero_()

    def _update_rlt_stalled_progress_expert_takeover(
        self,
        *,
        infos: dict[str, Any],
        expert_cfg: DictConfig | dict,
        device: torch.device,
    ) -> None:
        state = self._rlt_switch_state_to(device)
        gate_cfg = expert_cfg.get("gate", {})

        in_critical_phase = state["rlt_use_actor"]
        success = self._rlt_info_bool(infos, "success_current", device)
        active_before = state["expert_takeover_active"] & in_critical_phase & (~success)
        progress_guard = self._rlt_stalled_progress_guard(
            infos=infos,
            gate_cfg=gate_cfg,
            device=device,
        )
        state["expert_progress_guard"] = progress_guard

        eligible = in_critical_phase & progress_guard & (~success)
        if bool(gate_cfg.get("require_grasp", False)):
            eligible = eligible & self._rlt_info_bool(
                infos,
                "consecutive_grasp_current",
                device,
            )

        hole_x = self._rlt_info_float(infos, "peg_head_hole_x", device)
        abs_y = self._rlt_info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._rlt_info_float(infos, "peg_head_hole_abs_z", device)
        yz_dist = torch.sqrt(torch.square(abs_y) + torch.square(abs_z))
        yz_weight = float(gate_cfg.get("progress_yz_weight", 1.0))
        progress_score = hole_x - yz_weight * yz_dist

        initialized = state["progress_initialized"] & eligible & (~active_before)
        should_initialize = eligible & (~active_before) & (~initialized)
        monitor_progress = eligible & (~active_before) & initialized

        min_x_progress = float(gate_cfg.get("min_x_progress", 0.003))
        min_yz_progress = float(gate_cfg.get("min_yz_progress", 0.0015))
        min_score_progress = float(gate_cfg.get("min_score_progress", 0.002))
        x_improved = hole_x > (state["best_progress_x"] + min_x_progress)
        yz_improved = yz_dist < (state["best_progress_yz"] - min_yz_progress)
        score_improved = progress_score > (
            state["best_progress_score"] + min_score_progress
        )
        improved = monitor_progress & (x_improved | yz_improved | score_improved)

        no_progress = monitor_progress & (~improved)
        stalled_chunks = torch.where(
            no_progress,
            state["stalled_progress_chunks"] + 1.0,
            state["stalled_progress_chunks"],
        )
        stalled_chunks = torch.where(
            improved | should_initialize | (~eligible),
            torch.zeros_like(stalled_chunks),
            stalled_chunks,
        )

        stuck_chunks_before_takeover = max(
            1,
            int(gate_cfg.get("stuck_chunks_before_takeover", 3)),
        )
        trigger_now = no_progress & (
            stalled_chunks >= float(stuck_chunks_before_takeover)
        )

        update_best = should_initialize | improved
        state["best_progress_x"] = torch.where(
            should_initialize,
            hole_x,
            torch.where(
                update_best,
                torch.maximum(state["best_progress_x"], hole_x),
                state["best_progress_x"],
            ),
        )
        state["best_progress_yz"] = torch.where(
            should_initialize,
            yz_dist,
            torch.where(
                update_best,
                torch.minimum(state["best_progress_yz"], yz_dist),
                state["best_progress_yz"],
            ),
        )
        state["best_progress_score"] = torch.where(
            should_initialize,
            progress_score,
            torch.where(
                update_best,
                torch.maximum(state["best_progress_score"], progress_score),
                state["best_progress_score"],
            ),
        )

        state["expert_takeover_active"] = active_before | trigger_now
        state["progress_initialized"] = torch.where(
            eligible & (~active_before),
            state["progress_initialized"] | should_initialize,
            torch.zeros_like(state["progress_initialized"]),
        )
        state["stalled_progress_chunks"] = torch.where(
            active_before,
            state["stalled_progress_chunks"],
            stalled_chunks,
        )

    def _rlt_stalled_progress_guard(
        self,
        *,
        infos: dict[str, Any],
        gate_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        hole_x = self._rlt_info_float(infos, "peg_head_hole_x", device)
        abs_y = self._rlt_info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._rlt_info_float(infos, "peg_head_hole_abs_z", device)
        hole_radii = self._rlt_hole_radii(abs_y, gate_cfg, device)

        near_hole_x_min = float(gate_cfg.get("near_hole_x_min", -0.10))
        yz_margin = float(gate_cfg.get("near_hole_yz_margin", 2.0))
        guard = (
            (hole_x >= near_hole_x_min)
            & (abs_y <= yz_margin * hole_radii)
            & (abs_z <= yz_margin * hole_radii)
        )
        if bool(gate_cfg.get("require_prealigned", False)):
            guard = guard & self._rlt_prealigned_mask(
                infos=infos,
                gate_cfg=gate_cfg,
                device=device,
            )
        return guard

    def _rlt_expert_intervention_gate(
        self,
        *,
        infos: dict[str, Any] | None,
        expert_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        if infos is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        gate_cfg = expert_cfg.get("gate", {})
        if "partial_insert_current" in infos and not gate_cfg:
            gate = self._rlt_info_bool(infos, "partial_insert_current", device)
        else:
            gate = self._rlt_threshold_gate(
                infos=infos,
                gate_cfg=gate_cfg,
                device=device,
            )

        if bool(gate_cfg.get("require_grasp", False)):
            gate = gate & self._rlt_info_bool(
                infos,
                "consecutive_grasp_current",
                device,
            )
        if bool(gate_cfg.get("require_not_success", True)):
            gate = gate & (~self._rlt_info_bool(infos, "success_current", device))
        return gate

    def _rlt_threshold_gate(
        self,
        *,
        infos: dict[str, Any],
        gate_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        prealigned = self._rlt_prealigned_mask(
            infos=infos,
            gate_cfg=gate_cfg,
            device=device,
        )

        hole_x = self._rlt_info_float(infos, "peg_head_hole_x", device)
        abs_y = self._rlt_info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._rlt_info_float(infos, "peg_head_hole_abs_z", device)
        hole_radii = self._rlt_hole_radii(abs_y, gate_cfg, device)

        near_hole_x_min = float(gate_cfg.get("near_hole_x_min", -0.05))
        yz_margin = float(gate_cfg.get("near_hole_yz_margin", 1.25))
        return (
            prealigned
            & (hole_x >= near_hole_x_min)
            & (abs_y <= yz_margin * hole_radii)
            & (abs_z <= yz_margin * hole_radii)
        )

    def _rlt_prealigned_mask(
        self,
        *,
        infos: dict[str, Any],
        gate_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        if "prealigned_current" in infos:
            return self._rlt_info_bool(infos, "prealigned_current", device)

        yz_threshold = float(gate_cfg.get("prealign_yz_threshold", 0.01))
        return (
            self._rlt_info_float(infos, "peg_head_goal_yz_dist", device) < yz_threshold
        ) & (
            self._rlt_info_float(infos, "peg_body_goal_yz_dist", device) < yz_threshold
        )

    def _rlt_info_bool(
        self,
        infos: dict[str, Any],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        return self._rlt_info_tensor(
            infos,
            key,
            dtype=torch.bool,
            device=device,
        )

    def _rlt_info_float(
        self,
        infos: dict[str, Any],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        return self._rlt_info_tensor(
            infos,
            key,
            dtype=torch.float32,
            device=device,
        )

    def _rlt_info_tensor(
        self,
        infos: dict[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        value = torch.as_tensor(infos[key], device=device).to(dtype=dtype)
        if value.numel() == 1:
            return value.reshape(1).repeat(self.num_envs)
        if value.numel() == self.num_envs:
            return value.reshape(self.num_envs)
        return value.reshape(self.num_envs, -1)[:, -1]

    def _select_rlt_info_source(self, infos: dict[str, Any]) -> dict[str, Any]:
        if all(key in infos for key in self._RLT_REQUIRED_INFO_KEYS):
            return infos
        final_info = infos.get("final_info")
        if isinstance(final_info, dict) and all(
            key in final_info for key in self._RLT_REQUIRED_INFO_KEYS
        ):
            return final_info
        missing = [key for key in self._RLT_REQUIRED_INFO_KEYS if key not in infos]
        raise RuntimeError(
            "RLT policy switch is enabled, but ManiSkill info is missing "
            f"required keys {missing}. This usually means the env wrapper is not "
            "using the aligned peg-insertion info path."
        )

    def _rlt_auto_enter_actor(
        self,
        infos: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        auto_gate = self._rlt_switch_cfg.get("auto_gate", {})
        grasp = self._rlt_info_bool(infos, "consecutive_grasp_current", device)
        success = self._rlt_info_bool(infos, "success_current", device)
        hole_x = self._rlt_info_float(infos, "peg_head_hole_x", device)
        abs_y = self._rlt_info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._rlt_info_float(infos, "peg_head_hole_abs_z", device)
        hole_radii = self._rlt_hole_radii(abs_y, auto_gate, device)

        near_hole_x_min = float(auto_gate.get("near_hole_x_min", -0.16))
        yz_margin = float(auto_gate.get("near_hole_yz_margin", 1.5))
        near_hole = (
            (hole_x >= near_hole_x_min)
            & (abs_y <= yz_margin * hole_radii)
            & (abs_z <= yz_margin * hole_radii)
        )

        enter_actor = near_hole
        if bool(auto_gate.get("require_grasp", True)):
            enter_actor = enter_actor & grasp
        if bool(auto_gate.get("require_not_success", True)):
            enter_actor = enter_actor & (~success)
        return enter_actor

    def _rlt_hole_radii(
        self,
        like: torch.Tensor,
        auto_gate: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        if self._rlt_hole_radii is not None:
            return self._rlt_hole_radii.to(device, dtype=torch.float32)
        fallback_hole_radius = auto_gate.get("fallback_hole_radius", 0.03)
        return torch.full_like(like, float(fallback_hole_radius))

    def _rlt_elapsed_steps(
        self,
        infos: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        elapsed_steps = infos.get("elapsed_steps", infos.get("_elapsed_steps"))
        if elapsed_steps is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        elapsed_steps = torch.as_tensor(
            elapsed_steps,
            dtype=torch.float32,
            device=device,
        )
        if elapsed_steps.numel() == 1:
            return elapsed_steps.reshape(1).repeat(self.num_envs)
        return elapsed_steps.reshape(self.num_envs, -1)[:, -1]

    def _rlt_switch_state_to(self, device: torch.device) -> dict[str, torch.Tensor]:
        for key, value in self._rlt_switch_state.items():
            self._rlt_switch_state[key] = value.to(device)
        return self._rlt_switch_state

    def _init_persistent_done_state(self):
        self._persistent_done_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._persistent_done_obs = None
        self._persistent_done_infos = None

    def _reset_persistent_done_state(self, env_idx=None):
        if not hasattr(self, "_persistent_done_mask"):
            self._init_persistent_done_state()
            return

        if env_idx is None:
            self._persistent_done_mask.zero_()
            self._persistent_done_obs = None
            self._persistent_done_infos = None
            return

        self._persistent_done_mask[env_idx] = False

    def _update_persistent_done_state(self, dones, extracted_obs, infos):
        if self.auto_reset or not dones.any():
            return

        newly_done = dones & (~self._persistent_done_mask)
        if not newly_done.any():
            return

        if self._persistent_done_obs is None:
            self._persistent_done_obs = torch_clone_dict(extracted_obs)
        else:
            self._persistent_done_obs = self._restore_frozen_values(
                self._persistent_done_obs, extracted_obs, newly_done
            )

        if self._persistent_done_infos is None:
            self._persistent_done_infos = torch_clone_dict(infos)
        else:
            self._persistent_done_infos = self._restore_frozen_values(
                self._persistent_done_infos, infos, newly_done
            )

        self._persistent_done_mask |= newly_done

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == RLT_OPENPI_JOINT_WRAP_MODE:
            if self.env.unwrapped.obs_mode != "rgb":
                raise ValueError(
                    "wrap_obs_mode='rlt_openpi_joint' requires ManiSkill obs_mode='rgb'."
                )
            return wrap_rlt_openpi_joint_obs(
                raw_obs,
                infos=infos,
                task_descriptions=self.instruction,
                num_envs=self.num_envs,
                device=self.device,
                is_peg_insertion_side=self._is_peg_insertion_side,
            )
        return super()._wrap_obs(raw_obs, infos=infos)

    def _record_metrics(self, step_reward, infos):
        infos = super()._record_metrics(step_reward, infos)
        if not self.record_metrics or "episode" not in infos:
            return infos
        for key in (
            "entered_actor_phase_once",
            "actor_switch_step",
            "actor_switch_step_nonzero",
        ):
            if key in infos:
                value = infos[key]
                if isinstance(value, torch.Tensor):
                    infos["episode"][key] = value.reshape(self.num_envs, -1)[
                        :, -1
                    ].clone()
        return infos

    def _attach_rlt_switch_info(self, infos):
        if self._rlt_switch_state is None:
            return
        switch_info = self._export_rlt_switch_info(
            device=self.device,
            infos=infos,
        )
        for key in ("rlt_switch_flags", "intervene_flag"):
            infos[key] = switch_info[key]
        for key in (
            "entered_actor_phase_once",
            "actor_switch_step",
            "actor_switch_step_nonzero",
        ):
            if key in switch_info:
                infos[key] = switch_info[key]

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
            if seed is None:
                if self.use_fixed_reset_state_ids or not self._has_seeded_reset:
                    seed = self.seed
        if seed is not None:
            self._has_seeded_reset = True
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            if self._is_peg_insertion_side:
                reset_peg_insertion_event_state(
                    self.peg_event_state,
                    env_idx=env_idx,
                )
            self._reset_metrics(env_idx)
        else:
            if self._is_peg_insertion_side:
                reset_peg_insertion_event_state(self.peg_event_state)
            self._reset_metrics()
        self._reset_persistent_done_state(options.get("env_idx"))
        self._reset_rlt_switch(options.get("env_idx"))
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        return extracted_obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        infos = maybe_augment_peg_insertion_info(
            env=self.env.unwrapped,
            infos=infos,
            event_state=getattr(self, "peg_event_state", None),
            device=self.device,
            is_peg_insertion_side=self._is_peg_insertion_side,
        )
        infos["elapsed_steps"] = self.elapsed_steps.clone()
        terminations = _extract_termination_from_info(
            infos,
            num_envs=self.num_envs,
            device=self.device,
            fallback=terminations,
        )
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        step_reward = self._calc_step_reward(_reward, infos)

        if self.record_metrics:
            infos = self._record_metrics(step_reward, infos)
        self._attach_rlt_switch_info(infos)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def _snapshot_episode_state(self):
        state = {
            "prev_step_reward": self.prev_step_reward.clone(),
        }
        if self.record_metrics:
            state.update(
                {
                    "success_once": self.success_once.clone(),
                    "fail_once": self.fail_once.clone(),
                    "returns": self.returns.clone(),
                }
            )
        if self._is_peg_insertion_side:
            state["peg_event_state"] = snapshot_peg_insertion_event_state(
                self.peg_event_state
            )
        return state

    def _restore_episode_state(self, state, mask):
        self.prev_step_reward[mask] = state["prev_step_reward"][mask]
        if self.record_metrics:
            self.success_once[mask] = state["success_once"][mask]
            self.fail_once[mask] = state["fail_once"][mask]
            self.returns[mask] = state["returns"][mask]
        if self._is_peg_insertion_side:
            restore_peg_insertion_event_state(
                self.peg_event_state,
                state["peg_event_state"],
                mask,
            )

    def _zero_frozen_actions(self, actions, mask):
        if not mask.any():
            return actions
        if isinstance(actions, torch.Tensor):
            frozen_mask = mask.to(device=actions.device)
            actions = actions.clone()
            actions[frozen_mask] = 0.0
            return actions

        frozen_mask = mask.detach().cpu().numpy()
        actions = np.asarray(actions).copy()
        actions[frozen_mask] = 0.0
        return actions

    def _restore_frozen_values(self, values, previous_values, mask):
        if not isinstance(values, dict) or not isinstance(previous_values, dict):
            return values

        restored = torch_clone_dict(values)
        for key, prev_value in previous_values.items():
            if key not in restored:
                continue
            value = restored[key]
            if isinstance(value, torch.Tensor) and isinstance(prev_value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] == self.num_envs:
                    value_mask = mask.to(device=value.device)
                    value[value_mask] = prev_value.to(value.device)[value_mask]
            elif isinstance(value, dict) and isinstance(prev_value, dict):
                restored[key] = self._restore_frozen_values(value, prev_value, mask)
            elif isinstance(value, list) and isinstance(prev_value, list):
                mask_cpu = mask.detach().cpu().numpy()
                for env_idx, should_restore in enumerate(mask_cpu):
                    if (
                        should_restore
                        and env_idx < len(value)
                        and env_idx < len(prev_value)
                    ):
                        value[env_idx] = prev_value[env_idx]
        return restored

    def _restore_frozen_info_values(self, infos, previous_infos, mask):
        return self._restore_frozen_values(infos, previous_infos, mask)

    def _validate_chunk_actions(self, chunk_actions) -> None:
        if not hasattr(chunk_actions, "shape") or len(chunk_actions.shape) != 3:
            raise ValueError(
                "ManiskillRLTEnv.chunk_step expected action chunk shape "
                f"[num_envs, chunk_steps, action_dim], got {_shape_str(chunk_actions)}. "
                "Refuse to execute malformed actions."
            )

        if int(chunk_actions.shape[0]) != self.num_envs:
            raise ValueError(
                "ManiskillRLTEnv.chunk_step action batch mismatch: expected "
                f"num_envs={self.num_envs}, got shape {_shape_str(chunk_actions)}. "
                "Refuse to execute actions for the wrong env batch."
            )

        expected_action_dim = getattr(
            self.env.unwrapped.single_action_space, "shape", None
        )
        expected_action_dim = expected_action_dim[-1] if expected_action_dim else None
        if expected_action_dim is not None and int(chunk_actions.shape[2]) != int(
            expected_action_dim
        ):
            raise ValueError(
                "ManiskillRLTEnv.chunk_step action dim mismatch before env.step: "
                f"expected action_dim={expected_action_dim}, got shape "
                f"{_shape_str(chunk_actions)}. Check actor.model.action_dim and "
                "ManiSkill control_mode."
            )

    def _apply_chunk_rlt_switch_info(
        self,
        infos_list: list[dict[str, Any]],
        chunk_dones: torch.Tensor,
    ) -> None:
        if self._rlt_switch_state is None:
            return
        infos_last = infos_list[-1]
        switch_info = self._update_rlt_switch(
            infos=infos_last,
            chunk_dones=chunk_dones,
        )
        for key in ("rlt_switch_flags", "intervene_flag"):
            infos_last[key] = switch_info[key]
        for key in (
            "entered_actor_phase_once",
            "actor_switch_step",
            "actor_switch_step_nonzero",
        ):
            if key in switch_info:
                infos_last[key] = switch_info[key]
        infos_list[-1] = infos_last

    def _stack_chunk_rlt_flags(self, infos_list: list[dict[str, Any]]) -> None:
        raw_chunk_rlt_switch_flags = []
        raw_chunk_intervene_flag = []
        for infos in infos_list:
            if "rlt_switch_flags" in infos:
                raw_chunk_rlt_switch_flags.append(infos["rlt_switch_flags"])
            if "intervene_flag" in infos:
                raw_chunk_intervene_flag.append(infos["intervene_flag"])

        if not infos_list:
            return

        infos_last = infos_list[-1]
        if raw_chunk_rlt_switch_flags:
            infos_last["rlt_switch_flags"] = torch.stack(
                raw_chunk_rlt_switch_flags, dim=1
            )
        if raw_chunk_intervene_flag:
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
        infos_list[-1] = infos_last

    def _sync_rlt_switch_episode_info(self, infos):
        if not isinstance(infos, dict) or "episode" not in infos:
            return
        for key in (
            "entered_actor_phase_once",
            "actor_switch_step",
            "actor_switch_step_nonzero",
        ):
            if key in infos:
                infos["episode"][key] = (
                    infos[key].reshape(self.num_envs, -1)[:, -1].clone()
                )

    def chunk_step(self, chunk_actions):
        self._validate_chunk_actions(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        if not hasattr(self, "_persistent_done_mask"):
            self._init_persistent_done_state()
        frozen_dones = (
            self._persistent_done_mask.clone()
            if not self.auto_reset
            else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        )
        initially_frozen_dones = frozen_dones.clone()
        last_extracted_obs = (
            torch_clone_dict(self._persistent_done_obs)
            if frozen_dones.any() and self._persistent_done_obs is not None
            else None
        )
        last_infos = (
            torch_clone_dict(self._persistent_done_infos)
            if frozen_dones.any() and self._persistent_done_infos is not None
            else None
        )
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            if (
                frozen_dones.all()
                and last_extracted_obs is not None
                and last_infos is not None
            ):
                extracted_obs = torch_clone_dict(last_extracted_obs)
                infos = torch_clone_dict(last_infos)
                step_reward = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.float32
                )
                terminations = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
                truncations = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
            else:
                state_before_step = self._snapshot_episode_state()
                actions = self._zero_frozen_actions(actions, frozen_dones)
                extracted_obs, step_reward, terminations, truncations, infos = (
                    self.step(actions, auto_reset=False)
                )
                if frozen_dones.any():
                    self._restore_episode_state(state_before_step, frozen_dones)
                    if last_extracted_obs is not None:
                        extracted_obs = self._restore_frozen_values(
                            extracted_obs, last_extracted_obs, frozen_dones
                        )
                    step_reward = step_reward.clone()
                    step_reward[frozen_dones] = 0.0
                    terminations = terminations.clone()
                    truncations = truncations.clone()
                    terminations[frozen_dones] = False
                    truncations[frozen_dones] = False
                    if last_infos is not None:
                        infos = self._restore_frozen_info_values(
                            infos, last_infos, frozen_dones
                        )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            frozen_dones |= torch.logical_or(terminations, truncations)
            last_extracted_obs = torch_clone_dict(extracted_obs)
            last_infos = torch_clone_dict(infos)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        policy_switch_chunk_dones = torch.logical_or(
            raw_chunk_terminations,
            raw_chunk_truncations,
        )
        if initially_frozen_dones.any():
            policy_switch_chunk_dones = policy_switch_chunk_dones | (
                initially_frozen_dones[:, None].expand_as(policy_switch_chunk_dones)
            )
        self._apply_chunk_rlt_switch_info(
            infos_list=infos_list,
            chunk_dones=policy_switch_chunk_dones,
        )
        self._sync_rlt_switch_episode_info(infos_list[-1])
        self._stack_chunk_rlt_flags(infos_list)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )
        elif past_dones.any():
            self._update_persistent_done_state(past_dones, obs_list[-1], infos_list[-1])

        return (
            obs_list,
            chunk_rewards,
            raw_chunk_terminations,
            raw_chunk_truncations,
            infos_list,
        )
