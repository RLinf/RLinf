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

"""Build and update EnvWorker policy_info for RLT Stage 2 rollout routing."""

from __future__ import annotations

from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.envs.maniskill.rlt_intervention import ManiSkillLocalCorrectionController


class RLTStage2PolicyInfoAdapter:
    """Owns RLT Stage 2 env-side state exposed to rollout workers.

    The env worker should only orchestrate stage lifecycle and channel IO. This
    adapter keeps the RLT-specific policy_info state machine near env semantics:
    ManiSkill local correction, realworld critical-phase flags, and generic
    policy_info coercion from env ``infos``.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        train_batch_size: int | None,
        eval_batch_size: int | None,
    ) -> None:
        self.cfg = cfg
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_states: list[dict[str, torch.Tensor]] = []
        self.eval_states: list[dict[str, torch.Tensor]] = []
        self.train_maniskill_controllers: list[
            ManiSkillLocalCorrectionController | None
        ] = []
        self.eval_maniskill_controllers: list[
            ManiSkillLocalCorrectionController | None
        ] = []

    def init_stage(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Initialize policy_info state for one local env stage."""
        if not self.enabled(mode):
            return None

        batch_size = self._batch_size(mode)
        if self.env_type(mode) == "maniskill":
            controller = self._init_maniskill_controller(
                stage_id=stage_id,
                mode=mode,
                batch_size=batch_size,
                env=env,
            )
            states = self._states(mode)
            self._ensure_len(states, stage_id, {})
            states[stage_id] = controller.state
            return controller.export_policy_info(controller.state)

        state = self._init_generic_state(batch_size, mode=mode)
        states = self._states(mode)
        self._ensure_len(states, stage_id, {})
        states[stage_id] = state
        return self.export_policy_info(state)

    def update_stage(
        self,
        *,
        infos: dict[str, Any] | None,
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
        env: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Update policy_info state after one env chunk."""
        states = self._states(mode)
        if not self.enabled(mode) or infos is None or not states:
            return None

        if self.env_type(mode) != "maniskill":
            return self._update_generic_stage(
                infos=infos,
                chunk_dones=chunk_dones,
                stage_id=stage_id,
                mode=mode,
            )

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

    def enabled(self, mode: Literal["train", "eval"]) -> bool:
        if self.local_correction_enabled():
            return True
        return self.intervention_enabled() and self.env_type(mode) == "realworld"

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
            and self.intervention_mode() in {"local_correction", "human_override"}
        )

    def local_correction_enabled(self) -> bool:
        return (
            self.intervention_enabled()
            and self.intervention_mode() == "local_correction"
        )

    def env_type(self, mode: Literal["train", "eval"]) -> str:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        return str(env_cfg.get("env_type", "")).lower()

    @staticmethod
    def export_policy_info(
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return ManiSkillLocalCorrectionController.export_policy_info(state)

    def _init_maniskill_controller(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        batch_size: int,
        env: Any | None,
    ) -> ManiSkillLocalCorrectionController:
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

    def _init_generic_state(
        self,
        batch_size: int,
        *,
        mode: Literal["train", "eval"],
    ) -> dict[str, torch.Tensor]:
        return {
            "intervention_region": torch.zeros(batch_size, dtype=torch.bool),
            "intervention_phase": torch.zeros(batch_size, dtype=torch.int64),
            "expert_takeover": torch.zeros(batch_size, dtype=torch.bool),
            "deviation": torch.zeros(batch_size, dtype=torch.bool),
            "deviation_count": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_left": torch.zeros(batch_size, dtype=torch.int64),
            "takeover_used": torch.zeros(batch_size, dtype=torch.int64),
            "prev_yz_error": torch.full(
                (batch_size,),
                float("nan"),
                dtype=torch.float32,
            ),
            "prev_hole_x": torch.full(
                (batch_size,),
                float("nan"),
                dtype=torch.float32,
            ),
            "in_critical_phase": torch.full(
                (batch_size,),
                self._default_in_critical_phase(mode),
                dtype=torch.bool,
            ),
            "record_transition": torch.full(
                (batch_size,),
                self._default_record_transition(mode),
                dtype=torch.bool,
            ),
            "critical_phase_started": torch.full(
                (batch_size,),
                self._default_in_critical_phase(mode),
                dtype=torch.bool,
            ),
        }

    def _update_generic_stage(
        self,
        *,
        infos: dict[str, Any],
        chunk_dones: torch.Tensor,
        stage_id: int,
        mode: Literal["train", "eval"],
    ) -> dict[str, torch.Tensor]:
        states = self._states(mode)
        state = states[stage_id]
        device = chunk_dones.device
        done_any = chunk_dones.any(dim=1).to(device)
        batch_size = int(done_any.shape[0])
        for key, value in state.items():
            state[key] = value.to(device)

        expert_takeover = self._coerce_bool_info(
            self._lookup_info_value(infos, "expert_takeover"),
            batch_size=batch_size,
            device=device,
        )
        deviation = self._coerce_bool_info(
            self._lookup_info_value(infos, "deviation"),
            batch_size=batch_size,
            device=device,
        )
        intervention_region = self._coerce_bool_info(
            self._lookup_info_value(infos, "intervention_region"),
            batch_size=batch_size,
            device=device,
        )
        state["expert_takeover"] = torch.where(
            done_any,
            torch.zeros_like(expert_takeover),
            expert_takeover,
        )
        state["deviation"] = torch.where(done_any, torch.zeros_like(deviation), deviation)
        state["intervention_region"] = torch.where(
            done_any,
            torch.zeros_like(intervention_region),
            intervention_region,
        )

        for key in (
            "in_critical_phase",
            "record_transition",
            "critical_phase_started",
        ):
            if key not in state:
                continue
            default_value = (
                self._default_record_transition(mode)
                if key == "record_transition"
                else self._default_in_critical_phase(mode)
            )
            default_tensor = torch.full_like(state[key], default_value)
            current_value = self._coerce_bool_info(
                self._lookup_info_value(infos, key),
                batch_size=batch_size,
                device=device,
            )
            state[key] = torch.where(done_any, default_tensor, current_value)

        for key in (
            "deviation_count",
            "takeover_left",
            "takeover_used",
        ):
            state[key] = torch.where(
                done_any,
                torch.zeros_like(state[key]),
                self._coerce_int_info(
                    self._lookup_info_value(infos, key),
                    batch_size=batch_size,
                    device=device,
                ),
            )

        state["prev_yz_error"] = torch.full_like(state["prev_yz_error"], float("nan"))
        state["prev_hole_x"] = torch.full_like(state["prev_hole_x"], float("nan"))
        state["intervention_phase"] = torch.where(
            done_any,
            torch.zeros_like(state["intervention_phase"]),
            state["intervention_phase"],
        )
        return self.export_policy_info(state)

    def _batch_size(self, mode: Literal["train", "eval"]) -> int:
        batch_size = self.train_batch_size if mode == "train" else self.eval_batch_size
        if batch_size is None:
            raise RuntimeError(f"RLT policy_info {mode} batch size is not initialized.")
        return int(batch_size)

    def _states(
        self,
        mode: Literal["train", "eval"],
    ) -> list[dict[str, torch.Tensor]]:
        return self.train_states if mode == "train" else self.eval_states

    def _maniskill_controllers(
        self,
        mode: Literal["train", "eval"],
    ) -> list[ManiSkillLocalCorrectionController | None]:
        return (
            self.train_maniskill_controllers
            if mode == "train"
            else self.eval_maniskill_controllers
        )

    def _realworld_task_mode(self, mode: Literal["train", "eval"]) -> str:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        return str(env_cfg.get("task_mode", "critical_phase"))

    def _default_in_critical_phase(self, mode: Literal["train", "eval"]) -> bool:
        return self._realworld_task_mode(mode) == "critical_phase"

    def _default_record_transition(self, mode: Literal["train", "eval"]) -> bool:
        env_cfg = self.cfg.env.train if mode == "train" else self.cfg.env.eval
        if bool(env_cfg.get("record_prefix_before_critical_phase", False)):
            return True
        return self._default_in_critical_phase(mode)

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
    def _lookup_info_value(infos: dict[str, Any], key: str) -> Any:
        if key in infos:
            return infos[key]
        policy_info = infos.get("policy_info")
        if isinstance(policy_info, dict) and key in policy_info:
            return policy_info[key]
        final_info = infos.get("final_info")
        if isinstance(final_info, dict):
            if key in final_info:
                return final_info[key]
            final_policy_info = final_info.get("policy_info")
            if isinstance(final_policy_info, dict) and key in final_policy_info:
                return final_policy_info[key]
        return None

    @staticmethod
    def _coerce_bool_info(
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 1:
            return torch.full(
                (batch_size,),
                bool(tensor.reshape(-1)[0].item()),
                dtype=torch.bool,
                device=device,
            )
        tensor = tensor.reshape(batch_size, -1)
        return tensor.to(torch.bool).any(dim=1)

    @staticmethod
    def _coerce_int_info(
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.int64, device=device)
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() == 1:
            return torch.full(
                (batch_size,),
                int(tensor.reshape(-1)[0].item()),
                dtype=torch.int64,
                device=device,
            )
        tensor = tensor.reshape(batch_size, -1)
        return tensor[:, -1].to(torch.int64)

    @staticmethod
    def _ensure_len(target: list, stage_id: int, fill_value: Any) -> None:
        while len(target) <= stage_id:
            target.append(
                fill_value.copy() if isinstance(fill_value, dict) else fill_value
            )
