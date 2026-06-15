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

"""Build and update ManiSkill policy_info for RLT Stage 2 rollout routing."""

from __future__ import annotations

from typing import Any, Literal

import torch
from omegaconf import DictConfig


class RLTStage2PolicyInfoAdapter:
    """Owns RLT Stage 2 env-side state exposed to rollout workers.

    The env worker should only orchestrate stage lifecycle and channel IO. This
    adapter keeps the ManiSkill local-correction state machine out of EnvWorker.
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
        self.train_maniskill_controllers: list[Any | None] = []
        self.eval_maniskill_controllers: list[Any | None] = []

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
        mode = self.intervention_mode()
        if mode != "local_correction":
            raise ValueError(
                "RLT Stage2 ManiSkill policy_info only supports "
                "algorithm.intervention.mode='local_correction', got "
                f"{mode!r}."
            )
        return (
            self.td3_enabled()
            and bool(intervention_cfg.get("enable", False))
            and mode == "local_correction"
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
        policy_info = {
            "expert_takeover": state["expert_takeover"][:, None],
            "deviation": state["deviation"][:, None],
            "deviation_count": state["deviation_count"].to(torch.float32)[:, None],
            "intervention_phase": state["intervention_phase"].to(torch.float32)[
                :, None
            ],
            "takeover_left": state["takeover_left"].to(torch.float32)[:, None],
            "takeover_used": state["takeover_used"].to(torch.float32)[:, None],
        }
        for key in ("in_critical_phase", "record_transition"):
            if key in state:
                policy_info[key] = state[key].to(torch.bool)[:, None]
        return policy_info

    def _init_maniskill_controller(
        self,
        *,
        stage_id: int,
        mode: Literal["train", "eval"],
        batch_size: int,
        env: Any | None,
    ) -> Any:
        from .rlt_intervention import ManiSkillLocalCorrectionController

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
    ) -> list[Any | None]:
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
