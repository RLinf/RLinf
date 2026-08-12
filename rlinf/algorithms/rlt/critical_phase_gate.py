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

"""Stateful STEAM-style critical-phase gate for RLT rollouts."""

from dataclasses import dataclass
from typing import Any, Literal

import torch

RLT_GATE_INFO_KEYS = (
    "rlt_gate_score_min",
    "rlt_gate_prediction_variance",
    "rlt_gate_critical_phase",
    "rlt_gate_entry_step",
    "rlt_gate_expert_requested",
    "rlt_gate_expert_entry_step",
)


@dataclass
class _GateState:
    z_history: torch.Tensor
    proprio_history: torch.Tensor
    valid_count: torch.Tensor
    low_progress_count: torch.Tensor
    latched: torch.Tensor
    entry_step: torch.Tensor
    actor_active: torch.Tensor
    critical_chunk_count: torch.Tensor
    expert_low_progress_count: torch.Tensor
    expert_latched: torch.Tensor
    expert_entry_step: torch.Tensor
    chunk_index: torch.Tensor


class SteamCriticalPhaseGate:
    """Detect sustained low progress from chunk-aligned RLT feature pairs."""

    def __init__(self, model: Any, cfg: Any) -> None:
        self.model = model
        self.mode = str(cfg.get("mode", "active"))
        if self.mode not in ("active", "shadow"):
            raise ValueError("critical phase gate mode must be 'active' or 'shadow'")
        self.chunk_size = int(cfg.get("chunk_size", 10))
        self.lookback_chunks = int(cfg.get("lookback_chunks", 3))
        self.patience_chunks = int(cfg.get("patience_chunks", 2))
        self.enter_threshold = float(cfg.get("enter_threshold", 0.0))
        self.latch_until_done = bool(cfg.get("latch_until_done", True))
        expert_cfg = cfg.get("expert_takeover", {}) or {}
        self.expert_takeover_enabled = bool(expert_cfg.get("enable", False))
        self.expert_enter_threshold = float(
            expert_cfg.get("enter_threshold", self.enter_threshold)
        )
        self.expert_warmup_chunks = int(
            expert_cfg.get("warmup_chunks", self.lookback_chunks)
        )
        self.expert_patience_chunks = int(expert_cfg.get("patience_chunks", 3))
        self.expert_latch_until_done = bool(expert_cfg.get("latch_until_done", True))
        if self.chunk_size < 1 or self.lookback_chunks < 1:
            raise ValueError("chunk_size and lookback_chunks must be positive")
        if self.patience_chunks < 1:
            raise ValueError("patience_chunks must be positive")
        if self.expert_warmup_chunks < self.lookback_chunks:
            raise ValueError(
                "expert_takeover.warmup_chunks must be at least lookback_chunks "
                "so expert decisions only compare actor-controlled observations"
            )
        if self.expert_patience_chunks < 1:
            raise ValueError("expert_takeover.patience_chunks must be positive")
        self._states: dict[tuple[str, int], _GateState] = {}

    def eval(self) -> None:
        self.model.eval()

    def requires_grad_(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)
        return self

    def to(self, device):
        self.model.to(device)
        self._states.clear()
        return self

    def reset(
        self,
        *,
        mode: Literal["train", "eval"] | None = None,
        stage_id: int | None = None,
    ) -> None:
        if mode is None and stage_id is None:
            self._states.clear()
            return
        for key in list(self._states):
            key_mode, key_stage = key
            if (mode is None or key_mode == mode) and (
                stage_id is None or key_stage == int(stage_id)
            ):
                del self._states[key]

    def _new_state(
        self,
        z_rl: torch.Tensor,
        proprio: torch.Tensor,
    ) -> _GateState:
        batch_size = z_rl.shape[0]
        history_len = self.lookback_chunks + 1
        return _GateState(
            z_history=torch.zeros(
                batch_size,
                history_len,
                z_rl.shape[-1],
                device=z_rl.device,
                dtype=z_rl.dtype,
            ),
            proprio_history=torch.zeros(
                batch_size,
                history_len,
                proprio.shape[-1],
                device=proprio.device,
                dtype=proprio.dtype,
            ),
            valid_count=torch.zeros(batch_size, device=z_rl.device, dtype=torch.long),
            low_progress_count=torch.zeros(
                batch_size, device=z_rl.device, dtype=torch.long
            ),
            latched=torch.zeros(batch_size, device=z_rl.device, dtype=torch.bool),
            entry_step=torch.zeros(batch_size, device=z_rl.device, dtype=torch.long),
            actor_active=torch.zeros(batch_size, device=z_rl.device, dtype=torch.bool),
            critical_chunk_count=torch.zeros(
                batch_size, device=z_rl.device, dtype=torch.long
            ),
            expert_low_progress_count=torch.zeros(
                batch_size, device=z_rl.device, dtype=torch.long
            ),
            expert_latched=torch.zeros(
                batch_size, device=z_rl.device, dtype=torch.bool
            ),
            expert_entry_step=torch.zeros(
                batch_size, device=z_rl.device, dtype=torch.long
            ),
            chunk_index=torch.zeros(batch_size, device=z_rl.device, dtype=torch.long),
        )

    @staticmethod
    def _normalize_reset_mask(
        reset_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if reset_mask is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        mask = torch.as_tensor(reset_mask, device=device, dtype=torch.bool)
        return mask.reshape(batch_size, -1).any(dim=1)

    @torch.no_grad()
    def update(
        self,
        rlt_obs: dict[str, torch.Tensor],
        *,
        mode: Literal["train", "eval"],
        stage_id: int,
        reset_mask: torch.Tensor | None = None,
        update_state: bool = True,
        actor_routing_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z_rl = rlt_obs["z_rl"].detach()
        proprio = rlt_obs["proprio"].detach()
        key = (mode, int(stage_id))
        state = self._states.get(key)
        if state is None or state.z_history.shape[0] != z_rl.shape[0]:
            state = self._new_state(z_rl, proprio)
            self._states[key] = state

        if not update_state:
            flags = (
                state.latched
                if self.mode == "active"
                else torch.zeros_like(state.latched)
            )
            expert_flags = (
                state.expert_latched
                if self.mode == "active"
                else torch.zeros_like(state.expert_latched)
            )
            return (
                flags[:, None],
                expert_flags[:, None],
                {
                    "rlt_gate_score_min": torch.zeros_like(
                        state.entry_step, dtype=torch.float32
                    )[:, None],
                    "rlt_gate_prediction_variance": torch.zeros_like(
                        state.entry_step, dtype=torch.float32
                    )[:, None],
                    "rlt_gate_critical_phase": state.latched[:, None],
                    "rlt_gate_entry_step": state.entry_step[:, None],
                    "rlt_gate_expert_requested": state.expert_latched[:, None],
                    "rlt_gate_expert_entry_step": state.expert_entry_step[:, None],
                },
            )

        reset = self._normalize_reset_mask(
            reset_mask,
            batch_size=z_rl.shape[0],
            device=z_rl.device,
        )
        if reset.any():
            state.z_history[reset] = 0
            state.proprio_history[reset] = 0
            state.valid_count[reset] = 0
            state.low_progress_count[reset] = 0
            state.latched[reset] = False
            state.entry_step[reset] = 0
            state.actor_active[reset] = False
            state.critical_chunk_count[reset] = 0
            state.expert_low_progress_count[reset] = 0
            state.expert_latched[reset] = False
            state.expert_entry_step[reset] = 0
            state.chunk_index[reset] = 0

        state.z_history = torch.roll(state.z_history, shifts=-1, dims=1)
        state.proprio_history = torch.roll(state.proprio_history, shifts=-1, dims=1)
        state.z_history[:, -1] = z_rl
        state.proprio_history[:, -1] = proprio
        state.valid_count = torch.clamp(
            state.valid_count + 1, max=self.lookback_chunks + 1
        )
        ready = state.valid_count >= (self.lookback_chunks + 1)

        output = self.model.predict(
            {
                "z_rl_t": state.z_history[:, 0],
                "proprio_t": state.proprio_history[:, 0],
                "z_rl_tk": state.z_history[:, -1],
                "proprio_tk": state.proprio_history[:, -1],
            }
        )
        score = output.predicted_values.to(dtype=torch.float32)
        variance = getattr(output, "prediction_variance", None)
        if variance is None:
            variance = torch.zeros_like(score)
        else:
            variance = variance.to(dtype=torch.float32)
        score = torch.where(ready, score, torch.zeros_like(score))
        variance = torch.where(ready, variance, torch.zeros_like(variance))

        low_progress = ready & (score <= self.enter_threshold)
        state.low_progress_count = torch.where(
            low_progress,
            state.low_progress_count + 1,
            torch.zeros_like(state.low_progress_count),
        )
        enter_now = (~state.latched) & (
            state.low_progress_count >= self.patience_chunks
        )
        state.entry_step = torch.where(
            enter_now,
            state.chunk_index * self.chunk_size,
            state.entry_step,
        )
        if self.latch_until_done:
            state.latched = state.latched | enter_now
        else:
            state.latched = low_progress

        actor_active = state.latched & bool(actor_routing_enabled)
        actor_started_now = actor_active & (~state.actor_active)
        state.actor_active = actor_active
        state.critical_chunk_count = torch.where(
            actor_started_now,
            torch.zeros_like(state.critical_chunk_count),
            torch.where(
                actor_active,
                state.critical_chunk_count + 1,
                torch.zeros_like(state.critical_chunk_count),
            ),
        )
        expert_ready = (
            self.expert_takeover_enabled
            & actor_active
            & (state.critical_chunk_count >= self.expert_warmup_chunks)
        )
        expert_low_progress = expert_ready & (score <= self.expert_enter_threshold)
        state.expert_low_progress_count = torch.where(
            expert_low_progress,
            state.expert_low_progress_count + 1,
            torch.zeros_like(state.expert_low_progress_count),
        )
        expert_enter_now = (~state.expert_latched) & (
            state.expert_low_progress_count >= self.expert_patience_chunks
        )
        state.expert_entry_step = torch.where(
            expert_enter_now,
            state.chunk_index * self.chunk_size,
            state.expert_entry_step,
        )
        if self.expert_latch_until_done:
            state.expert_latched = state.expert_latched | expert_enter_now
        else:
            state.expert_latched = expert_low_progress
        state.chunk_index = state.chunk_index + 1

        route_flags = state.latched
        route_expert_flags = state.expert_latched
        if self.mode == "shadow":
            route_flags = torch.zeros_like(route_flags)
            route_expert_flags = torch.zeros_like(route_expert_flags)
        return (
            route_flags[:, None],
            route_expert_flags[:, None],
            {
                "rlt_gate_score_min": score[:, None],
                "rlt_gate_prediction_variance": variance[:, None],
                "rlt_gate_critical_phase": state.latched[:, None],
                "rlt_gate_entry_step": state.entry_step[:, None],
                "rlt_gate_expert_requested": state.expert_latched[:, None],
                "rlt_gate_expert_entry_step": state.expert_entry_step[:, None],
            },
        )


__all__ = ["RLT_GATE_INFO_KEYS", "SteamCriticalPhaseGate"]
