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

"""Stateful STEAM critical-phase gate for RLT rollouts."""

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.phase_head import SteamPhaseHead
from rlinf.data.datasets.steam import BinaryPairDataCollator
from rlinf.data.datasets.steam.pair_dataset import _to_uint8_hwc

RLT_GATE_INFO_KEYS = (
    "rlt_gate_entered",
    "rlt_gate_entry_step",
    "rlt_gate_score_ready",
    "rlt_gate_score_min",
    "rlt_gate_actor_active",
    "rlt_route_expert_entered",
    "rlt_route_expert_entry_step",
)


@dataclass
class GateDecision:
    """Routing decision and diagnostics emitted by one gate step."""

    actor_switch: torch.Tensor
    expert_requested: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    phase_features: torch.Tensor | None = None


@dataclass
class _GateState:
    image_history: dict[str, np.ndarray]
    valid_count: torch.Tensor
    low_progress_count: torch.Tensor
    latched: torch.Tensor
    entered: torch.Tensor
    entry_step: torch.Tensor
    actor_active: torch.Tensor
    critical_chunk_count: torch.Tensor
    expert_low_progress_count: torch.Tensor
    expert_latched: torch.Tensor
    route_expert_entered: torch.Tensor
    route_expert_entry_step: torch.Tensor
    chunk_index: torch.Tensor


@dataclass
class _GatePrediction:
    score_min: torch.Tensor
    phase_probability: torch.Tensor
    phase_features: torch.Tensor | None


class SteamCriticalPhaseGate:
    """Detect sustained low progress from raw, chunk-aligned frame pairs."""

    def __init__(self, model: Any, cfg: Any) -> None:
        self.model = model
        actor_cfg = cfg.get("actor_switch", {}) or {}
        self.actor_switch_enabled = bool(actor_cfg.get("enable", False))
        self.actor_mode = str(actor_cfg.get("mode", "active"))
        if self.actor_mode not in ("active", "shadow"):
            raise ValueError(
                "critical phase gate actor_switch.mode must be 'active' or 'shadow'"
            )
        self.chunk_size = int(cfg.get("chunk_size", 10))
        self.lookback_chunks = int(cfg.get("lookback_chunks", 3))
        configured_patience = actor_cfg.get("patience_chunks", None)
        configured_threshold = actor_cfg.get("enter_threshold", None)
        self.patience_chunks = int(
            2 if configured_patience is None else configured_patience
        )
        self.enter_threshold = float(
            0.0 if configured_threshold is None else configured_threshold
        )
        self.latch_until_done = bool(cfg.get("latch_until_done", True))
        self.emit_phase_features = bool(
            actor_cfg.get("collect_phase_features", False)
        )
        self.phase_head = None
        self.phase_head_metadata: dict[str, Any] = {}
        phase_head_path = actor_cfg.get("phase_head_path", None)
        if phase_head_path:
            if not os.path.exists(os.fspath(phase_head_path)):
                raise FileNotFoundError(
                    f"STEAM phase-head checkpoint not found: {phase_head_path}"
                )
            self.phase_head, self.phase_head_metadata = SteamPhaseHead.from_checkpoint(
                phase_head_path,
                device=self.device,
            )
            self.phase_head.eval()
            self.phase_head.requires_grad_(False)
        if self.actor_switch_enabled and self.phase_head is None:
            raise ValueError(
                "actor_switch.enable=True requires actor_switch.phase_head_path"
            )
        if self.phase_head is not None:
            if configured_threshold is None:
                self.enter_threshold = float(
                    self.phase_head_metadata.get(
                        "recommended_enter_threshold",
                        0.5,
                    )
                )
            if configured_patience is None:
                self.patience_chunks = int(
                    self.phase_head_metadata.get(
                        "recommended_patience_chunks",
                        1,
                    )
                )
        self.camera_mapping = {
            str(cfg.get("main_image_key", "image")): "main_images",
            str(cfg.get("wrist_image_key", "wrist_image")): "wrist_images",
        }

        expert_cfg = cfg.get("expert_takeover", {}) or {}
        self.expert_takeover_enabled = bool(expert_cfg.get("enable", False))
        expert_mode = expert_cfg.get("mode", "active")
        self.expert_mode = str(expert_mode)
        if self.expert_mode not in ("active", "shadow"):
            raise ValueError(
                "critical phase gate expert_takeover.mode must be 'active' or 'shadow'"
            )
        self.expert_enter_threshold = float(expert_cfg.get("enter_threshold", 0.0))
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

        processor = getattr(model, "processor", None)
        if processor is None:
            raise ValueError(
                "STEAM critical-phase gate requires a checkpoint-loaded processor"
            )
        model_config = getattr(model, "config", None)
        self._collator = BinaryPairDataCollator(
            processor=processor,
            max_length=int(getattr(model_config, "max_token_len", 200)),
            train=False,
            num_bins=int(getattr(model_config, "num_bins", 2)),
        )
        processor_keys = set(processor.image_processor.image_keys)
        missing_keys = processor_keys - set(self.camera_mapping)
        if missing_keys:
            raise ValueError(
                "STEAM checkpoint expects camera keys that are not mapped from "
                f"ManiSkill RLT observations: {sorted(missing_keys)}"
            )
        self._states: dict[tuple[str, int], _GateState] = {}

    @property
    def device(self) -> torch.device:
        """Return the device holding the STEAM model parameters."""
        return next(self.model.parameters()).device

    @property
    def controls_actor_routing(self) -> bool:
        """Return whether STEAM owns the base-to-actor routing decision."""
        return self.actor_switch_enabled and self.actor_mode == "active"

    def eval(self) -> None:
        self.model.eval()
        if self.phase_head is not None:
            self.phase_head.eval()

    def requires_grad_(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)
        if self.phase_head is not None:
            self.phase_head.requires_grad_(requires_grad)
        return self

    def to(self, device):
        self.model.to(device)
        if self.phase_head is not None:
            self.phase_head.to(device)
        state_tensor_fields = (
            "valid_count",
            "low_progress_count",
            "latched",
            "entered",
            "entry_step",
            "actor_active",
            "critical_chunk_count",
            "expert_low_progress_count",
            "expert_latched",
            "route_expert_entered",
            "route_expert_entry_step",
            "chunk_index",
        )
        for state in self._states.values():
            for field_name in state_tensor_fields:
                setattr(state, field_name, getattr(state, field_name).to(device))
        return self

    def empty_diagnostics(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a non-decision row for rollout bootstrap bookkeeping."""
        bool_keys = {
            "rlt_gate_entered",
            "rlt_gate_score_ready",
            "rlt_gate_actor_active",
            "rlt_route_expert_entered",
        }
        long_keys = {
            "rlt_gate_entry_step",
            "rlt_route_expert_entry_step",
        }
        diagnostics = {}
        for key in RLT_GATE_INFO_KEYS:
            dtype = torch.float32
            if key in bool_keys:
                dtype = torch.bool
            elif key in long_keys:
                dtype = torch.long
            diagnostics[key] = torch.zeros(
                (int(batch_size), 1),
                dtype=dtype,
                device=self.device,
            )
        return diagnostics

    def empty_phase_features(self, batch_size: int) -> torch.Tensor:
        """Return a stack-compatible placeholder for non-decision rows."""
        model_config = getattr(self.model, "config", None)
        feature_dim = int(getattr(model_config, "fusion_hidden_dim", 512)) * (
            int(getattr(model_config, "num_frames_per_pair", 2)) + 1
        )
        ensemble_size = int(getattr(model_config, "ensemble_size", 1))
        return torch.zeros(
            int(batch_size),
            ensemble_size,
            feature_dim,
            device=self.device,
            dtype=torch.float16,
        )

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

    @staticmethod
    def _batch_images(value: Any) -> np.ndarray:
        """Convert a batched ManiSkill image observation to uint8 BHWC."""
        if torch.is_tensor(value):
            value = value.detach().cpu()
            batch_size = int(value.shape[0])
            return np.stack([_to_uint8_hwc(value[idx]) for idx in range(batch_size)])
        array = np.asarray(value)
        if array.ndim != 4:
            raise ValueError(f"expected a batched rank-4 image, got {array.shape}")
        return np.stack([_to_uint8_hwc(array[idx]) for idx in range(array.shape[0])])

    def _extract_images(self, env_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        expected_keys = set(self._collator.processor.image_processor.image_keys)
        for steam_key, env_key in self.camera_mapping.items():
            if steam_key not in expected_keys:
                continue
            value = env_obs.get(env_key)
            if value is None:
                raise KeyError(
                    f"STEAM gate requires env observation {env_key!r} for "
                    f"checkpoint camera {steam_key!r}"
                )
            images[steam_key] = self._batch_images(value)
        return images

    @staticmethod
    def _extract_prompts(env_obs: dict[str, Any], batch_size: int) -> list[str]:
        prompts = env_obs.get("task_descriptions")
        if isinstance(prompts, str):
            return [prompts] * batch_size
        if prompts is None:
            raise KeyError("STEAM gate requires env observation 'task_descriptions'")
        if torch.is_tensor(prompts):
            prompts = prompts.detach().cpu().tolist()
        result = [str(prompt) for prompt in prompts]
        if len(result) != batch_size:
            raise ValueError(
                "task_descriptions batch size does not match images: "
                f"{len(result)} != {batch_size}"
            )
        return result

    def _new_state(self, images: dict[str, np.ndarray]) -> _GateState:
        first_images = next(iter(images.values()))
        batch_size = first_images.shape[0]
        history_len = self.lookback_chunks + 1
        image_history = {
            key: np.zeros(
                (batch_size, history_len, *value.shape[1:]),
                dtype=np.uint8,
            )
            for key, value in images.items()
        }
        device = self.device
        return _GateState(
            image_history=image_history,
            valid_count=torch.zeros(batch_size, device=device, dtype=torch.long),
            low_progress_count=torch.zeros(batch_size, device=device, dtype=torch.long),
            latched=torch.zeros(batch_size, device=device, dtype=torch.bool),
            entered=torch.zeros(batch_size, device=device, dtype=torch.bool),
            entry_step=torch.zeros(batch_size, device=device, dtype=torch.long),
            actor_active=torch.zeros(batch_size, device=device, dtype=torch.bool),
            critical_chunk_count=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
            expert_low_progress_count=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
            expert_latched=torch.zeros(batch_size, device=device, dtype=torch.bool),
            route_expert_entered=torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
            route_expert_entry_step=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
            chunk_index=torch.zeros(batch_size, device=device, dtype=torch.long),
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

    @staticmethod
    def _normalize_actor_switch(
        actor_switch: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if actor_switch is None:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        switch = torch.as_tensor(actor_switch, device=device, dtype=torch.bool)
        if switch.numel() % batch_size != 0:
            raise ValueError(
                "external actor switch size does not match STEAM gate batch: "
                f"{switch.numel()} values for batch size {batch_size}"
            )
        return switch.reshape(batch_size, -1)[:, -1]

    @staticmethod
    def _to_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {
                key: SteamCriticalPhaseGate._to_device(child, device)
                for key, child in value.items()
            }
        return value

    def _predict_pair(
        self,
        state: _GateState,
        prompts: list[str],
    ) -> _GatePrediction:
        samples = []
        batch_size = state.valid_count.shape[0]
        for batch_idx in range(batch_size):
            image_t = {
                key: history[batch_idx, 0]
                for key, history in state.image_history.items()
            }
            image_tk = {
                key: history[batch_idx, -1]
                for key, history in state.image_history.items()
            }
            masks = dict.fromkeys(image_t, True)
            samples.append(
                {
                    "image_t": image_t,
                    "image_tk": image_tk,
                    "image_mask_t": masks,
                    "image_mask_tk": masks,
                    "prompt": prompts[batch_idx],
                }
            )
        observation = self._collator.collate_observations(samples)
        observation = self._to_device(observation, self.device)
        need_phase_features = self.emit_phase_features or self.phase_head is not None
        members = getattr(self.model, "members", None)
        phase_features = None
        if need_phase_features and members is not None:
            member_outputs = [member.predict(observation) for member in members]
            member_scores = torch.stack(
                [output.predicted_values for output in member_outputs],
                dim=0,
            ).to(dtype=torch.float32)
            score_min = member_scores.min(dim=0).values
            phase_features = torch.stack(
                [output.hidden_states for output in member_outputs],
                dim=0,
            )
        else:
            output = self.model.predict(observation)
            score_min = output.predicted_values.to(dtype=torch.float32)
            if need_phase_features:
                hidden_states = getattr(output, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "STEAM phase-head collection requires fused hidden_states"
                    )
                phase_features = hidden_states.unsqueeze(0)

        phase_probability = torch.zeros_like(score_min)
        if self.phase_head is not None:
            if phase_features is None:
                raise RuntimeError("STEAM phase head requires fused features")
            phase_probability = self.phase_head.predict(phase_features).to(
                torch.float32
            )
        return _GatePrediction(
            score_min=score_min,
            phase_probability=phase_probability,
            phase_features=phase_features,
        )

    @torch.no_grad()
    def step(
        self,
        env_obs: dict[str, Any],
        *,
        mode: Literal["train", "eval"],
        stage_id: int,
        reset_mask: torch.Tensor | None = None,
        external_actor_switch: torch.Tensor | None = None,
        actor_routing_enabled: bool = True,
        expert_routing_enabled: bool = True,
    ) -> GateDecision:
        images = self._extract_images(env_obs)
        batch_size = next(iter(images.values())).shape[0]
        prompts = self._extract_prompts(env_obs, batch_size)
        key = (mode, int(stage_id))
        state = self._states.get(key)
        if state is None or state.valid_count.shape[0] != batch_size:
            state = self._new_state(images)
            self._states[key] = state

        reset = self._normalize_reset_mask(
            reset_mask,
            batch_size=batch_size,
            device=self.device,
        )
        if reset.any():
            reset_cpu = reset.detach().cpu().numpy()
            for history in state.image_history.values():
                history[reset_cpu] = 0
            state.valid_count[reset] = 0
            state.low_progress_count[reset] = 0
            state.latched[reset] = False
            state.entered[reset] = False
            state.entry_step[reset] = 0
            state.actor_active[reset] = False
            state.critical_chunk_count[reset] = 0
            state.expert_low_progress_count[reset] = 0
            state.expert_latched[reset] = False
            state.route_expert_entered[reset] = False
            state.route_expert_entry_step[reset] = 0
            state.chunk_index[reset] = 0

        for image_key, current_images in images.items():
            state.image_history[image_key] = np.roll(
                state.image_history[image_key],
                shift=-1,
                axis=1,
            )
            state.image_history[image_key][:, -1] = current_images
        state.valid_count = torch.clamp(
            state.valid_count + 1,
            max=self.lookback_chunks + 1,
        )
        ready = state.valid_count >= self.lookback_chunks + 1

        if ready.any():
            prediction = self._predict_pair(state, prompts)
            score = torch.where(
                ready,
                prediction.score_min,
                torch.zeros_like(prediction.score_min),
            )
            phase_probability = torch.where(
                ready,
                prediction.phase_probability,
                torch.zeros_like(prediction.phase_probability),
            )
            phase_features = prediction.phase_features
        else:
            score = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            phase_probability = torch.zeros_like(score)
            phase_features = None

        actor_candidate = (
            ready & (phase_probability >= self.enter_threshold)
            if self.actor_switch_enabled
            else torch.zeros_like(ready)
        )
        state.low_progress_count = torch.where(
            actor_candidate,
            state.low_progress_count + 1,
            torch.zeros_like(state.low_progress_count),
        )
        enter_now = (~state.latched) & (
            state.low_progress_count >= self.patience_chunks
        )
        first_enter_now = enter_now & (~state.entered)
        state.entry_step = torch.where(
            first_enter_now,
            state.chunk_index * self.chunk_size,
            state.entry_step,
        )
        state.entered = state.entered | enter_now
        if self.latch_until_done:
            state.latched = state.latched | enter_now
        else:
            state.latched = (state.latched | enter_now) & actor_candidate

        steam_critical_active = state.latched
        if self.controls_actor_routing:
            critical_phase_active = steam_critical_active
        else:
            critical_phase_active = self._normalize_actor_switch(
                external_actor_switch,
                batch_size=batch_size,
                device=self.device,
            )
        actor_active = critical_phase_active & bool(actor_routing_enabled)
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
        if self.expert_latch_until_done:
            state.expert_latched = state.expert_latched | expert_enter_now
        else:
            state.expert_latched = (
                state.expert_latched | expert_enter_now
            ) & expert_low_progress
        # Preserve the critical-phase signal during learner warmup so those
        # chunks enter replay. SimulatorRLTRoute independently prevents actor
        # execution until the warmup updates are complete.
        route_flags = critical_phase_active
        route_expert_flags = state.expert_latched & actor_active
        if self.expert_mode == "shadow":
            route_expert_flags = torch.zeros_like(route_expert_flags)

        route_expert_active = route_expert_flags & bool(expert_routing_enabled)
        route_expert_started_now = route_expert_active & (~state.route_expert_entered)
        state.route_expert_entry_step = torch.where(
            route_expert_started_now,
            state.chunk_index * self.chunk_size,
            state.route_expert_entry_step,
        )
        state.route_expert_entered = state.route_expert_entered | route_expert_active

        diagnostics = {
            "rlt_gate_entered": state.entered[:, None],
            "rlt_gate_entry_step": state.entry_step[:, None],
            "rlt_gate_score_ready": ready[:, None],
            "rlt_gate_score_min": score[:, None],
            "rlt_gate_actor_active": actor_active[:, None],
            "rlt_route_expert_entered": state.route_expert_entered[:, None],
            "rlt_route_expert_entry_step": state.route_expert_entry_step[:, None],
        }
        state.chunk_index = state.chunk_index + 1
        emitted_phase_features = None
        if self.emit_phase_features:
            if phase_features is None:
                emitted_phase_features = self.empty_phase_features(batch_size)
            else:
                emitted_phase_features = (
                    phase_features.permute(1, 0, 2).detach().to(torch.float16)
                )
        return GateDecision(
            actor_switch=route_flags[:, None],
            expert_requested=route_expert_flags[:, None],
            diagnostics=diagnostics,
            phase_features=emitted_phase_features,
        )


def build_rlt_critical_phase_gate(
    cfg: Any,
    *,
    device: str,
    num_action_chunks: int,
    env_decoupled_mode: bool,
) -> SteamCriticalPhaseGate | None:
    """Build a checkpoint-backed gate when it is enabled in rollout config."""
    if cfg is None or not bool(cfg.get("enable", False)):
        return None
    if env_decoupled_mode:
        raise ValueError(
            "The stateful RLT critical phase gate does not support "
            "runner.enable_decoupled_mode."
        )
    model_cfg = cfg.get("model", None)
    if model_cfg is None:
        raise ValueError("rollout.rlt_critical_phase_gate.model is required.")
    model_path = model_cfg.get("model_path", None)
    if not model_path or not os.path.exists(os.fspath(model_path)):
        raise FileNotFoundError(
            "RLT critical phase gate requires a trained local critic "
            f"checkpoint, got {model_path!r}."
        )
    chunk_size = int(cfg.get("chunk_size", 10))
    if chunk_size != int(num_action_chunks):
        raise ValueError(
            "RLT critical phase gate chunk_size must match rollout action "
            f"chunks: {chunk_size} != {num_action_chunks}."
        )

    from rlinf.models.embodiment.value_model.steam import SteamCriticModel

    model = SteamCriticModel.from_checkpoint(
        model_path,
        device=device,
        precision=model_cfg.get("precision", None),
    )
    gate = SteamCriticalPhaseGate(model, cfg)
    gate.eval()
    gate.requires_grad_(False)
    return gate


__all__ = [
    "GateDecision",
    "RLT_GATE_INFO_KEYS",
    "SteamCriticalPhaseGate",
    "build_rlt_critical_phase_gate",
]
