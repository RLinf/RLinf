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

from rlinf.data.datasets.steam import BinaryPairDataCollator
from rlinf.data.datasets.steam.pair_dataset import _to_uint8_hwc

RLT_GATE_INFO_KEYS = (
    "rlt_gate_entered",
    "rlt_gate_entry_step",
    "rlt_gate_score_ready",
    "rlt_gate_score_min",
    "rlt_gate_score_mean",
    "rlt_gate_prediction_variance",
    "rlt_gate_steam_critical_active",
    "rlt_gate_actor_active",
    "rlt_route_base_active",
    "rlt_route_actor_active",
    "rlt_route_actor_entered",
    "rlt_route_actor_entry_step",
    "rlt_route_expert_active",
    "rlt_route_expert_entered",
    "rlt_route_expert_entry_step",
    "rlt_gate_chunk_index",
    "rlt_gate_critical_chunk_count",
    "rlt_gate_expert_candidate",
    "rlt_gate_expert_active",
    "rlt_gate_expert_requested",
    "rlt_gate_expert_entered",
    "rlt_gate_expert_entry_step",
)


@dataclass
class GateDecision:
    """Routing decision and diagnostics emitted by one gate step."""

    actor_switch: torch.Tensor
    expert_requested: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class _GateState:
    image_history: dict[str, np.ndarray]
    valid_count: torch.Tensor
    low_progress_count: torch.Tensor
    latched: torch.Tensor
    entered: torch.Tensor
    entry_step: torch.Tensor
    actor_active: torch.Tensor
    route_actor_entered: torch.Tensor
    route_actor_entry_step: torch.Tensor
    critical_chunk_count: torch.Tensor
    expert_low_progress_count: torch.Tensor
    expert_latched: torch.Tensor
    expert_entered: torch.Tensor
    expert_entry_step: torch.Tensor
    route_expert_entered: torch.Tensor
    route_expert_entry_step: torch.Tensor
    chunk_index: torch.Tensor


@dataclass
class _GatePrediction:
    score_min: torch.Tensor
    score_mean: torch.Tensor
    prediction_variance: torch.Tensor


class SteamCriticalPhaseGate:
    """Detect sustained low progress from raw, chunk-aligned frame pairs."""

    def __init__(self, model: Any, cfg: Any) -> None:
        self.model = model
        actor_cfg = cfg.get("actor_switch", None)
        if actor_cfg is None:
            self.actor_switch_enabled = True
            self.actor_mode = str(cfg.get("mode", "active"))
        else:
            actor_cfg = actor_cfg or {}
            self.actor_switch_enabled = bool(actor_cfg.get("enable", True))
            self.actor_mode = str(actor_cfg.get("mode", cfg.get("mode", "active")))
        if self.actor_mode not in ("active", "shadow"):
            raise ValueError(
                "critical phase gate actor_switch.mode must be 'active' or 'shadow'"
            )
        # Keep the legacy attribute for callers inspecting an old-style config.
        self.mode = self.actor_mode
        self.chunk_size = int(cfg.get("chunk_size", 10))
        self.lookback_chunks = int(cfg.get("lookback_chunks", 3))
        self.patience_chunks = int(cfg.get("patience_chunks", 2))
        self.enter_threshold = float(cfg.get("enter_threshold", 0.0))
        self.latch_until_done = bool(cfg.get("latch_until_done", True))
        self.camera_mapping = {
            str(cfg.get("main_image_key", "image")): "main_images",
            str(cfg.get("wrist_image_key", "wrist_image")): "wrist_images",
        }

        expert_cfg = cfg.get("expert_takeover", {}) or {}
        self.expert_takeover_enabled = bool(expert_cfg.get("enable", False))
        expert_mode = expert_cfg.get("mode", "active")
        if actor_cfg is None and self.actor_mode == "shadow":
            expert_mode = "shadow"
        self.expert_mode = str(expert_mode)
        if self.expert_mode not in ("active", "shadow"):
            raise ValueError(
                "critical phase gate expert_takeover.mode must be 'active' or 'shadow'"
            )
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

    def requires_grad_(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)
        return self

    def to(self, device):
        self.model.to(device)
        self._states.clear()
        return self

    def empty_diagnostics(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a non-decision row for rollout bootstrap bookkeeping."""
        bool_keys = {
            "rlt_gate_entered",
            "rlt_gate_score_ready",
            "rlt_gate_steam_critical_active",
            "rlt_gate_actor_active",
            "rlt_route_base_active",
            "rlt_route_actor_active",
            "rlt_route_actor_entered",
            "rlt_route_expert_active",
            "rlt_route_expert_entered",
            "rlt_gate_expert_candidate",
            "rlt_gate_expert_active",
            "rlt_gate_expert_requested",
            "rlt_gate_expert_entered",
        }
        long_keys = {
            "rlt_gate_entry_step",
            "rlt_route_actor_entry_step",
            "rlt_gate_chunk_index",
            "rlt_gate_critical_chunk_count",
            "rlt_gate_expert_entry_step",
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
            route_actor_entered=torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
            route_actor_entry_step=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
            critical_chunk_count=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
            expert_low_progress_count=torch.zeros(
                batch_size, device=device, dtype=torch.long
            ),
            expert_latched=torch.zeros(batch_size, device=device, dtype=torch.bool),
            expert_entered=torch.zeros(batch_size, device=device, dtype=torch.bool),
            expert_entry_step=torch.zeros(batch_size, device=device, dtype=torch.long),
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
        output = self.model.predict(observation)
        score_min = output.predicted_values.to(dtype=torch.float32)
        score_mean = getattr(output, "prediction_mean", None)
        if score_mean is None:
            score_mean = score_min
        else:
            score_mean = score_mean.to(dtype=torch.float32)
        prediction_variance = getattr(output, "prediction_variance", None)
        if prediction_variance is None:
            prediction_variance = torch.zeros_like(score_min)
        else:
            prediction_variance = prediction_variance.to(dtype=torch.float32)
        return _GatePrediction(
            score_min=score_min,
            score_mean=score_mean,
            prediction_variance=prediction_variance,
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
            state.route_actor_entered[reset] = False
            state.route_actor_entry_step[reset] = 0
            state.critical_chunk_count[reset] = 0
            state.expert_low_progress_count[reset] = 0
            state.expert_latched[reset] = False
            state.expert_entered[reset] = False
            state.expert_entry_step[reset] = 0
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
            score_mean = torch.where(
                ready,
                prediction.score_mean,
                torch.zeros_like(prediction.score_mean),
            )
            prediction_variance = torch.where(
                ready,
                prediction.prediction_variance,
                torch.zeros_like(prediction.prediction_variance),
            )
        else:
            score = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            score_mean = torch.zeros_like(score)
            prediction_variance = torch.zeros_like(score)

        low_progress = ready & (score <= self.enter_threshold)
        state.low_progress_count = torch.where(
            low_progress,
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
            state.latched = (state.latched | enter_now) & low_progress

        steam_critical_active = state.latched
        if self.controls_actor_routing:
            actor_active = steam_critical_active
        else:
            actor_active = self._normalize_actor_switch(
                external_actor_switch,
                batch_size=batch_size,
                device=self.device,
            )
        actor_active = actor_active & bool(actor_routing_enabled)
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
        first_expert_enter_now = expert_enter_now & (~state.expert_entered)
        state.expert_entry_step = torch.where(
            first_expert_enter_now,
            state.chunk_index * self.chunk_size,
            state.expert_entry_step,
        )
        state.expert_entered = state.expert_entered | expert_enter_now
        if self.expert_latch_until_done:
            state.expert_latched = state.expert_latched | expert_enter_now
        else:
            state.expert_latched = (
                state.expert_latched | expert_enter_now
            ) & expert_low_progress
        route_flags = actor_active
        route_expert_flags = state.expert_latched & actor_active
        if self.expert_mode == "shadow":
            route_expert_flags = torch.zeros_like(route_expert_flags)

        route_expert_active = route_expert_flags & bool(expert_routing_enabled)
        route_actor_active = actor_active & (~route_expert_active)
        route_actor_started_now = route_actor_active & (~state.route_actor_entered)
        state.route_actor_entry_step = torch.where(
            route_actor_started_now,
            state.chunk_index * self.chunk_size,
            state.route_actor_entry_step,
        )
        state.route_actor_entered = state.route_actor_entered | route_actor_active
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
            "rlt_gate_score_mean": score_mean[:, None],
            "rlt_gate_prediction_variance": prediction_variance[:, None],
            "rlt_gate_steam_critical_active": steam_critical_active[:, None],
            "rlt_gate_actor_active": actor_active[:, None],
            "rlt_route_base_active": (~(route_actor_active | route_expert_active))[
                :, None
            ],
            "rlt_route_actor_active": route_actor_active[:, None],
            "rlt_route_actor_entered": state.route_actor_entered[:, None],
            "rlt_route_actor_entry_step": state.route_actor_entry_step[:, None],
            "rlt_route_expert_active": route_expert_active[:, None],
            "rlt_route_expert_entered": state.route_expert_entered[:, None],
            "rlt_route_expert_entry_step": state.route_expert_entry_step[:, None],
            "rlt_gate_chunk_index": state.chunk_index[:, None],
            "rlt_gate_critical_chunk_count": state.critical_chunk_count[:, None],
            "rlt_gate_expert_candidate": expert_low_progress[:, None],
            "rlt_gate_expert_active": state.expert_latched[:, None],
            "rlt_gate_expert_requested": route_expert_flags[:, None],
            "rlt_gate_expert_entered": state.expert_entered[:, None],
            "rlt_gate_expert_entry_step": state.expert_entry_step[:, None],
        }
        state.chunk_index = state.chunk_index + 1
        return GateDecision(
            actor_switch=route_flags[:, None],
            expert_requested=route_expert_flags[:, None],
            diagnostics=diagnostics,
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
