#!/usr/bin/env python3
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

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.rocm_patches import patch_vision_patch_embed
from rlinf.models.embodiment.reward.vlm_reward_utils.common import (
    apply_gt_success_bonus,
    load_vlm_processor,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    BufferedVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.lora import load_lora_adapter
from rlinf.models.embodiment.reward.vlm_reward_utils.potential import (
    ScalarPotentialHead,
    extract_prompt_features,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    BaseRewardParser,
    get_reward_parser,
)


class VLMRewardModel(BaseRewardModel):
    """A frozen VLM reward model that maps (images, task) -> scalar reward.

    This implementation intentionally avoids hardcoding family-specific HF class
    names. It loads by `model_path` via Auto* APIs (consistent with RLinf SFT).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.model_path: str = cfg.get("model_path")
        if not self.model_path:
            raise ValueError("reward.model.model_path must be set for VLMRewardModel")
        self.lora_path = self.cfg.get("lora_path")
        self.gt_success_bonus = float(cfg.get("gt_success_bonus", 0.0))

        self.dtype = torch_dtype_from_precision(cfg.precision)

        self.setup_processor()
        self.setup_model()

        self.setup_input_builder()
        self.setup_reward_parser()

        self.gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 32)),
            "do_sample": bool(cfg.get("do_sample", True)),
            "temperature": float(cfg.get("temperature", 0.0)),
        }

    def setup_processor(self) -> None:
        self._processor = load_vlm_processor(
            self.model_path, self.cfg.get("subprocessor_kwargs", {})
        )

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "base_vlm_input_builder")
        )(**self.cfg.get("input_builder_params", {}), _processor=self._processor)

    def setup_reward_parser(self) -> None:
        self.reward_parser = get_reward_parser(
            self.cfg.get("reward_parser_name", "base_reward_parser")
        )(**self.cfg.get("reward_parser_params", {}))

    def apply_gt_success_bonus(
        self, rewards: torch.Tensor, reward_input: dict[str, Any]
    ) -> torch.Tensor:
        return apply_gt_success_bonus(rewards, reward_input, self.gt_success_bonus)

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "VLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
        )

    def _generate_and_parse_rewards(
        self, batched_inputs: dict[str, Any]
    ) -> torch.Tensor:
        """Run model.generate and parse decoded outputs into rewards."""
        prompt_length = batched_inputs["input_ids"].shape[-1]
        output_ids = self._model.generate(**batched_inputs, **self.gen_kwargs)
        outputs = self._processor.batch_decode(
            output_ids[..., prompt_length:], skip_special_tokens=True
        )
        rewards = self.reward_parser.parse_rewards(outputs)

        return rewards

    def setup_model(self) -> None:
        # AutoModelForVision2Seq was renamed to AutoModelForImageTextToText
        # in transformers >= 5.0
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import (
                AutoModelForImageTextToText as AutoModelForVision2Seq,
            )

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        patch_vision_patch_embed(self._model)

        if self.lora_path:
            full_weights_path = os.path.join(
                self.lora_path, "actor", "model_state_dict", "full_weights.pt"
            )

            checkpoint_state_dict = torch.load(
                full_weights_path,
                map_location="cpu",
                weights_only=True,
            )
            lora_state_dict = {
                key.removeprefix("module."): value
                for key, value in checkpoint_state_dict.items()
                if "lora_" in key
            }
            if lora_state_dict:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    set_peft_model_state_dict,
                )

                lora_rank = next(
                    int(value.shape[0])
                    for key, value in lora_state_dict.items()
                    if "lora_A" in key
                )
                target_modules = sorted(
                    {
                        key.split(".lora_")[0].split(".")[-1]
                        for key in lora_state_dict
                        if ".lora_" in key
                    }
                )

                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    lora_dropout=0.0,
                    target_modules=target_modules,
                    init_lora_weights="gaussian",
                )
                self._model = get_peft_model(self._model, lora_config)
                set_peft_model_state_dict(self._model, lora_state_dict)
                del lora_state_dict
                del checkpoint_state_dict
            else:
                checkpoint_state_dict = {
                    key.removeprefix("module."): value
                    for key, value in checkpoint_state_dict.items()
                }
                self._model.load_state_dict(checkpoint_state_dict, strict=False)
                del checkpoint_state_dict

        self._model.eval()

    @torch.no_grad()
    def compute_reward(
        self,
        observations: Any,
    ) -> torch.Tensor:
        batched_inputs = self.input_builder.build_inputs(
            observations, self._model.device
        )
        rewards = self._generate_and_parse_rewards(batched_inputs)
        del batched_inputs
        return self.apply_gt_success_bonus(rewards, observations)


class BufferedVLMRewardModel(VLMRewardModel):
    def __init__(self, cfg: DictConfig):
        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(cfg.get("infer_micro_batch_size", 0))
        self.interval_reward: float = float(cfg.get("interval_reward", 0.0))

        super().__init__(cfg)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "buffered_vlm_input_builder")
        )(
            **self.cfg.get("input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        assert isinstance(self.input_builder, BufferedVLMInputBuilder), (
            "BufferedVLMRewardModel only supports BufferedVLMInputBuilder"
        )

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "BufferedVLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
        )

    def slice_history_input(
        self,
        history_input: dict[str, dict[str, list[list[Any]]]],
        start: int,
        end: int,
    ) -> dict[str, dict[str, list[list[Any]]]]:
        return {
            buffer_name: {
                history_key: env_sequences[start:end]
                for history_key, env_sequences in history_buffer.items()
            }
            for buffer_name, history_buffer in history_input.items()
        }

    def slice_observations(
        self,
        observations: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, Any]:
        return {
            key: self._slice_batch_value(value, start, end)
            for key, value in observations.items()
        }

    def _slice_batch_value(self, value: Any, start: int, end: int) -> Any:
        if isinstance(value, dict):
            return {
                key: self._slice_batch_value(item, start, end)
                for key, item in value.items()
            }
        if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
            return value[start:end]
        return value

    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        history_input: dict[str, dict[str, list[list[Any]]]] = reward_input[
            "history_input"
        ]
        input_batch_size = len(next(iter(next(iter(history_input.values())).values())))
        observations = {
            key: value for key, value in reward_input.items() if key != "history_input"
        }

        infer_micro_batch_size = self.infer_micro_batch_size or input_batch_size

        reward_chunks: list[torch.Tensor] = []
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            micro_observations = self.slice_observations(observations, start, end)
            micro_history_input = self.slice_history_input(history_input, start, end)
            reward_chunk = torch.full(
                (end - start,), fill_value=self.interval_reward, dtype=torch.float32
            )

            batched_inputs, valid_input_ids = self.input_builder.build_inputs(
                micro_observations,
                self._model.device,
                micro_history_input,
            )
            if len(valid_input_ids) == 0:
                reward_chunks.append(reward_chunk)
                continue

            parsed_rewards = self._generate_and_parse_rewards(batched_inputs)
            del batched_inputs
            reward_chunk[valid_input_ids] = parsed_rewards.to(dtype=torch.float32)
            reward_chunks.append(reward_chunk)

        rewards = torch.cat(reward_chunks, dim=0)
        return self.apply_gt_success_bonus(rewards, observations)


class ShapedVLMRewardModel(BufferedVLMRewardModel):
    """Combine scalar potential shaping with a one-shot VLM success bonus."""

    def __init__(self, cfg: DictConfig) -> None:
        for field_name in ("lora_path", "scalar_head_path", "success_lora_path"):
            if not cfg.get(field_name):
                raise ValueError(
                    f"reward.model.{field_name} must be set for ShapedVLMRewardModel"
                )

        self.success_lora_path = str(cfg.success_lora_path)
        self.scalar_head_path = str(cfg.scalar_head_path)
        self.potential_gamma = float(cfg.get("potential_gamma", 1.0))
        self.potential_scale = float(cfg.get("potential_scale", 1.0))
        self.potential_ema_alpha = float(cfg.get("potential_ema_alpha", 0.2))
        self.potential_clip = float(cfg.get("potential_clip", 0.0))
        self.success_threshold = float(cfg.get("success_threshold", 0.95))
        self.success_bonus = float(cfg.get("success_bonus", 0.0))
        self.success_confirmation_windows = int(
            cfg.get("success_confirmation_windows", 1)
        )
        if self.success_confirmation_windows < 1:
            raise ValueError("success_confirmation_windows must be positive")

        self._previous_potentials: torch.Tensor | None = None
        self._success_fired: torch.Tensor | None = None
        self._success_streak: torch.Tensor | None = None
        super().__init__(cfg)

        self.setup_scalar_head()
        self.success_input_builder = get_input_builder(
            self.cfg.get(
                "success_input_builder_name",
                "vlm_trend_success_potential_input_builder",
            )
        )(
            **self.cfg.get("success_input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        if not isinstance(self.success_input_builder, BufferedVLMInputBuilder):
            raise TypeError("success_input_builder must be a BufferedVLMInputBuilder")
        self.success_reward_parser: BaseRewardParser = get_reward_parser(
            self.cfg.get(
                "success_reward_parser_name",
                "vlm_trend_binary_digit_reward_parser",
            )
        )(**self.cfg.get("success_reward_parser_params", {}))
        self.success_gen_kwargs = {
            "max_new_tokens": int(cfg.get("success_max_new_tokens", 3)),
            "do_sample": False,
            "temperature": 0.0,
        }

    def setup_model(self) -> None:
        """Load the frozen base VLM and both named LoRA adapters."""
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import (
                AutoModelForImageTextToText as AutoModelForVision2Seq,
            )

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        patch_vision_patch_embed(self._model)
        self._model = load_lora_adapter(
            self._model, str(self.lora_path), adapter_name="default"
        )
        self._model = load_lora_adapter(
            self._model, self.success_lora_path, adapter_name="success"
        )
        self._model.eval()

    def setup_scalar_head(self) -> None:
        """Load the scalar potential head checkpoint."""
        payload = torch.load(
            self.scalar_head_path, map_location="cpu", weights_only=True
        )
        config = payload["config"]
        self.scalar_head = ScalarPotentialHead(
            int(config["input_dim"]),
            int(config["hidden_dim"]),
            float(config["dropout"]),
        )
        self.scalar_head.load_state_dict(payload["model_state_dict"])
        self.scalar_head.to(device=self._model.device, dtype=torch.float32)
        self.scalar_head.eval()

    @torch.no_grad()
    def compute_scalar_potential(
        self, batched_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return sigmoid-bounded potential values."""
        features = extract_prompt_features(self._model, batched_inputs)
        return torch.sigmoid(self.scalar_head(features))

    def _history_input_batch_size(
        self,
        history_input: dict[str, dict[str, list[list[Any]]]],
        observations: dict[str, Any],
    ) -> int:
        for history_buffer in history_input.values():
            for histories in history_buffer.values():
                return len(histories)
        for value in observations.values():
            if isinstance(value, dict):
                try:
                    return self._history_input_batch_size({}, value)
                except ValueError:
                    continue
            if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
                return len(value)
        raise ValueError("Unable to infer reward input batch size")

    def _score_micro_batch(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        potentials = torch.zeros(batch_size, dtype=torch.float32)
        success_scores = torch.zeros(batch_size, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)

        potential_inputs, valid_ids = self.input_builder.build_inputs(
            observations,
            self._model.device,
            history_input,
        )
        if not valid_ids:
            return potentials, valid_mask, success_scores

        potentials[valid_ids] = self.compute_scalar_potential(potential_inputs).cpu()
        valid_mask[valid_ids] = True
        del potential_inputs

        success_inputs, success_ids = self.success_input_builder.build_inputs(
            observations,
            self._model.device,
            history_input,
        )
        if success_ids:
            self._model.set_adapter("success")
            try:
                prompt_length = success_inputs["input_ids"].shape[-1]
                output_ids = self._model.generate(
                    **success_inputs,
                    **self.success_gen_kwargs,
                )
                outputs = self._processor.batch_decode(
                    output_ids[..., prompt_length:],
                    skip_special_tokens=True,
                )
                success_scores[success_ids] = self.success_reward_parser.parse_rewards(
                    outputs
                )
            finally:
                self._model.set_adapter("default")
            del success_inputs
        return potentials, valid_mask, success_scores

    @torch.no_grad()
    def compute_reward(self, reward_input: dict[str, Any]) -> torch.Tensor:
        """Compute potential shaping and sparse success rewards."""
        history_input = reward_input["history_input"]
        observations = {
            key: value for key, value in reward_input.items() if key != "history_input"
        }
        batch_size = self._history_input_batch_size(history_input, observations)
        if any(history_buffer for history_buffer in history_input.values()):
            micro_batch_size = self.infer_micro_batch_size or batch_size
            potential_chunks = []
            valid_chunks = []
            success_chunks = []
            for start in range(0, batch_size, micro_batch_size):
                end = min(start + micro_batch_size, batch_size)
                potentials, valid_mask, success_scores = self._score_micro_batch(
                    self.slice_observations(observations, start, end),
                    self.slice_history_input(history_input, start, end),
                    end - start,
                )
                potential_chunks.append(potentials)
                valid_chunks.append(valid_mask)
                success_chunks.append(success_scores)
            potentials = torch.cat(potential_chunks)
            valid_mask = torch.cat(valid_chunks)
            success_scores = torch.cat(success_chunks)
        else:
            # HistoryManager emits {} between interval ticks. Skip scoring, but
            # still apply dones so episode-local shaping state does not leak.
            potentials = torch.zeros(batch_size, dtype=torch.float32)
            valid_mask = torch.zeros(batch_size, dtype=torch.bool)
            success_scores = torch.zeros(batch_size, dtype=torch.float32)

        dones = observations.get("dones")
        rewards = self.potential_differences(potentials, valid_mask, dones)
        if self.success_bonus != 0.0:
            rewards = self.apply_model_success_bonus(
                rewards,
                success_scores,
                valid_mask,
                dones,
            )
        return self.apply_gt_success_bonus(rewards, observations)

    def apply_model_success_bonus(
        self,
        rewards: torch.Tensor,
        success_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        dones: Any = None,
    ) -> torch.Tensor:
        """Add a one-shot success bonus and reset state at episode end."""
        if self._success_fired is None or self._success_fired.shape != rewards.shape:
            self._success_fired = torch.zeros_like(rewards, dtype=torch.bool)
            self._success_streak = torch.zeros_like(rewards, dtype=torch.int32)
        if self._success_streak is None:
            raise RuntimeError("success streak state was not initialized")

        above_threshold = valid_mask & (success_scores >= self.success_threshold)
        self._success_streak[valid_mask & ~above_threshold] = 0
        self._success_streak[above_threshold] += 1
        triggered = (
            valid_mask
            & ~self._success_fired
            & (self._success_streak >= self.success_confirmation_windows)
        )
        rewards = rewards + triggered.to(rewards.dtype) * self.success_bonus
        self._success_fired |= triggered
        if dones is not None:
            done_mask = torch.as_tensor(dones).reshape(-1).bool().cpu()
            if done_mask.shape == self._success_fired.shape:
                self._success_fired[done_mask] = False
                self._success_streak[done_mask] = 0
        return rewards

    def potential_differences(
        self,
        potentials: torch.Tensor,
        valid_mask: torch.Tensor,
        dones: Any = None,
    ) -> torch.Tensor:
        """Convert absolute potentials to episode-local shaping rewards."""
        if (
            self._previous_potentials is None
            or self._previous_potentials.shape != potentials.shape
        ):
            self._previous_potentials = torch.full_like(potentials, torch.nan)
        previous = self._previous_potentials
        initialized = valid_mask & torch.isfinite(previous)
        rewards = torch.zeros_like(potentials)
        smoothed = potentials.clone()
        smoothed[initialized] = (
            self.potential_ema_alpha * potentials[initialized]
            + (1.0 - self.potential_ema_alpha) * previous[initialized]
        )
        rewards[initialized] = self.potential_scale * (
            self.potential_gamma * smoothed[initialized] - previous[initialized]
        )
        if self.potential_clip > 0.0:
            rewards.clamp_(-self.potential_clip, self.potential_clip)
        previous[valid_mask] = smoothed[valid_mask]

        if dones is not None:
            done_mask = torch.as_tensor(dones).reshape(-1).bool().cpu()
            if done_mask.shape == previous.shape:
                previous[done_mask] = torch.nan
        return rewards
