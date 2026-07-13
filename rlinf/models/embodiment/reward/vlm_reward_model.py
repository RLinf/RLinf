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
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import AutoModelForVision2Seq, AutoProcessor

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)


class ScalarPotentialHead(torch.nn.Module):
    """Small scalar head trained on the final Qwen prompt representation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


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
        self.success_lora_path = self.cfg.get("success_lora_path")
        self.gt_success_bonus = float(cfg.get("gt_success_bonus", 0.0))
        self.inference_mode = str(cfg.get("inference_mode", "generate"))
        self.scalar_head_path = cfg.get("scalar_head_path")
        self.success_threshold = float(cfg.get("success_threshold", 0.95))
        self.success_bonus = float(cfg.get("success_bonus", 0.0))
        self.success_confirmation_windows = int(
            cfg.get("success_confirmation_windows", 1)
        )
        if self.success_confirmation_windows < 1:
            raise ValueError("success_confirmation_windows must be positive")

        self.dtype = torch_dtype_from_precision(cfg.precision)

        self.setup_processor()
        self.setup_model()
        self.setup_scalar_head()

        self.setup_input_builder()
        self.setup_reward_parser()

        self.gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 32)),
            "do_sample": bool(cfg.get("do_sample", True)),
            "temperature": float(cfg.get("temperature", 0.0)),
        }

    def setup_processor(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        subprocessor_kwargs = self.cfg.get("subprocessor_kwargs", {})
        for subprocessor_name, subprocessor_cfg in subprocessor_kwargs.items():
            subprocessor_cfg = dict(subprocessor_cfg)
            subprocessor = getattr(self._processor, subprocessor_name, None)
            if subprocessor is None:
                continue
            for key, value in subprocessor_cfg.items():
                if hasattr(subprocessor, key):
                    setattr(subprocessor, key, value)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "base_vlm_input_builder")
        )(**self.cfg.get("input_builder_params", {}), _processor=self._processor)

    def setup_scalar_head(self) -> None:
        self.scalar_head: ScalarPotentialHead | None = None
        if self.inference_mode != "scalar_head":
            return
        if not self.scalar_head_path:
            raise ValueError("scalar_head_path is required for scalar_head inference")
        payload = torch.load(
            self.scalar_head_path, map_location="cpu", weights_only=False
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
        """Return a sigmoid-bounded potential from the trained scalar head."""
        if self.scalar_head is None:
            raise RuntimeError("Scalar potential head is not initialized")
        outputs = self._model(
            **batched_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = batched_inputs["attention_mask"].bool()
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        last_positions = positions.masked_fill(~attention_mask, -1).amax(dim=1)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        features = hidden[batch_indices, last_positions].float()
        logits = self.scalar_head(features)
        return torch.sigmoid(logits)

    def setup_reward_parser(self) -> None:
        self.reward_parser = get_reward_parser(
            self.cfg.get("reward_parser_name", "base_reward_parser")
        )(**self.cfg.get("reward_parser_params", {}))

    def apply_gt_success_bonus(
        self, rewards: torch.Tensor, reward_input: dict[str, Any]
    ) -> torch.Tensor:
        if rewards is None or self.gt_success_bonus == 0.0:
            return rewards
        env_infos = (
            reward_input.get("env_infos") if isinstance(reward_input, dict) else None
        )
        if not isinstance(env_infos, dict):
            return rewards

        success = None
        final_info = env_infos.get("final_info", {})
        for info_dict in (
            env_infos,
            env_infos.get("episode"),
            final_info,
            final_info.get("episode") if isinstance(final_info, dict) else None,
        ):
            if not isinstance(info_dict, dict):
                continue
            for key in ("success", "success_at_end", "success_once"):
                value = info_dict.get(key)
                if value is not None:
                    success = torch.as_tensor(value).reshape(-1).bool()
                    break
            if success is not None:
                break

        if success is None or success.shape[0] != rewards.shape[0]:
            return rewards
        bonus = success.to(device=rewards.device, dtype=rewards.dtype)
        return rewards + (bonus * self.gt_success_bonus).view(
            -1, *([1] * (rewards.dim() - 1))
        )

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
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )

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

        self._success_adapter_name: str | None = None
        if self.success_lora_path:
            success_weights_path = os.path.join(
                self.success_lora_path,
                "actor",
                "model_state_dict",
                "full_weights.pt",
            )
            success_checkpoint = torch.load(
                success_weights_path,
                map_location="cpu",
                weights_only=True,
            )
            success_lora_state = {
                key.removeprefix("module."): value
                for key, value in success_checkpoint.items()
                if "lora_" in key
            }
            if success_lora_state:
                if not hasattr(self._model, "add_adapter"):
                    raise ValueError(
                        "A success LoRA adapter requires a primary LoRA adapter"
                    )
                success_lora_rank = next(
                    int(value.shape[0])
                    for key, value in success_lora_state.items()
                    if "lora_A" in key
                )
                success_target_modules = sorted(
                    {
                        key.split(".lora_")[0].split(".")[-1]
                        for key in success_lora_state
                        if ".lora_" in key
                    }
                )
                self._model.add_adapter(
                    "success",
                    LoraConfig(
                        r=success_lora_rank,
                        lora_alpha=success_lora_rank,
                        lora_dropout=0.0,
                        target_modules=success_target_modules,
                        init_lora_weights="gaussian",
                    ),
                )
                set_peft_model_state_dict(
                    self._model,
                    success_lora_state,
                    adapter_name="success",
                )
                self._model.set_adapter("default")
                self._success_adapter_name = "success"
            else:
                raise ValueError(
                    "success_lora_path must point to a checkpoint containing "
                    "LoRA weights"
                )
            del success_lora_state
            del success_checkpoint

        self._model.eval()

    @torch.no_grad()
    def compute_reward(
        self,
        observations: Any,
    ) -> torch.Tensor:
        batched_inputs = self.input_builder.build_inputs(
            observations, self._model.device
        )
        if self.inference_mode == "scalar_head":
            rewards = self.compute_scalar_potential(batched_inputs).cpu()
            del batched_inputs
            return self.apply_gt_success_bonus(rewards, observations)

        rewards = self._generate_and_parse_rewards(batched_inputs)
        del batched_inputs
        return self.apply_gt_success_bonus(rewards, observations)


class HistoryVLMRewardModel(VLMRewardModel):
    def __init__(self, cfg: DictConfig):
        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(cfg.get("infer_micro_batch_size", 0))
        self.interval_reward: float = float(cfg.get("interval_reward", 0.0))
        self.potential_scale: float = float(cfg.get("potential_scale", 1.0))
        self.potential_gamma: float = float(cfg.get("potential_gamma", 1.0))
        self.potential_ema_alpha: float = float(cfg.get("potential_ema_alpha", 1.0))
        self.potential_clip: float = float(cfg.get("potential_clip", 0.0))
        self._previous_potentials: torch.Tensor | None = None
        self._success_fired: torch.Tensor | None = None
        self._success_streak: torch.Tensor | None = None

        super().__init__(cfg)
        self.success_gen_kwargs = {
            "max_new_tokens": int(cfg.get("success_max_new_tokens", 3)),
            "do_sample": False,
            "temperature": 0.0,
        }

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.cfg.get("input_builder_name", "history_vlm_input_builder")
        )(
            **self.cfg.get("input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        assert isinstance(self.input_builder, HistoryVLMInputBuilder), (
            "HistoryVLMRewardModel only supports HistoryVLMInputBuilder"
        )

        self.success_input_builder: HistoryVLMInputBuilder | None = None
        self.success_reward_parser = None
        if self.success_lora_path:
            self.success_input_builder = get_input_builder(
                self.cfg.get(
                    "success_input_builder_name",
                    "qwentrend_terminal_success_input_builder",
                )
            )(
                **self.cfg.get("success_input_builder_params", {}),
                _processor=self._processor,
                history_buffer_names=self.history_buffer_names,
            )
            assert isinstance(self.success_input_builder, HistoryVLMInputBuilder)
            self.success_reward_parser = get_reward_parser(
                self.cfg.get(
                    "success_reward_parser_name",
                    "qwentrend_binary_digit_reward_parser",
                )
            )(**self.cfg.get("success_reward_parser_params", {}))

    def forward(
        self, input_data: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "HistoryVLMRewardModel is a frozen inference-time reward model; training via forward() is not supported."
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

    def _infer_batch_size(self, value: Any) -> int:
        if isinstance(value, dict):
            for item in value.values():
                try:
                    return self._infer_batch_size(item)
                except ValueError:
                    continue
            raise ValueError("Unable to infer batch size from an empty mapping.")
        if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
            return len(value)
        raise ValueError(
            f"Unable to infer batch size from value of type {type(value)!r}."
        )

    def _history_input_batch_size(
        self,
        history_input: dict[str, dict[str, list[list[Any]]]],
        observations: dict[str, Any],
    ) -> int:
        for history_buffer in history_input.values():
            for histories in history_buffer.values():
                return len(histories)
        return self._infer_batch_size(observations)

    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        history_input: dict[str, dict[str, list[list[Any]]]] = reward_input[
            "history_input"
        ]
        observations = {
            key: value for key, value in reward_input.items() if key != "history_input"
        }
        input_batch_size = self._history_input_batch_size(history_input, observations)
        if not any(history_buffer for history_buffer in history_input.values()):
            return torch.zeros(input_batch_size, dtype=torch.float32)

        infer_micro_batch_size = self.infer_micro_batch_size or input_batch_size

        reward_chunks: list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []
        success_chunks: list[torch.Tensor] = []
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            micro_observations = self.slice_observations(observations, start, end)
            micro_history_input = self.slice_history_input(history_input, start, end)
            reward_chunk = torch.full(
                (end - start,), fill_value=self.interval_reward, dtype=torch.float32
            )
            valid_chunk = torch.zeros(end - start, dtype=torch.bool)
            success_chunk = torch.zeros(end - start, dtype=torch.float32)

            batched_inputs, valid_input_ids = self.input_builder.build_inputs(
                micro_observations,
                self._model.device,
                micro_history_input,
            )
            if len(valid_input_ids) == 0:
                reward_chunks.append(reward_chunk)
                valid_chunks.append(valid_chunk)
                success_chunks.append(success_chunk)
                continue

            if self.inference_mode == "scalar_head":
                potentials = self.compute_scalar_potential(batched_inputs).cpu()
                reward_chunk[valid_input_ids] = potentials
                valid_chunk[valid_input_ids] = True
                del batched_inputs
            else:
                parsed_rewards = self._generate_and_parse_rewards(batched_inputs)
                del batched_inputs
                reward_chunk[valid_input_ids] = parsed_rewards.to(dtype=torch.float32)
                valid_chunk[valid_input_ids] = True
            if self.success_input_builder is not None:
                success_inputs, success_input_ids = (
                    self.success_input_builder.build_inputs(
                        micro_observations,
                        self._model.device,
                        micro_history_input,
                    )
                )
                if success_input_ids:
                    if self._success_adapter_name is not None:
                        self._model.set_adapter(self._success_adapter_name)
                    prompt_length = success_inputs["input_ids"].shape[-1]
                    success_output_ids = self._model.generate(
                        **success_inputs,
                        **self.success_gen_kwargs,
                    )
                    success_outputs = self._processor.batch_decode(
                        success_output_ids[..., prompt_length:],
                        skip_special_tokens=True,
                    )
                    success_chunk[success_input_ids] = (
                        self.success_reward_parser.parse_rewards(success_outputs)
                    )
                    del success_output_ids
                    del success_outputs
                    if self._success_adapter_name is not None:
                        self._model.set_adapter("default")
                    del success_inputs

            reward_chunks.append(reward_chunk)
            valid_chunks.append(valid_chunk)
            success_chunks.append(success_chunk)

        rewards = torch.cat(reward_chunks, dim=0)
        if self.inference_mode == "scalar_head":
            valid_mask = torch.cat(valid_chunks, dim=0)
            dones = observations.get("dones")
            potentials = rewards
            rewards = self.potential_differences(
                potentials,
                valid_mask,
                dones,
            )
            if self.success_input_builder is not None:
                success_potentials = torch.cat(success_chunks, dim=0)
                if self.success_bonus != 0.0:
                    rewards = self.apply_model_success_bonus(
                        rewards, success_potentials, valid_mask, dones
                    )
        return self.apply_gt_success_bonus(rewards, observations)

    def apply_model_success_bonus(
        self,
        rewards: torch.Tensor,
        success_probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
        dones: Any = None,
    ) -> torch.Tensor:
        """Add a one-shot VLM success bonus and reset state at episode end."""
        if self._success_fired is None or self._success_fired.shape != rewards.shape:
            self._success_fired = torch.zeros_like(rewards, dtype=torch.bool)
            self._success_streak = torch.zeros_like(rewards, dtype=torch.int32)
        assert self._success_streak is not None
        above_threshold = valid_mask & (success_probabilities >= self.success_threshold)
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
        if self._previous_potentials is None or (
            self._previous_potentials.shape != potentials.shape
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
