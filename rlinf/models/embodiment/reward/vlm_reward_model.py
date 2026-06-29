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
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
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
        from transformers import AutoProcessor

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
        from transformers import AutoModelForVision2Seq

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

                from peft import (
                    LoraConfig,
                    get_peft_model,
                    set_peft_model_state_dict,
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


class HistoryVLMRewardModel(VLMRewardModel):
    def __init__(self, cfg: DictConfig):
        self.history_buffer_names = list(cfg.history_buffers.keys())
        self.infer_micro_batch_size: int = int(cfg.get("infer_micro_batch_size", 0))
        self.interval_reward: float = float(cfg.get("interval_reward", 0.0))

        super().__init__(cfg)

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
