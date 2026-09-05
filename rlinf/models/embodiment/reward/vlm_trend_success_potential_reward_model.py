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

"""Dedicated VLM Trend Success + Potential reward inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.vlm_reward_model import BufferedVLMRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    BufferedVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    BaseRewardParser,
    get_reward_parser,
)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _load_lora_state_from_full_weights(path: str) -> dict[str, torch.Tensor]:
    """Load ``lora_*`` tensors from an explicit ``full_weights.pt`` file.

    Args:
        path: Full path to ``full_weights.pt``. Directories are rejected.

    Returns:
        Mapping of LoRA parameter names to tensors.

    Raises:
        FileNotFoundError: If ``path`` is not a file or contains no LoRA tensors.
    """
    weights_path = Path(path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected the full path to full_weights.pt, got {path}. "
            "Pass the file itself, for example "
            ".../actor/model_state_dict/full_weights.pt."
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    lora_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if "lora_" in key
    }
    if not lora_state:
        raise FileNotFoundError(f"{weights_path} contains no lora_* tensors")
    return lora_state


def load_lora_adapter(
    model: torch.nn.Module, path: str, adapter_name: str = "default"
) -> torch.nn.Module:
    """Load one PEFT adapter directory or an RLinf ``full_weights.pt`` LoRA dump.

    Args:
        model: Base model or an existing ``PeftModel``.
        path: Full path to ``full_weights.pt`` (typically
            ``.../actor/model_state_dict/full_weights.pt`` from VLM SFT), or a
            PEFT adapter directory that contains ``adapter_config.json``.
            Parent checkpoint directories are not searched.
        adapter_name: PEFT adapter name to attach.

    Returns:
        The model with the named adapter loaded.

    Raises:
        FileNotFoundError: If ``path`` is neither a PEFT adapter directory nor
            a ``full_weights.pt`` file.
        RuntimeError: If the checkpoint contains unexpected LoRA keys.
    """
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        set_peft_model_state_dict,
    )

    adapter_dir = Path(path)
    if (adapter_dir / ADAPTER_CONFIG_FILENAME).is_file():
        if isinstance(model, PeftModel):
            model.load_adapter(str(adapter_dir), adapter_name=adapter_name)
            if adapter_name != "default":
                model.set_adapter("default")
            return model
        return PeftModel.from_pretrained(
            model, str(adapter_dir), adapter_name=adapter_name
        )

    if not adapter_dir.is_file():
        raise FileNotFoundError(
            f"No LoRA adapter found at {path}. Pass the full path to "
            "full_weights.pt (typically "
            ".../actor/model_state_dict/full_weights.pt from VLM SFT) or a "
            f"PEFT adapter directory that contains {ADAPTER_CONFIG_FILENAME}."
        )

    state = _load_lora_state_from_full_weights(path)
    rank = next(int(value.shape[0]) for key, value in state.items() if "lora_A" in key)
    targets = sorted(
        {key.split(".lora_")[0].split(".")[-1] for key in state if ".lora_" in key}
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=targets,
        init_lora_weights="gaussian",
    )
    if isinstance(model, PeftModel):
        model.add_adapter(adapter_name, config)
    else:
        model = get_peft_model(model, config, adapter_name=adapter_name)

    state = {
        key.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B."
        ): value
        for key, value in state.items()
    }
    result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {result.unexpected_keys}")
    if adapter_name != "default":
        model.set_adapter("default")
    return model


class ScalarPotentialHead(torch.nn.Module):
    """Map frozen VLM prompt features to scalar potential logits."""

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
        """Map features of shape ``(batch, dim)`` to scalar logits."""
        return self.net(features).squeeze(-1)


@torch.no_grad()
def extract_prompt_features(
    model: torch.nn.Module,
    batched_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pool the last attended token from the model's final hidden layer."""
    outputs = model(
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
    return hidden[batch_indices, last_positions].float()


class VLMTrendSuccessPotentialRewardModel(BufferedVLMRewardModel):
    """Combine scalar potential shaping with a one-shot VLM success bonus."""

    def __init__(self, cfg: DictConfig) -> None:
        for field_name in ("lora_path", "scalar_head_path", "success_lora_path"):
            if not cfg.get(field_name):
                raise ValueError(
                    f"reward.model.{field_name} must be set for "
                    "VLMTrendSuccessPotentialRewardModel"
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
        if not any(history_buffer for history_buffer in history_input.values()):
            return torch.zeros(batch_size, dtype=torch.float32)

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
        dones = observations.get("dones")
        rewards = self.potential_differences(potentials, valid_mask, dones)
        if self.success_bonus != 0.0:
            rewards = self.apply_model_success_bonus(
                rewards,
                torch.cat(success_chunks),
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
