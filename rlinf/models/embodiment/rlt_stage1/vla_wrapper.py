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

"""OpenPI wrapper reused by RLT Stage 1 for ManiSkill simulation."""

from __future__ import annotations

import torch
from openpi.models import model as _model

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.rlt_stage2.vla_wrapper import Stage2VLAWrapper


class Stage1VLAWrapper(Stage2VLAWrapper):
    """Frozen or jointly trainable OpenPI wrapper for Stage 1."""

    def __init__(
        self,
        *,
        model_path: str,
        config_name: str,
        norm_stats_path: str | None = None,
        num_images_in_input: int = 2,
        num_action_chunks: int = 10,
        action_dim: int = 8,
        num_steps: int = 5,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(
            model_path=model_path,
            config_name=config_name,
            norm_stats_path=norm_stats_path,
            num_images_in_input=num_images_in_input,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            num_steps=num_steps,
            device=device,
        )
        self.embedding_dim = 2048

    def unfreeze(self) -> None:
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def extract_embeddings(
        self,
        observation: _model.Observation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, img_masks, lang_tokens, lang_masks, _ = (
            self.model._preprocess_observation(observation, train=False)
        )
        images = [img.to(self.device) for img in images]
        img_masks = [mask.to(self.device) for mask in img_masks]
        prefix_output, prefix_pad_masks, _ = self.model._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
        return prefix_output.detach().to(torch.float32), prefix_pad_masks.detach()

    def compute_vla_loss(
        self,
        observation: _model.Observation,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            data={"observation": observation, "actions": actions},
            forward_type=ForwardType.SFT,
        ).mean()

    def compute_vla_loss_with_embeddings(
        self,
        observation: _model.Observation,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}

        original_build_prefix_cache = self.model._build_prefix_cache

        def _capture_prefix(*args, **kwargs):
            prefix_output, prefix_pad_masks, past_key_values = original_build_prefix_cache(
                *args, **kwargs
            )
            if "prefix_output" not in captured:
                captured["prefix_output"] = prefix_output.detach().to(torch.float32)
                captured["prefix_pad_masks"] = prefix_pad_masks.detach()
            return prefix_output, prefix_pad_masks, past_key_values

        self.model._build_prefix_cache = _capture_prefix
        try:
            loss = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )
        finally:
            self.model._build_prefix_cache = original_build_prefix_cache

        return captured["prefix_output"], captured["prefix_pad_masks"], loss.mean()
