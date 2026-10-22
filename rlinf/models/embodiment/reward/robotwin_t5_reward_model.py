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

"""Text-conditioned success classifier for RoboTwin frames.

The architecture follows the RoboTwin reward model of the WorldArena 2.0 RL benchmark:
ResNet-18 patch tokens attend to a frozen T5 encoding of the instruction.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

__all__ = ["RoboTwinT5RewardModel"]


class RoboTwinT5RewardModel(BaseRewardModel):
    """Per-frame success probability for a RoboTwin task instruction.

    Config keys: ``from_pretrained`` (the classifier checkpoint), ``t5_model_name``,
    ``image_size`` (``[3, H, W]`` the frames are resized to) and ``batch_size`` (frames
    per forward).
    """

    _VISUAL_DIM = 512
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from transformers import AutoTokenizer, T5EncoderModel

        self.image_size = tuple(cfg.get("image_size", (3, 224, 224)))[1:]
        self.batch_size = int(cfg.get("batch_size", 100))
        self.max_text_length = int(cfg.get("max_text_length", 64))

        backbone = tv_models.resnet18(weights=None)
        self.visual_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.t5_tokenizer = AutoTokenizer.from_pretrained(cfg.t5_model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(cfg.t5_model_name)
        self.t5_encoder.requires_grad_(False)
        self.text_proj = nn.Linear(self.t5_encoder.config.d_model, self._VISUAL_DIM)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self._VISUAL_DIM,
            num_heads=int(cfg.get("num_attn_heads", 8)),
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(self._VISUAL_DIM)
        self.reward_head = nn.Sequential(
            nn.Linear(self._VISUAL_DIM, int(cfg.get("hidden_dim", 256))),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(cfg.get("hidden_dim", 256)), 1),
        )
        self.register_buffer(
            "_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1), False
        )
        self._load(cfg.from_pretrained)

    def _load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        cleaned = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "_orig_mod.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = value
        self.load_state_dict(cleaned, strict=True)

    def _logits(
        self, images: torch.Tensor, instructions: Optional[list[str]]
    ) -> torch.Tensor:
        """``[B, 3, H, W]`` frames in ``[0, 1]`` to ``[B]`` logits."""
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(
                images, size=self.image_size, mode="bilinear", align_corners=False
            )
        feat = self.visual_encoder((images - self._mean) / self._std)
        tokens = feat.flatten(2).transpose(1, 2)
        if instructions is not None:
            text = self.t5_tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            ).to(tokens.device)
            with torch.no_grad():
                hidden = self.t5_encoder(**text).last_hidden_state
            text_tokens = self.text_proj(hidden)
            attended, _ = self.cross_attn(
                tokens,
                text_tokens,
                text_tokens,
                key_padding_mask=text["attention_mask"] == 0,
            )
            tokens = self.ln_attn(tokens + attended)
        return self.reward_head(tokens.mean(dim=1)).squeeze(-1)

    def forward(
        self, input_data: torch.Tensor, instructions: Optional[list[str]] = None
    ) -> dict[str, Any]:
        logits = self._logits(input_data, instructions)
        return {"logits": logits, "probabilities": torch.sigmoid(logits)}

    @torch.no_grad()
    def compute_reward(self, observations: dict[str, Any]) -> torch.Tensor:
        """Success probability per ``main_images`` frame (``[B, H, W, 3]`` uint8)."""
        images = torch.as_tensor(observations["main_images"]).permute(0, 3, 1, 2)
        return self.predict_rew(
            images.float() / 127.5 - 1.0, observations.get("task_descriptions")
        )

    @torch.no_grad()
    def predict_rew(
        self, obs: torch.Tensor, instructions: Optional[list[str]] = None
    ) -> torch.Tensor:
        """Success probability per frame for world-model frames in ``[-1, 1]``."""
        obs = ((obs + 1.0) / 2.0).to(self._mean.device)
        probs = []
        for start in range(0, obs.shape[0], self.batch_size):
            stop = start + self.batch_size
            texts = None if instructions is None else instructions[start:stop]
            probs.append(torch.sigmoid(self._logits(obs[start:stop], texts)).float())
        return torch.cat(probs)
