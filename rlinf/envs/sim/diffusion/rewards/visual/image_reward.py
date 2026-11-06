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

from typing import Any

import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from rlinf.config import torch_dtype_from_precision
from rlinf.envs.sim.diffusion.rewards import ImageRewardBackendBase
from rlinf.envs.sim.diffusion.utils import cfg_get, cfg_require
from rlinf.scheduler import Worker

REWARD_IMAGE_SIZE = 224
REWARD_MAX_TOKENS = 35


class ImageRewardScorer:
    """ImageReward scorer shared by the env reward and the ReFL actor.

    Scoring stays differentiable with respect to the images, and the model
    parameters are frozen.
    """

    def __init__(
        self,
        model_path: str,
        med_config: str,
        device: torch.device | str,
        precision: str = "bf16",
    ):
        import ImageReward as RM

        self.device = device
        self.dtype = torch_dtype_from_precision(precision)
        self.model = RM.load(model_path, device=device, med_config=med_config)
        self.model.requires_grad_(False)
        self.model.to(dtype=self.dtype)
        self.preprocess = Compose(
            [
                Resize(
                    REWARD_IMAGE_SIZE,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                CenterCrop(REWARD_IMAGE_SIZE),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @classmethod
    def from_config(cls, cfg: Any, device: torch.device | str) -> "ImageRewardScorer":
        return cls(
            model_path=str(cfg_require(cfg, "model_path")),
            med_config=str(cfg_require(cfg, "med_config")),
            device=device,
            precision=str(cfg_get(cfg, "precision", "bf16")),
        )

    def tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.model.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=REWARD_MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)
        return tokens.input_ids, tokens.attention_mask

    def score(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score images in [0, 1] with shape [B, 3, H, W]; returns shape [B]."""
        pixels = self.preprocess(images).to(dtype=self.dtype)
        rewards = self.model.score_gard(input_ids, attention_mask, pixels)
        return rewards.squeeze(-1)


class ImageRewardBackend(ImageRewardBackendBase):
    """Image reward from the ImageReward human-preference model."""

    def __init__(self, scorer: ImageRewardScorer, batch_size: int):
        self.scorer = scorer
        self.batch_size = batch_size

    @classmethod
    def _from_config(cls, cfg: Any) -> "ImageRewardBackend":
        use_gpu = bool(cfg_get(cfg, "use_gpu", True))
        device = Worker.torch_platform.current_device() if use_gpu else "cpu"
        return cls(
            scorer=ImageRewardScorer.from_config(cfg, device),
            batch_size=int(cfg_get(cfg, "batch_size", 32)),
        )

    @torch.no_grad()
    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        prompts = [record["task_description"] for record in records]
        images = torch.from_numpy(self.to_image_batch(outputs)).permute(0, 3, 1, 2)
        rewards = []
        for start in range(0, len(prompts), self.batch_size):
            end = start + self.batch_size
            pixels = images[start:end].to(self.scorer.device).float() / 255.0
            rewards.append(
                self.scorer.score(pixels, *self.scorer.tokenize(prompts[start:end]))
            )
        rewards_tensor = torch.cat(rewards).float().cpu()
        return {"avg": rewards_tensor, "image_reward": rewards_tensor}


REWARD_CLS = ImageRewardBackend


__all__ = ["ImageRewardBackend", "ImageRewardScorer", "REWARD_CLS"]
