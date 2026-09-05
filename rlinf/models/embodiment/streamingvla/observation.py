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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class StreamingVLAObservation:
    """Transformed StreamingVLA batch, including the required action state."""

    images: dict[str, Any]
    image_masks: dict[str, Any]
    state: Any
    action_states: Any | None = None
    tokenized_prompt: Any | None = None
    tokenized_prompt_mask: Any | None = None
    token_ar_mask: Any | None = None
    token_loss_mask: Any | None = None

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "StreamingVLAObservation":
        """Move all tensors to a device while preserving discrete dtypes."""

        def move(value):
            if torch.is_tensor(value):
                if dtype is not None and torch.is_floating_point(value):
                    return value.to(device=device, dtype=dtype)
                return value.to(device=device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(move(item) for item in value)
            return value

        return StreamingVLAObservation(
            images=move(self.images),
            image_masks=move(self.image_masks),
            state=move(self.state),
            action_states=move(self.action_states),
            tokenized_prompt=move(self.tokenized_prompt),
            tokenized_prompt_mask=move(self.tokenized_prompt_mask),
            token_ar_mask=move(self.token_ar_mask),
            token_loss_mask=move(self.token_loss_mask),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamingVLAObservation":
        """Construct an observation from the OpenPI transform output mapping."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )

        images = data["image"]
        for key, image in list(images.items()):
            if hasattr(image, "dtype") and image.dtype == np.uint8:
                images[key] = image.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif torch.is_tensor(image) and image.dtype == torch.uint8:
                images[key] = (
                    image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )

        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            action_states=data.get("action_states"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )
