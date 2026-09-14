# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation and action transforms for Psi0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pad_last_dim(value: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad an array's final dimension to ``target_dim``."""
    value = np.asarray(value)
    if value.shape[-1] > target_dim:
        raise ValueError(
            f"Cannot pad last dimension {value.shape[-1]} to smaller {target_dim}."
        )
    if value.shape[-1] == target_dim:
        return value.copy()
    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, target_dim - value.shape[-1])
    return np.pad(value, pad_width, mode="constant")


@dataclass(frozen=True)
class Psi0ProcessedBatch:
    """Inputs expected by the fixed Psi0 inference path."""

    observations: list[list[Image.Image]]
    states: torch.Tensor
    instructions: list[str]


class Psi0ProcessorAdapter:
    """Translate RLinf observations using checkpoint-owned Psi0 transforms."""

    def __init__(
        self,
        *,
        field_transform: Any,
        action_dim: int = 36,
        state_dim: int = 36,
        image_size: tuple[int, int] = (180, 320),
        image_transform: Any | None = None,
    ) -> None:
        self.field_transform = field_transform
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_size = image_size
        self.image_transform = image_transform

    @classmethod
    def from_upstream_transform(
        cls,
        field_transform: Any,
        model_transform: Any | None = None,
        *,
        image_size: tuple[int, int] = (180, 320),
    ) -> "Psi0ProcessorAdapter":
        """Create an adapter from Psi0's checkpoint-owned transforms."""
        required = ("action_min", "action_max")
        if field_transform.normalize_state:
            required += ("state_min", "state_max")
        missing = [
            name for name in required if getattr(field_transform, name, None) is None
        ]
        if missing:
            raise ValueError(f"Psi0 normalization statistics are missing: {missing}.")
        image_transform = None
        if model_transform is not None:
            from torchvision.transforms import v2

            image_transform = v2.Compose(
                [model_transform.resize(), model_transform.center_crop()]
            )
        return cls(
            field_transform=field_transform,
            action_dim=int(field_transform.pad_action_dim or 36),
            state_dim=int(field_transform.pad_state_dim or 36),
            image_size=image_size,
            image_transform=image_transform,
        )

    def normalize_state(self, states: Any) -> torch.Tensor:
        """Pad the SIMPLE state and apply the checkpoint-owned transform."""
        states_np = _as_numpy(states).astype(np.float32, copy=False)
        if states_np.ndim == 2:
            states_np = states_np[:, None, :]
        if states_np.ndim != 3:
            raise ValueError(f"Psi0 states must be [B, T, D], got {states_np.shape}.")
        states_np = pad_last_dim(states_np, self.state_dim)
        if self.field_transform.normalize_state:
            states_np = self.field_transform.normalize_state_func(states_np)
        return torch.from_numpy(np.asarray(states_np, dtype=np.float32))

    def denormalize_action(self, actions: Any) -> torch.Tensor:
        """Apply the checkpoint-owned inverse action transform."""
        action_shape = tuple(actions.shape)
        if action_shape[-1] != self.action_dim:
            raise ValueError(
                f"Psi0 actions require {self.action_dim} dimensions, got {action_shape}."
            )
        result = self.field_transform.denormalize(actions)
        if torch.is_tensor(result):
            return result.float()
        return torch.from_numpy(np.asarray(result, dtype=np.float32))

    def process(self, env_obs: dict[str, Any]) -> Psi0ProcessedBatch:
        """Convert an RLinf observation batch into fixed Psi0 inputs."""
        images = _as_numpy(env_obs["main_images"])
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Psi0 main_images must be RGB/HWC [B,H,W,3], got {images.shape}."
            )
        if images.dtype != np.uint8:
            raise TypeError(f"Psi0 main_images must be uint8, got {images.dtype}.")
        if not images.flags.c_contiguous:
            images = np.ascontiguousarray(images)

        height, width = self.image_size
        observations = []
        for image in images:
            pil_image = Image.fromarray(image)
            if self.image_transform is not None:
                pil_image = self.image_transform(pil_image)
            else:
                pil_image = pil_image.resize((width, height))
            observations.append([pil_image])
        instructions = [str(item) for item in env_obs["task_descriptions"]]
        if len(instructions) != len(observations):
            raise ValueError("Psi0 image and instruction batch sizes do not match.")
        states = self.normalize_state(env_obs["states"])
        if states.shape[0] != len(observations):
            raise ValueError("Psi0 image and state batch sizes do not match.")
        return Psi0ProcessedBatch(
            observations=observations,
            states=states,
            instructions=instructions,
        )
