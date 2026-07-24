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

"""Base model classes for PyTorch models, aligned with JAX models/model.py."""

from __future__ import annotations

import abc
import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

logger = logging.getLogger("openpi")

# Image keys expected by the model
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize images to target size with padding to preserve aspect ratio.

    Args:
        images: (..., H, W, C) float32 tensor in [-1, 1] or uint8 in [0, 255]
        height: target height
        width: target width
        mode: interpolation mode

    Returns:
        Resized and padded tensor
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # (B, H, W, C) -> (B, C, H, W)
    images = images.permute(0, 3, 1, 2)
    B, C, cur_h, cur_w = images.shape

    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    resized = tF.interpolate(
        images, size=(resized_h, resized_w), mode=mode, align_corners=False
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype in (torch.float32, torch.float16, torch.bfloat16):
        resized = resized.clamp(-1.0, 1.0)

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w

    fill_val = 0 if images.dtype == torch.uint8 else -1.0
    padded = tF.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=fill_val
    )

    # (B, C, H, W) -> (B, H, W, C)
    padded = padded.permute(0, 2, 3, 1)

    if squeeze_batch:
        padded = padded.squeeze(0)
    return padded


@dataclasses.dataclass
class Observation:
    """Holds observations, i.e., inputs to the model. PyTorch-compatible version."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    pcd_xyz: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Convert a nested dict to an Observation."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )

        images = data["image"]
        image_masks = data["image_mask"]

        # Convert uint8 images to [-1, 1] float32
        for key in images:
            if images[key].dtype == torch.uint8:
                images[key] = images[key].to(torch.float32) / 255.0 * 2.0 - 1.0
            elif images[key].dtype == np.uint8:
                images[key] = (
                    torch.from_numpy(images[key].astype(np.float32)) / 255.0 * 2.0 - 1.0
                )

        return cls(
            images=images,
            image_masks=image_masks,
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            pcd_xyz=data.get("pcd_xyz"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        if "actions" in result:
            del result["actions"]
        return result


def _tensor_to_dtype(t: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Cast a tensor to dtype, but skip integer tensors (e.g. token indices)."""
    if t is None:
        return None
    if t.dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.bool):
        return t
    return t.to(dtype=dtype)


def _observation_to_dtype(obs: Observation, dtype: torch.dtype) -> Observation:
    """Cast all float tensors in an Observation to the target dtype.

    Used to ensure inputs match FSDP2 MixedPrecisionPolicy parameter dtype,
    since cast_forward_inputs cannot reach tensors nested inside dataclasses.

    Images are kept in float32 — SigLIPViT runs stem in float32 internally (matching JAX).
    """
    return Observation(
        images=obs.images,  # Leave images in float32 for SigLIPViT stem
        image_masks={k: _tensor_to_dtype(v, dtype) for k, v in obs.image_masks.items()},
        state=obs.state.to(dtype=dtype),
        tokenized_prompt=_tensor_to_dtype(obs.tokenized_prompt, dtype),
        tokenized_prompt_mask=_tensor_to_dtype(obs.tokenized_prompt_mask, dtype),
        token_ar_mask=_tensor_to_dtype(obs.token_ar_mask, dtype),
        token_loss_mask=_tensor_to_dtype(obs.token_loss_mask, dtype),
        pcd_xyz=_tensor_to_dtype(obs.pcd_xyz, dtype),
    )


def preprocess_observation(
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    rng: torch.Generator | None = None,
) -> Observation:
    """Preprocess observations with optional image augmentations.

    For training, applies random crop, rotate, and color jitter augmentations.
    Resizes images to the target resolution with padding.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images)}"
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # Standardize to (B, H, W, C) format if input is (B, C, H, W)
        if image.shape[-1] <= 4 and image.shape[1] > 4:
            pass  # already (B, H, W, C)
        elif image.shape[1] <= 4:
            image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        if image.shape[1:3] != image_resolution:
            image = resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmentations
            image = image / 2.0 + 0.5
            # image is in (B, H, W, C) format
            # Apply augmentations per image in batch
            B = image.shape[0]
            augmented = []
            for i in range(B):
                aug_image = image[i]  # (H, W, C)
                aug_image = aug_image.permute(2, 0, 1)  # (C, H, W) for torchvision

                if "wrist" not in key:
                    # Random crop (95% of size) then resize back
                    import torchvision.transforms.functional as TF

                    h, w = aug_image.shape[1], aug_image.shape[2]
                    crop_h, crop_w = int(h * 0.95), int(w * 0.95)
                    top = (
                        torch.randint(0, h - crop_h + 1, (1,), generator=rng).item()
                        if rng is not None
                        else 0
                    )
                    left = (
                        torch.randint(0, w - crop_w + 1, (1,), generator=rng).item()
                        if rng is not None
                        else 0
                    )
                    aug_image = TF.crop(aug_image, top, left, crop_h, crop_w)
                    aug_image = TF.resize(aug_image, [h, w], antialias=True)

                    # Random rotation (-5 to 5 degrees)
                    angle = (
                        (torch.rand(1, generator=rng).item() * 10 - 5)
                        if rng is not None
                        else 0.0
                    )
                    aug_image = TF.rotate(aug_image, angle, fill=0)

                # Color jitter
                # Simulate augmax ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                if rng is not None:
                    b = torch.rand(1, generator=rng).item() * 0.3
                    c = torch.rand(1, generator=rng).item() * 0.4
                    s = torch.rand(1, generator=rng).item() * 0.5
                    aug_image = TF.adjust_brightness(aug_image, 1.0 + b)
                    aug_image = TF.adjust_contrast(aug_image, 1.0 + c)
                    aug_image = TF.adjust_saturation(aug_image, 1.0 + s)

                aug_image = aug_image.permute(1, 2, 0)  # back to (H, W, C)
                augmented.append(aug_image)
            image = torch.stack(augmented, dim=0)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        out_images[key] = image

    # Build masks
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool)
        else:
            mask = observation.image_masks[key]
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            out_masks[key] = mask.bool()

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        pcd_xyz=observation.pcd_xyz,
    )


@dataclasses.dataclass
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models."""

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @abc.abstractmethod
    def create(self, **kwargs) -> BaseModel:
        """Create a new model, initializing parameters."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """Create fake observations for testing."""
        return Observation(
            images={
                k: torch.ones(batch_size, *IMAGE_RESOLUTION, 3) for k in IMAGE_KEYS
            },
            image_masks={
                k: torch.ones(batch_size, dtype=torch.bool) for k in IMAGE_KEYS
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        """Create fake actions for testing."""
        return torch.ones(batch_size, self.action_horizon, self.action_dim)


class BaseModel(nn.Module, abc.ABC):
    """Base class for all model implementations."""

    def __init__(self, action_dim: int, action_horizon: int, max_token_len: int):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len

    @abc.abstractmethod
    def compute_loss(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute the loss for a batch of observations and actions."""

    @abc.abstractmethod
    def sample_actions(
        self,
        observation: Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions given an observation."""

    def forward(
        self,
        observation: Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Default forward pass computes the loss."""
        return self.compute_loss(observation, actions, train=train, rng=rng)
