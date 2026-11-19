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
import math
from collections.abc import Mapping, Sequence
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
    # Short-memory cameras are [B, K, H, W, C], with [B, K] image masks.
    history_time_offsets: torch.Tensor | None = None

    @classmethod
    def from_observation_like(cls, observation: Any) -> Observation:
        """Convert a local mapping or official OpenPI Observation to this type.

        The official ``openpi.models.model.Observation`` exposes ``to_dict()``,
        whose keys match the local ``from_dict`` boundary. Its loader has already
        applied all data transforms, so this only changes the Python container.
        """
        if isinstance(observation, cls):
            return observation
        if isinstance(observation, Mapping):
            return cls.from_dict(dict(observation))

        to_dict = getattr(observation, "to_dict", None)
        if callable(to_dict):
            return cls.from_dict(to_dict())
        raise TypeError(
            "SFT observation must be a local Observation, mapping, or an "
            f"OpenPI Observation with to_dict(); got {type(observation)!r}."
        )

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
            history_time_offsets=data.get("history_time_offsets"),
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
        history_time_offsets=obs.history_time_offsets,
    )


def _blend_images(
    images: torch.Tensor, other: torch.Tensor, factors: torch.Tensor
) -> torch.Tensor:
    """Blend with per-image factors, rounding each product to the image dtype."""
    first = (images * factors[:, 0]).to(images.dtype)
    second = (other * factors[:, 1]).to(images.dtype)
    return (first + second).clamp(0, 1)


def _augment_images(
    images: torch.Tensor,
    *,
    spatial: bool,
    rng: torch.Generator | None,
) -> torch.Tensor:
    """Augment BHWC images in [0, 1] with independent parameters per image."""
    import torchvision.transforms.functional as TF

    batch, height, width, channels = images.shape
    crop_height, crop_width = int(height * 0.95), int(width * 0.95)
    parameters = []
    # Keep the draw order shared with checkpointed torch RNG state.
    for _ in range(batch):
        row = []
        if spatial:
            top = torch.randint(0, height - crop_height + 1, (1,), generator=rng).item()
            left = torch.randint(0, width - crop_width + 1, (1,), generator=rng).item()
            angle = torch.rand(1, generator=rng).item() * 10 - 5
            radians = math.radians(angle)
            cosine, sine = math.cos(radians), math.sin(radians)
            row.extend((top, left, cosine, -sine, 0.0, sine, cosine, 0.0))
        for amount in (0.3, 0.4, 0.5):
            factor = 1.0 + (torch.rand(1, generator=rng).item() * 2 - 1) * amount
            row.extend((factor, 1.0 - factor))
        parameters.append(row)
    # Python scalar factors use float32 opmath with half-precision images.
    parameter_dtype = torch.float64 if images.dtype == torch.float64 else torch.float32
    parameters = images.new_tensor(parameters, dtype=parameter_dtype)

    if spatial:
        offsets = parameters[:, :2].to(torch.long)
        rows = torch.arange(crop_height, device=images.device) + offsets[:, :1]
        columns = torch.arange(crop_width, device=images.device) + offsets[:, 1:2]
        indices = torch.arange(batch, device=images.device)[:, None, None]
        images = images[indices, rows[:, :, None], columns[:, None, :], :]
        images = TF.resize(images.permute(0, 3, 1, 2), [height, width], antialias=True)

        # Normalize after rotation, using the pixel centers used by torchvision.
        theta = parameters[:, 2:8].to(images.dtype).reshape(batch, 2, 3)
        base = images.new_empty(1, height, width, 3)
        base[..., 0] = torch.linspace(
            -width * 0.5 + 0.5, width * 0.5 - 0.5, width, device=images.device
        )
        base[..., 1] = torch.linspace(
            -height * 0.5 + 0.5, height * 0.5 - 0.5, height, device=images.device
        )[:, None]
        base[..., 2] = 1
        scale = images.new_tensor([width * 0.5, height * 0.5])
        base = base.reshape(1, height * width, 3)
        # Batched GEMM can round across a nearest-neighbor pixel boundary.
        grids = []
        for matrix in theta:
            matrix = matrix.clone().reshape(1, 2, 3).transpose(1, 2) / scale
            grids.append(base.bmm(matrix))
        grid = torch.cat(grids, dim=0)
        output_dtype = images.dtype
        needs_cast = images.dtype != grid.dtype
        images = tF.grid_sample(
            images.to(grid.dtype),
            grid.reshape(batch, height, width, 2),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        if needs_cast:
            images = images.to(output_dtype)
        factors = parameters[:, 8:].reshape(batch, 3, 2, 1, 1, 1)
    else:
        images = images.permute(0, 3, 1, 2)
        factors = parameters.reshape(batch, 3, 2, 1, 1, 1)

    images = _blend_images(images, images.new_zeros(()), factors[:, 0])
    gray = TF.rgb_to_grayscale(images) if channels == 3 else images
    mean = torch.stack([image.mean().reshape(1, 1, 1) for image in gray])
    images = _blend_images(images, mean, factors[:, 1])
    if channels != 1:
        images = _blend_images(images, TF.rgb_to_grayscale(images), factors[:, 2])
    return images.permute(0, 2, 3, 1).contiguous()


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

    Random draws use ``rng`` when provided; when ``rng`` is ``None`` (the
    default for every training caller) they fall back to the global torch
    RNG, matching upstream openpi. The eval path (``train=False``) applies
    no augmentation and is unaffected.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images)}"
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        history_shape = image.shape[:2] if image.ndim == 5 else None
        if history_shape is not None:
            image = image.flatten(0, 1)

        # Standardize to (B, H, W, C) format if input is (B, C, H, W)
        if image.shape[-1] <= 4 and image.shape[1] > 4:
            pass  # already (B, H, W, C)
        elif image.shape[1] <= 4:
            image = image.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        if image.shape[1:3] != image_resolution:
            image = resize_with_pad_torch(image, *image_resolution)

        if train and history_shape is None:
            image = (
                _augment_images(image / 2.0 + 0.5, spatial="wrist" not in key, rng=rng)
                * 2.0
                - 1.0
            )
        elif train:
            import torchvision.transforms.functional as TF

            # Convert from [-1, 1] to [0, 1] for augmentations
            image = image / 2.0 + 0.5
            # Keep matching patches aligned by sharing augmentation across time.
            augmentation_images = (
                image.reshape(*history_shape, *image.shape[1:])
                if history_shape is not None
                else image
            )
            B = augmentation_images.shape[0]
            augmented = []
            for i in range(B):
                aug_image = augmentation_images[i].movedim(-1, -3)

                if "wrist" not in key:
                    # Random crop (95% of size) then resize back
                    h, w = aug_image.shape[-2:]
                    crop_h, crop_w = int(h * 0.95), int(w * 0.95)
                    # generator=None draws from the global torch RNG
                    top = torch.randint(0, h - crop_h + 1, (1,), generator=rng).item()
                    left = torch.randint(0, w - crop_w + 1, (1,), generator=rng).item()
                    aug_image = TF.crop(aug_image, top, left, crop_h, crop_w)
                    aug_image = TF.resize(aug_image, [h, w], antialias=True)

                    # Random rotation (-5 to 5 degrees)
                    angle = torch.rand(1, generator=rng).item() * 10 - 5
                    aug_image = TF.rotate(aug_image, angle, fill=0)

                # Color jitter
                # Simulate augmax ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                b = (torch.rand(1, generator=rng).item() * 2 - 1) * 0.3
                c = (torch.rand(1, generator=rng).item() * 2 - 1) * 0.4
                s = (torch.rand(1, generator=rng).item() * 2 - 1) * 0.5
                aug_image = TF.adjust_brightness(aug_image, 1.0 + b)
                aug_image = TF.adjust_contrast(aug_image, 1.0 + c)
                aug_image = TF.adjust_saturation(aug_image, 1.0 + s)

                aug_image = aug_image.movedim(-3, -1)
                augmented.append(aug_image)
            image = torch.stack(augmented, dim=0)
            if history_shape is not None:
                image = image.flatten(0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        out_images[key] = (
            image.reshape(*history_shape, *image.shape[1:])
            if history_shape is not None
            else image
        )

    # Build masks
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            mask_shape = (
                out_images[key].shape[:2] if out_images[key].ndim == 5 else batch_shape
            )
            out_masks[key] = torch.ones(
                mask_shape, dtype=torch.bool, device=out_images[key].device
            )
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
        history_time_offsets=observation.history_time_offsets,
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
