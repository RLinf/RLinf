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

import torch


def _image_to_chw_float(
    image: torch.Tensor, image_size: int | None = None
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(
            f"Expected image [B,H,W,C] or [B,C,H,W], got {tuple(image.shape)}"
        )
    needs_rescale = not image.is_floating_point()
    if not needs_rescale and image.numel() > 0:
        valid_unit_range = (
            torch.isfinite(image).all() & (image >= 0).all() & (image <= 1).all()
        )
        if not bool(valid_unit_range):
            raise ValueError(
                "Floating-point images must contain finite values in [0, 1]"
            )
    if image.shape[-1] in (1, 3):
        image = image.permute(0, 3, 1, 2).contiguous()
    elif image.shape[1] not in (1, 3):
        raise ValueError(
            f"Expected image channel dimension to be 1 or 3, got {tuple(image.shape)}"
        )
    if image_size is not None and tuple(image.shape[-2:]) != (image_size, image_size):
        image = torch.nn.functional.interpolate(
            image.to(dtype=torch.float32),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        image = image.to(dtype=torch.float32)
    if needs_rescale:
        image = image / 255.0
    return image


def build_lerobot_batch_from_env_obs(
    env_obs: dict[str, Any],
    *,
    image_size: int | None = None,
) -> dict[str, Any]:
    """Convert RLinf environment observations to LeRobot policy inputs.

    Args:
        env_obs: Batched images, robot states, and task descriptions.
        image_size: Optional square image size expected by the policy.

    Returns:
        A LeRobot-compatible policy input batch.
    """
    batch: dict[str, Any] = {}
    batch["observation.images.image"] = _image_to_chw_float(
        env_obs["main_images"], image_size
    )
    if env_obs.get("wrist_images") is not None:
        batch["observation.images.image2"] = _image_to_chw_float(
            env_obs["wrist_images"], image_size
        )
    batch["observation.state"] = env_obs["states"].to(dtype=torch.float32)
    batch["task"] = list(env_obs["task_descriptions"])
    return batch
