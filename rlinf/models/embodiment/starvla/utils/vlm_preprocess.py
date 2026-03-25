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

"""VLM input preprocessing utilities for starVLA rollouts and training."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .profile import resolve_vlm_interface


def _to_uint8_image_array(arr: np.ndarray) -> np.ndarray:
    """Convert arbitrary image array to uint8 in [0, 255]."""
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        max_val = float(np.max(arr)) if arr.size > 0 else 0.0
        min_val = float(np.min(arr)) if arr.size > 0 else 0.0
        if max_val <= 1.0 and min_val >= 0.0:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = np.rint(arr).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_pil_preserve(image: Any):
    """Convert tensor/ndarray(/nested views) to PIL while preserving structure."""

    def _convert(obj: Any):
        if isinstance(obj, list):
            return [_convert(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(item) for item in obj)
        if isinstance(obj, Image.Image):
            return obj
        if torch.is_tensor(obj):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise TypeError(f"Unsupported image type: {type(obj)}")

        if arr.ndim == 2:
            return Image.fromarray(_to_uint8_image_array(arr), mode="L")
        if arr.ndim != 3:
            raise ValueError(
                f"Expected image array with ndim 2/3, got shape={arr.shape}."
            )

        if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported channel count: shape={arr.shape}.")

        arr = _to_uint8_image_array(arr)
        if arr.shape[-1] == 1:
            return Image.fromarray(arr[..., 0], mode="L")
        if arr.shape[-1] == 3:
            return Image.fromarray(arr, mode="RGB")
        return Image.fromarray(arr, mode="RGBA")

    return _convert(image)


def resize_images(images: list[Image.Image], target_size: int):
    """Resize images with starVLA's training-time resize utility."""
    from starVLA.training.trainer_utils.trainer_tools import (
        resize_images as _resize_images,
    )

    return _resize_images(images, target_size=target_size)


def get_train_image_size(starvla_model: Any) -> Optional[int]:
    """Read training image size from checkpoint config if provided."""
    cfg = getattr(starvla_model, "config", None)
    if cfg is None:
        return None
    vla_data = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    return getattr(vla_data, "image_size", None) if vla_data is not None else None


def build_base_vlm_inputs(
    starvla_model: Any,
    *,
    examples: list[dict[str, Any]],
    vlm_type: Optional[str] = None,
    vlm_interface: Any = None,
) -> dict[str, torch.Tensor]:
    """Build backbone-only VLM inputs from rollout examples."""
    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]

    train_obs_image_size = get_train_image_size(starvla_model)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    if vlm_type == "florence":
        single_image_batch = []
        for idx, views in enumerate(batch_images):
            if not isinstance(views, (list, tuple)) or len(views) == 0:
                raise ValueError(
                    f"Florence backbone expects non-empty image list per sample, got sample index {idx}."
                )
            single_image_batch.append([views[0]])
        batch_images = single_image_batch

    iface = vlm_interface or resolve_vlm_interface(starvla_model)
    build_inputs = getattr(iface, "build_qwenvl_inputs", None)
    if not callable(build_inputs):
        raise RuntimeError("VLM interface does not provide 'build_qwenvl_inputs(...)'.")
    return dict(
        build_inputs(
            images=batch_images,
            instructions=instructions,
        )
    )
