# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""OpenPI transforms for the SO-101 joint-space action contract."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

SO101_ACTION_DIM = 6


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Return an RGB image in HWC uint8 format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Return the six canonical SO-101 joint actions from a padded model output."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :SO101_ACTION_DIM], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != SO101_ACTION_DIM:
            raise ValueError(f"Expected SO-101 actions (*, 6), got {actions.shape}.")
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Pad six joint values and map one camera into Pi0.5's input schema."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if self.model_type != _model.ModelType.PI05:
            raise ValueError(f"SO-101 supports Pi0.5 inputs, got {self.model_type}.")

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (SO101_ACTION_DIM,):
            raise ValueError(f"Expected SO-101 state shape (6,), got {state.shape}.")

        image = _parse_image(data["observation/image"])
        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != SO101_ACTION_DIM:
                raise ValueError(f"Expected SO-101 actions (N, 6), got {actions.shape}.")
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return inputs
