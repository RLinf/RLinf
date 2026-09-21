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

"""SO-101 input/output transforms used by the shared OpenPI policy.

This module follows the existing ``policies/`` convention: it adapts an
environment's observations and actions at the OpenPI policy boundary. It does
not define an SO-101 model; ``pi05_so101`` uses the shared PI05 implementation.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_STATE_DIM = len(SO101_JOINT_NAMES)
SO101_ACTION_DIM = len(SO101_JOINT_NAMES)


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a LeRobot image to HWC uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("SO-101 image contains non-finite values.")
        # LeRobot normally sends uint8 frames, but callers may provide either
        # [0, 1] or [0, 255] floating-point images.
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Select the six SO-101 dimensions from padded model actions."""

    joint_dim: int = SO101_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.joint_dim])}


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Map one wrist image and six joint positions into the PI05 input space."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05
    joint_dim: int = SO101_STATE_DIM

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (self.joint_dim,):
            raise ValueError(
                f"Expected SO-101 state shape ({self.joint_dim},), got {state.shape}."
            )

        image = _parse_image(data["observation/image"])
        if self.model_type == _model.ModelType.PI0_FAST:
            image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
        elif self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        else:
            raise ValueError(f"Unsupported OpenPI model type: {self.model_type}")
        result = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                image_names[0]: image,
                image_names[1]: np.zeros_like(image),
                image_names[2]: np.zeros_like(image),
            },
            "image_mask": {
                image_names[0]: np.True_,
                image_names[1]: np.False_,
                image_names[2]: np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != self.joint_dim:
                raise ValueError(
                    "Expected SO-101 actions shape "
                    f"(N, {self.joint_dim}), got {actions.shape}."
                )
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = str(prompt)
        return result
