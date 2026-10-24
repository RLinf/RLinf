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

"""Bench2Dex absolute active-joint targets and four RGB views."""

import dataclasses

import numpy as np
from openpi import transforms

IMAGE_KEYS = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
CAMERA_NAMES = (
    "cam_stereo_left",
    "cam_stereo_right",
    "cam_wrist_left",
    "cam_wrist_right",
)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class Bench2DexInputs(transforms.DataTransformFn):
    """Keep runtime active-joint order and the official stereo/wrist order."""

    action_dim: int = 26

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (self.action_dim,):
            raise ValueError(
                f"Expected {self.action_dim} active joints, got {state.shape}."
            )
        extra = data["observation/extra_view_image"]
        if len(extra) != 3:
            raise ValueError(
                f"Expected stereo-right and two wrist views, got {len(extra)}."
            )
        images = [_parse_image(data["observation/image"])]
        images.extend(_parse_image(view) for view in extra)
        result = {
            "state": state,
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict.fromkeys(IMAGE_KEYS, np.True_),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected {self.action_dim} action joints, got {actions.shape}."
                )
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Bench2DexOutputs(transforms.DataTransformFn):
    """Return absolute active-joint positions without scaling or delta conversion."""

    action_dim: int = 26

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < self.action_dim:
            raise ValueError(
                f"Expected at least {self.action_dim} action joints, got {actions.shape}."
            )
        return {"actions": actions[..., : self.action_dim]}
