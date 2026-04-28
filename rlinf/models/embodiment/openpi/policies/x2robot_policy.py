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

import dataclasses
from typing import ClassVar

import einops
import numpy as np
from openpi import transforms


def _convert_image(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.shape[-1] != 3:
        img = einops.rearrange(img, "c h w -> h w c")
    if img.shape[-1] != 3:
        raise ValueError(f"Image must have 3 channels, got shape {img.shape}.")
    return img


@dataclasses.dataclass(frozen=True)
class X2RobotInputs(transforms.DataTransformFn):
    action_dim: int = 14
    only_right_obs: bool = False
    random_pos_offset: float = 0.0

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    )

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        missing = set(self.EXPECTED_CAMERAS) - set(images)
        if missing:
            raise ValueError(
                f"Images must contain {self.EXPECTED_CAMERAS}, "
                f"missing {tuple(sorted(missing))}."
            )

        state = transforms.pad_to_dim(np.asarray(data["state"]), self.action_dim)
        processed_images = {
            name: _convert_image(images[name]) for name in self.EXPECTED_CAMERAS
        }
        inputs = {
            "image": {
                "base_0_rgb": processed_images["face_view"],
                "left_wrist_0_rgb": processed_images["left_wrist_view"],
                "right_wrist_0_rgb": processed_images["right_wrist_view"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(
                np.asarray(data["actions"]), self.action_dim
            )
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = data["actions_is_pad"]

        if self.random_pos_offset > 0.0:
            pos_offset = (np.random.rand(3) * 2 - 1.0) * self.random_pos_offset
            inputs["state"][..., :3] += pos_offset
            inputs["state"][..., 7:10] += pos_offset
            if "actions" in inputs:
                inputs["actions"][..., :3] += pos_offset
                inputs["actions"][..., 7:10] += pos_offset

        if self.only_right_obs:
            inputs["image_mask"]["base_0_rgb"] = np.False_
            inputs["image_mask"]["left_wrist_0_rgb"] = np.False_
            inputs["state"][..., :7] = 0.0
            if "actions" in inputs:
                inputs["actions"][..., :7] = 0.0

        return inputs


@dataclasses.dataclass(frozen=True)
class X2RobotOutputs(transforms.DataTransformFn):
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim])
        return {"actions": actions}
