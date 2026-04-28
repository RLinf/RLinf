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


def _select_wrist_views(extra_view_images):
    extra_view_images = np.asarray(extra_view_images)
    if extra_view_images.ndim >= 5:
        if extra_view_images.shape[1] < 2:
            raise ValueError(
                "x2robot requires extra_view_images with two wrist views, "
                f"got shape {extra_view_images.shape}."
            )
        return extra_view_images[:, 0], extra_view_images[:, 1]
    if extra_view_images.shape[0] < 2:
        raise ValueError(
            "x2robot requires extra_view_images with two wrist views, "
            f"got shape {extra_view_images.shape}."
        )
    return extra_view_images[0], extra_view_images[1]


@dataclasses.dataclass(frozen=True)
class X2RobotInputs(transforms.DataTransformFn):
    action_dim: int = 14

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    )

    def __call__(self, data: dict) -> dict:
        if "images" in data:
            images = data["images"]
            missing = set(self.EXPECTED_CAMERAS) - set(images)
            if missing:
                raise ValueError(
                    f"Images must contain {self.EXPECTED_CAMERAS}, "
                    f"missing {tuple(sorted(missing))}."
                )
            face_view = images["face_view"]
            left_wrist_view = images["left_wrist_view"]
            right_wrist_view = images["right_wrist_view"]
            state = data["state"]
        else:
            if "observation/extra_view_image" not in data:
                raise ValueError(
                    "x2robot runtime inputs require observation/extra_view_image "
                    "with left/right wrist views."
                )
            left_wrist_view, right_wrist_view = _select_wrist_views(
                data["observation/extra_view_image"]
            )
            face_view = data["observation/image"]
            state = data["observation/state"]

        state = transforms.pad_to_dim(np.asarray(state), self.action_dim)
        processed_images = {
            "left_wrist_view": _convert_image(left_wrist_view),
            "face_view": _convert_image(face_view),
            "right_wrist_view": _convert_image(right_wrist_view),
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

        return inputs


@dataclasses.dataclass(frozen=True)
class X2RobotOutputs(transforms.DataTransformFn):
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim])
        return {"actions": actions}
