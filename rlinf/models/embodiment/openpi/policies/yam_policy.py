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

"""YAM joint angles stay in radians and grippers use 0=closed, 1=open.

ModelTransformFactory pads the 14-D state/actions after normalization.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Map top/left/right RGB and left-then-right absolute qpos14 to OpenPI."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(
                f"Expected finite YAM state with shape (14,), got {state.shape}."
            )
        side_views = data.get("observation/extra_view_image")
        if side_views is None:
            side_views = (
                data["observation/extra_view_image-0"],
                data["observation/extra_view_image-1"],
            )
        elif isinstance(side_views, np.ndarray) and side_views.ndim != 4:
            raise ValueError(
                "Expected stacked extra_view_image with shape (2, H, W, C), "
                f"got {side_views.shape}."
            )
        if len(side_views) != 2:
            raise ValueError("YAM requires two extra views ordered left/right.")
        images = {
            "base_0_rgb": _parse_image(data["observation/image"]),
            "left_wrist_0_rgb": _parse_image(side_views[0]),
            "right_wrist_0_rgb": _parse_image(side_views[1]),
        }
        inputs = {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
        }
        if "actions" in data:
            actions = np.array(data["actions"], dtype=np.float32, copy=True)
            if (
                actions.ndim != 2
                or actions.shape[-1] != 14
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    f"Expected finite YAM actions with shape (horizon, 14), got {actions.shape}."
                )
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Return absolute qpos14 actions from the unnormalized model output."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if (
            actions.ndim != 2
            or actions.shape[-1] < 14
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                f"Expected finite model actions with at least 14 values, got {actions.shape}."
            )
        return {"actions": actions[:, :14]}
