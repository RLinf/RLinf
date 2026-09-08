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

"""OpenPI transforms for the single-arm XSquare Turtle2 setup."""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

TURTLE_ACTION_DIM = 6
_WRIST_IMAGE_KEYS = ("left_wrist_0_rgb", "right_wrist_0_rgb")


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert one CHW or HWC image to uint8 HWC."""
    image = np.asarray(image).squeeze()
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D CHW or HWC image, got {image.shape}.")
    if image.shape[-1] in (1, 3, 4):
        pass
    elif image.shape[0] in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    else:
        raise ValueError(f"Cannot infer image layout from shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or np.nanmax(image) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _parse_extra_views(extra_views: np.ndarray | None) -> list[np.ndarray]:
    """Return zero, one, or two wrist views from common stacked layouts."""
    if extra_views is None:
        return []
    extra = np.asarray(extra_views)
    if extra.ndim == 4:
        # RealWorldEnv emits NCHW or NHWC after the batch dimension is removed.
        return [_parse_image(extra[index]) for index in range(min(extra.shape[0], 2))]
    if extra.ndim == 3 and extra.shape[0] in (6, 8):
        # Accept legacy channel-concatenated CHW tensors (2*C, H, W).
        channels_per_view = extra.shape[0] // 2
        return [
            _parse_image(
                extra[index * channels_per_view : (index + 1) * channels_per_view]
            )
            for index in range(2)
        ]
    if extra.ndim == 3:
        return [_parse_image(extra)]
    raise ValueError(
        f"Expected extra Turtle2 views as NHWC, NCHW, HWC, or CHW; got {extra.shape}."
    )


@dataclasses.dataclass(frozen=True)
class TurtleInputs(transforms.DataTransformFn):
    """Map Turtle2's three cameras and 6-D state to OpenPI inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_views = _parse_extra_views(data.get("observation/extra_view_image"))
        for index, key in enumerate(_WRIST_IMAGE_KEYS):
            if index < len(extra_views):
                images[key] = extra_views[index]
                image_masks[key] = np.True_
            else:
                images[key] = np.zeros_like(base_image)
                image_masks[key] = np.False_

        state = np.asarray(data["observation/state"])
        if state.shape != (TURTLE_ACTION_DIM,):
            raise ValueError(
                f"Expected Turtle2 state shape ({TURTLE_ACTION_DIM},), got {state.shape}."
            )

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != TURTLE_ACTION_DIM:
                raise ValueError(
                    "Expected Turtle2 actions with shape (horizon, 6), "
                    f"got {actions.shape}."
                )
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class TurtleOutputs(transforms.DataTransformFn):
    """Slice the 6-DoF Turtle2 action from the padded OpenPI output."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < TURTLE_ACTION_DIM:
            raise ValueError(
                "Expected OpenPI actions with shape (horizon, >=6), "
                f"got {actions.shape}."
            )
        return {"actions": actions[:, :TURTLE_ACTION_DIM]}
