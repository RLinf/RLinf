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
"""OpenPI input/output transforms for the SO101 LeRobot dataset.

The SO101 datasets collected by ``CollectEpisode`` / ``LeRobotDatasetWriter``
have:

* ``state``        — float32, shape ``(6,)``: 5 arm joint angles + 1 gripper.
* ``actions``      — float32, shape ``(6,)``: 5 absolute joint targets + 1
  gripper target (all in degrees).
* ``image``        — uint8,   shape ``(128, 128, 3)``: main front camera.
* ``extra_view_image`` — uint8, shape ``(128, 128, 3)``: optional wrist /
  side view; the second camera in the auto-generated LeRobot feature schema.
"""

import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


SO101_STATE_DIM = 6
SO101_ACTION_DIM = 6


def make_so101_example() -> dict:
    """Random input example for the SO101 policy (for warmup / shape checks)."""
    return {
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/extra_view_image": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(SO101_STATE_DIM),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Slice the first ``SO101_ACTION_DIM`` columns out of the padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :SO101_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Pack SO101 LeRobot samples into the format expected by pi0 / pi05."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        assert state.shape == (SO101_STATE_DIM,), (
            f"Expected state shape ({SO101_STATE_DIM},), got {tuple(state.shape)}"
        )

        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])
        extra_view_image = _parse_image(
            data.get("observation/extra_view_image", np.zeros_like(base_image))
        )

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, extra_view_image, np.zeros_like(base_image))
            # Mask the unused third slot for pi0; both real cameras are real.
            image_masks = (np.True_, np.True_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, extra_view_image, np.zeros_like(base_image))
            # pi0-FAST doesn't use padding masks.
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            assert actions.ndim == 2 and actions.shape[-1] == SO101_ACTION_DIM, (
                f"Expected actions shape (N, {SO101_ACTION_DIM}), got {actions.shape}"
            )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs
