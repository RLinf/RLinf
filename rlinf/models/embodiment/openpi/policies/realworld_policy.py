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

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_realworld_example() -> dict:
    """Creates a random input example for the real-world Franka policy."""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "prompt": "Pick up the object and put it into another bin",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_wrist_images(images):
    """Parse wrist/extra_view images that may have shape [N_IMG, H, W, C] or [H, W, C].

    - [H, W, C]: left_wrist = image, right_wrist = zeros.
    - [1, H, W, C]: left_wrist = image[0], right_wrist = zeros.
    - [2, H, W, C]: left_wrist = image[0], right_wrist = image[1].
    """
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[0] >= 1:
        first = _parse_image(images[0])
        if images.shape[0] == 2:
            second = _parse_image(images[1])
        else:
            second = np.zeros_like(first)
        return first, second
    else:
        img = _parse_image(images) if images.ndim == 3 else _parse_image(images[0])
        return img, np.zeros_like(img)


@dataclasses.dataclass(frozen=True)
class RealworldInputs(transforms.DataTransformFn):
    """Converts inputs to the format expected by the model for real-world Franka.

    Dual view: observation/image + observation/wrist_image.
    State: 7-dim [tcp_pose(6) + gripper(1)].
    Action: 7-dim [delta_xyz(3) + delta_rpy(3) + gripper(1)].
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image, right_wrist_image = _parse_wrist_images(
            data["observation/extra_view_image"]
        )

        # Realworld env concatenates state dict alphabetically (19-dim):
        # gripper_position (1), tcp_force (3), tcp_pose (6), tcp_torque (3), tcp_vel (6).
        # pi0_realworld_pnp expects 7-dim: tcp_pose (6) + gripper (1); norm_stats are 7-dim.

        states = np.asarray(data["observation/state"])

        if states.shape[-1] == 19:
            _TCP_POSE_START, _TCP_POSE_END = 4, 10  # indices 4..9
            _GRIPPER_IDX = 0
            _STATE_7_INDICES = list(range(_TCP_POSE_START, _TCP_POSE_END)) + [
                _GRIPPER_IDX
            ]  # tcp_pose then gripper
            states = states[..., _STATE_7_INDICES]

        inputs = {
            "state": states,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealworldOutputs(transforms.DataTransformFn):
    """Converts model outputs back to dataset format: 7-dim actions [dx, dy, dz, drx, dry, drz, gripper]."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
