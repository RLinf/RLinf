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
"""Policy transforms for dual-Franka rot6d SFT.

state[:20] = [L_grip, R_grip, L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)].
actions[:20] = per-arm [xyz(3), rot6d(6), grip(1)] (training layout).
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

_STATE_SLICE_DIM = 20  # padded to action_dim downstream

# Asserted so a rig rename fails loud instead of swapping camera meanings.
_EXPECTED_EXTRA_VIEW_ORDER = ("base_0_rgb", "right_wrist_0_rgb")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _rearrange_state(state: np.ndarray) -> np.ndarray:
    # RealWorldEnv._wrap_obs concatenates the env state dict in alphabetical
    # key order, so the rot6d env emits [L_grip, R_grip, L_xyz, L_rot6d,
    # R_xyz, R_rot6d] (gripper_position < tcp_pose_rot6d). Reorder to
    # training-time [L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip] so
    # RigidBodyDeltaActions sees state and actions in identical slot order.
    s = np.asarray(state)[..., :_STATE_SLICE_DIM]
    return np.concatenate(
        [s[..., 2:11], s[..., 0:1], s[..., 11:20], s[..., 1:2]], axis=-1
    )


def _extract_extra_views(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (base, right_wrist) from stacked (inference) or split (training)."""
    stacked = data.get("observation/extra_view_image")
    if stacked is not None:
        names = data.get("observation/extra_view_image_names")
        if names is not None and tuple(names) != _EXPECTED_EXTRA_VIEW_ORDER:
            raise AssertionError(
                f"extra-view camera order drifted: got {tuple(names)}, "
                f"expected {_EXPECTED_EXTRA_VIEW_ORDER}."
            )
        extra = np.asarray(stacked)
        return _parse_image(extra[0]), _parse_image(extra[1])
    return (
        _parse_image(data["observation/extra_view_image-0"]),
        _parse_image(data["observation/extra_view_image-1"]),
    )


@dataclasses.dataclass(frozen=True)
class DualFrankaRot6dInputs(transforms.DataTransformFn):
    """Feeds dual-Franka rot6d observations into pi0 / pi05."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = _rearrange_state(data["observation/state"])
        state = transforms.pad_to_dim(state, self.action_dim)

        left_wrist_image = _parse_image(data["observation/image"])
        base_image, right_wrist_image = _extract_extra_views(data)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaRot6dOutputs(transforms.DataTransformFn):
    """Recover 20-d rot6d action from padded model output."""

    output_action_dim: int = 20

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
