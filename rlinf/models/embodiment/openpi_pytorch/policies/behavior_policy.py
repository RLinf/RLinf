# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained BEHAVIOR input/output transforms (pi05 eval).

Vendored from ``rlinf/models/embodiment/openpi/policies/behavior_policy.py`` with
the installed-``openpi`` dependencies removed: the upstream ``transforms.DataTransformFn``
base and ``openpi.models.model.ModelType`` are replaced by a local lightweight
callable base, and the BEHAVIOR pi05 branch is kept exactly (state extraction,
image key mapping, and the 23-dim action slice). Logic is byte-identical to the
old transforms (verified by a cross-check test against the old policy).
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

# Keep a local copy of the R1Pro proprio slices to avoid importing omnigibson
# in rollout worker init threads (omnigibson registers signal handlers at import time).
R1PRO_PROPRIO_INDICES = {
    "arm_left_qpos": np.s_[158:165],
    "gripper_left_qpos": np.s_[193:195],
    "arm_right_qpos": np.s_[197:204],
    "trunk_qpos": np.s_[236:240],
    "base_qvel": np.s_[253:256],
    "gripper_right_qpos": np.s_[232:234],
}


class DataTransformFn:
    """Minimal callable-transform base (replaces openpi.transforms.DataTransformFn)."""

    def __call__(self, data: dict) -> dict:  # pragma: no cover - interface only
        raise NotImplementedError


def extract_state_from_proprio(proprio_data: np.ndarray) -> np.ndarray:
    """Extract 23-dim policy state from the full proprio vector."""
    base_qvel = proprio_data[..., R1PRO_PROPRIO_INDICES["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., R1PRO_PROPRIO_INDICES["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., R1PRO_PROPRIO_INDICES["arm_left_qpos"]]  # 7
    arm_right_qpos = proprio_data[..., R1PRO_PROPRIO_INDICES["arm_right_qpos"]]  # 7
    left_gripper_width = proprio_data[
        ..., R1PRO_PROPRIO_INDICES["gripper_left_qpos"]
    ].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[
        ..., R1PRO_PROPRIO_INDICES["gripper_right_qpos"]
    ].sum(axis=-1, keepdims=True)  # 1
    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            arm_right_qpos,
            left_gripper_width,  # gripper rearranged from 21 to 14 to match the action space
            right_gripper_width,
        ],
        axis=-1,
    )


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.shape[0] == 2 and image.shape[1] == 3:
        image = einops.rearrange(image, "n c h w -> n h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BehaviorInputs(DataTransformFn):
    """Map a BEHAVIOR observation dict to the model input dict (pi05 branch)."""

    extract_state_from_proprio: bool = True
    use_all_wrist_images: bool = True

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])  # [h, w, c]

        # Handle both stacked wrist images and separate keys (BEHAVIOR v2.1).
        if "observation/wrist_image" in data:
            wrist_image = _parse_image(data["observation/wrist_image"])  # [2, h, w, c]
            left_wrist = wrist_image[0, ...]
            right_wrist = wrist_image[1, ...]
        else:
            left_wrist = _parse_image(data["observation/left_wrist_image"])
            right_wrist = _parse_image(data["observation/right_wrist_image"])

        state = (
            extract_state_from_proprio(data["observation/state"])
            if self.extract_state_from_proprio
            else data["observation/state"]
        )

        inputs = {
            "state": state[:32],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.use_all_wrist_images
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class BehaviorOutputs(DataTransformFn):
    """Slice model actions back to the BEHAVIOR env action dimension."""

    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
