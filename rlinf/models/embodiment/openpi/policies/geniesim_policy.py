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

"""Input/output transforms for the GenieSim G2 dual-arm pi05 policy.

Adapted from ``openpi_jax``'s ``openpi.policies.go1_policy``. The environment
supplies canonical arm/gripper state; dataset-side joint canonicalization is
not part of inference. RL fine-tuning also transforms ground-truth actions.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms

# Canonical G2 dual-arm layout: 14 arm joints followed by 2 grippers. Everything
# past dim 16 is waist/padding and is masked to zero (``include_waist=False``).
GENIESIM_CANONICAL_DIM = 16
GENIESIM_NUM_ARM_JOINTS = 14

# The two gripper columns of the canonical layout.
GENIESIM_GRIPPER_SLICE = (GENIESIM_NUM_ARM_JOINTS, GENIESIM_CANONICAL_DIM)


def flip_geniesim_gripper(arr: np.ndarray) -> np.ndarray:
    """Negate gripper columns in place between positive GenieSim opening angles
    and the checkpoint's negative-opening convention. The map is its own inverse."""
    lo, hi = GENIESIM_GRIPPER_SLICE
    if arr.shape[-1] < hi:
        raise ValueError(
            f"GenieSim gripper flip needs at least {hi} columns, got {arr.shape}"
        )
    arr[..., lo:hi] *= -1
    return arr


def _parse_image(image) -> np.ndarray:
    """Normalize one frame to HWC uint8, matching ``Go1Inputs``' handling."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GenieSimInputs(transforms.DataTransformFn):
    """Map canonical G2 state and three camera views to pi05 inputs."""

    action_dim: int
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    # When the env cannot supply a wrist view, feed a zero frame with its mask
    # cleared rather than dropping the slot -- the model always expects three.
    require_wrist_images: bool = True
    # Negate the gripper columns on the way in. See ``flip_geniesim_gripper``:
    # GenieSim's positive opening angle is the opposite sign of the ``g2a_sim``
    # convention this checkpoint was trained on.
    flip_gripper_sign: bool = True

    def _apply_mask(self, arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return arr
        width = min(arr.shape[-1], len(mask))
        arr[..., :width][..., mask[:width]] = 0
        return arr

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32).copy()
        if self.flip_gripper_sign:
            state = flip_geniesim_gripper(state)
        state = self._apply_mask(state, self.state_mask)
        state = transforms.pad_to_dim(state, self.action_dim)
        state = state.squeeze()

        base_image = _parse_image(data["observation/image"])

        # ``extra_view_image`` is [2, H, W, C] = (left wrist, right wrist), the
        # layout produced by ``GenieSimAdapter``'s extra_image_keys ordering.
        left_wrist = right_wrist = None
        if data.get("observation/extra_view_image") is not None:
            extra = np.asarray(data["observation/extra_view_image"])
            if extra.shape[0] >= 1:
                left_wrist = _parse_image(extra[0])
            if extra.shape[0] >= 2:
                right_wrist = _parse_image(extra[1])
        if data.get("observation/wrist_image") is not None:
            left_wrist = _parse_image(data["observation/wrist_image"])

        if self.require_wrist_images and (left_wrist is None or right_wrist is None):
            raise ValueError(
                "GenieSim pi05 was trained with head + both wrist cameras; got "
                f"left={left_wrist is not None} right={right_wrist is not None}. "
                "Set require_wrist_images=False to run with zero-filled views."
            )

        def _slot(img):
            return img if img is not None else np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": _slot(left_wrist),
                "right_wrist_0_rgb": _slot(right_wrist),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_(left_wrist is not None),
                "right_wrist_0_rgb": np.bool_(right_wrist is not None),
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            if self.flip_gripper_sign:
                actions = flip_geniesim_gripper(actions)
            actions = self._apply_mask(actions, self.action_mask)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GenieSimOutputs(transforms.DataTransformFn):
    """Trim padded actions and restore the GenieSim gripper sign.

    Runs after AbsoluteActions; gripper columns remain absolute throughout."""

    output_dim: int = GENIESIM_CANONICAL_DIM
    flip_gripper_sign: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.output_dim]).copy()
        if self.flip_gripper_sign:
            actions = flip_geniesim_gripper(actions)
        return {"actions": actions}
