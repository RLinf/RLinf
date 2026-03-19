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
"""Policy transforms for real-world Franka deployment with multi-camera and
configurable state dimension selection.

Compared to the generic ``FrankaEEInputs`` (which expects pre-processed 7D
state and a single image), this module handles:

* **State selection** — extracts 7D from the 19D RealWorldEnv state vector
  (alphabetically sorted: gripper_position, tcp_force, tcp_pose, tcp_torque,
  tcp_vel).  Configurable via ``state_indices``.
* **Multi-camera Pi0 slot mapping** — flexibly maps observation keys to the
  three Pi0 image slots via ``pi0_slot_keys``.  Works with both the rollout
  worker path (``observation/extra_view_images``) and the SFT training path
  (``observation/extra_image_*``).
"""

import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def make_realworld_example() -> dict:
    """Creates a random input example for the real-world Franka policy."""
    return {
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/extra_image_0": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "observation/extra_image_1": np.random.randint(
            256, size=(128, 128, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(7).astype(np.float32),
        "prompt": "Pick and place the object.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RealworldOutputs(transforms.DataTransformFn):
    """Converts model outputs back to 7D actions [dx, dy, dz, drx, dry, drz, gripper]."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class RealworldInputs(transforms.DataTransformFn):
    """Converts inputs to the format expected by Pi0 for real-world Franka tasks.

    Handles both the rollout worker path (where ``obs_processor`` produces
    ``observation/extra_view_images`` as a stacked array) and the SFT training
    path (where ``RepackTransform`` produces individual
    ``observation/extra_image_*`` keys from LeRobot dataset columns).
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    # Indices to select from the full state vector to get 7D.
    # Default picks tcp_pose[6] + gripper[1] from the 19D alphabetically-sorted
    # RealWorldEnv state (gripper_position, tcp_force, tcp_pose, tcp_torque, tcp_vel).
    # Set to None to skip selection (when state is already 7D).
    state_indices: tuple[int, ...] | None = (4, 5, 6, 7, 8, 9, 0)

    # Mapping of the three Pi0 image slots to observation keys.
    # The three entries correspond to (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb).
    # Use None to pad a slot with zeros and mask it out.
    # NOTE: must match the pi0_slot_keys in the training dataconfig
    # (pi0_realworld_pnp in dataconfig/__init__.py):
    #   slot 0 (base_0_rgb)       → extra_image_0
    #   slot 1 (left_wrist_0_rgb) → image  (main camera)
    #   slot 2 (right_wrist_0_rgb)→ extra_image_1
    pi0_slot_keys: tuple[str | None, str | None, str | None] = (
        "observation/extra_image_0",
        "observation/image",
        "observation/extra_image_1",
    )

    def __call__(self, data: dict) -> dict:
        # --- State dimension selection ---
        state = np.asarray(data["observation/state"])
        if self.state_indices is not None and state.shape[-1] != len(
            self.state_indices
        ):
            state = state[..., list(self.state_indices)]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = transforms.pad_to_dim(state, self.action_dim)

        # --- Unpack extra_view_images into individual keys ---
        # In the rollout worker path, obs_processor produces a stacked
        # [N_EXTRA, H, W, C] array under "observation/extra_view_images".
        # Split it into observation/extra_image_0, _1, ... so that
        # pi0_slot_keys can reference them uniformly in both paths.
        if "observation/extra_view_images" in data:
            extra = data["observation/extra_view_images"]
        elif "observation/extra_view_image" in data:
            extra = data["observation/extra_view_image"]
        else:
            extra = None
        if extra is not None:
            extra = np.asarray(extra)
            if extra.ndim == 5:
                for idx in range(extra.shape[1]):
                    data.setdefault(f"observation/extra_image_{idx}", extra[:, idx])
            elif extra.ndim == 4:
                for idx in range(extra.shape[0]):
                    data.setdefault(f"observation/extra_image_{idx}", extra[idx])
            elif extra.ndim == 3:
                data.setdefault("observation/extra_image_0", extra)

        # --- Resolve each Pi0 image slot ---
        slot_images: list[np.ndarray | None] = []
        for key in self.pi0_slot_keys:
            raw = data[key] if key is not None and key in data else None
            slot_images.append(_parse_image(raw) if raw is not None else None)

        ref = next((img for img in slot_images if img is not None), None)
        if ref is None:
            raise ValueError("At least one image must be provided.")

        resolved = tuple(
            img if img is not None else np.zeros_like(ref) for img in slot_images
        )
        masks = tuple(np.True_ if img is not None else np.False_ for img in slot_images)

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            image_masks = masks
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, resolved, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            assert actions.ndim == 2 and actions.shape[-1] == 7, (
                f"Expected actions shape (N, 7), got {actions.shape}"
            )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
