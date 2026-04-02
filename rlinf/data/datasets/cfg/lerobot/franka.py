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

"""
Franka-specific data transforms for the internal dataset framework.

Adapted from rlinf/models/embodiment/openpi/policies/franka_policy.py,
replacing external openpi dependencies with internal PyTorch transforms.

Key differences from the OpenPI version:
- Uses PyTorch tensors instead of numpy for image processing
- State/action padding removed (handled by PadStatesAndActions in model transforms)
- RL field passthrough added for value learning pipeline
- Bug fix: action_train_with_rotation_6d now properly affects action slicing
  in FrankaEEOutputs (original always sliced to 7, ignoring the flag)
"""

from typing import Any

import numpy as np
import torch

from . import transforms as _transforms


def _parse_image(image) -> torch.Tensor:
    """Parse image to uint8 torch tensor. Adapted from franka_policy._parse_image."""
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(np.asarray(image))
    if image.dtype.is_floating_point:
        image = (255 * image).to(torch.uint8)
    if image.shape[0] == 3:  # CHW -> HWC (match original convention)
        image = image.permute(1, 2, 0)
    return image


class FrankaEEInputs(_transforms.DataTransformFn):
    """Franka input transforms, adapted from franka_policy.FrankaEEInputs.

    Converts Franka dataset format to model input format. Only 1 real camera
    (base_0_rgb); padding cameras are zero-filled.

    Differences from original OpenPI version:
    - model_type is a string ("pi0", "pi05", "pi0_fast") instead of ModelType enum
    - No action_dim / pad_to_dim here — PadStatesAndActions handles padding later
    - action_train_with_rotation_6d properly validates action shape (N, 10)
    - For non-rotation_6d Franka datasets, accepts either:
      * 7D actions [x,y,z,rx,ry,rz,gripper]
      * 6D actions [x,y,z,rx,ry,rz], in which case gripper=0 is appended
    - RL fields (return, reward, done) and padding masks passed through for value learning
    """

    def __init__(
        self,
        mask_padding: bool = True,
        model_type: str = "pi05",
        action_train_with_rotation_6d: bool = False,
    ):
        self.mask_padding = mask_padding
        self.model_type = model_type
        self.action_train_with_rotation_6d = action_train_with_rotation_6d

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state).float()
        assert state.ndim == 1, f"Expected 1D state, got shape {tuple(state.shape)}"

        base_image = _parse_image(data["observation/image"])

        # Camera layout and masking differ by model type
        model_type_lower = self.model_type.lower()
        if model_type_lower in ("pi0", "pi05"):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (
                base_image,
                torch.zeros_like(base_image),
                torch.zeros_like(base_image),
            )
            image_masks = (np.True_, np.False_, np.False_)
        elif model_type_lower == "pi0_fast":
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (
                base_image,
                torch.zeros_like(base_image),
                torch.zeros_like(base_image),
            )
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = data["actions"]
            if hasattr(actions, "shape") and len(actions.shape) == 2:
                if self.action_train_with_rotation_6d:
                    assert actions.shape[-1] == 10, (
                        f"Expected actions shape (N, 10), got {actions.shape}"
                    )
                else:
                    assert actions.shape[-1] == 7, (
                        f"Expected actions shape (N, 7), got {actions.shape}"
                    )
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        # RL field passthrough for value learning pipeline
        for rl_key in ["return", "reward", "done"]:
            if rl_key in data:
                value = data[rl_key]
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        value = value.reshape(1)
                    inputs[rl_key] = (
                        torch.from_numpy(value).float()
                        if value.dtype in [np.float32, np.float64]
                        else torch.from_numpy(value)
                    )
                elif isinstance(value, torch.Tensor):
                    if value.ndim == 0:
                        value = value.unsqueeze(0)
                    inputs[rl_key] = value
                else:
                    inputs[rl_key] = value

        # Pass through padding masks (e.g. reward_is_pad) for value learning
        for key in data:
            if key.endswith("_is_pad") and key not in inputs:
                inputs[key] = data[key]

        return inputs


class FrankaEEOutputs(_transforms.DataTransformFn):
    """Franka output transforms, adapted from franka_policy.FrankaEEOutputs.

    Bug fix: When action_train_with_rotation_6d=True, slices to 10 dims
    [x,y,z,rot6d(6),gripper]. The original always sliced to 7, ignoring the flag.
    """

    def __init__(self, action_train_with_rotation_6d: bool = False):
        self.action_train_with_rotation_6d = action_train_with_rotation_6d
        self.action_dim = 10 if action_train_with_rotation_6d else 7

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" in data:
            actions = data["actions"]
            if isinstance(actions, np.ndarray):
                actions = actions[..., : self.action_dim]
            elif isinstance(actions, torch.Tensor):
                actions = actions[..., : self.action_dim]
            else:
                actions = np.asarray(actions)[..., : self.action_dim]
            return {"actions": actions}
        return data
