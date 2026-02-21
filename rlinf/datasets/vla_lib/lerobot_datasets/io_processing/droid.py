# Copyright 2025 The RLinf Authors.
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
DROID-specific data transforms that match the original OpenPI implementation.

This module provides transforms specifically designed for DROID datasets,
converting them to the standard format expected by PI0 models.
"""

from typing import Any

import numpy as np
import torch

# from ..transforms import DataTransformFn
from .. import transforms as _transforms


class DroidInputs(_transforms.DataTransformFn):
    """DROID input transforms that match OpenPI's droid_policy.DroidInputs."""

    def __init__(self, model_type: str = "pi05"):
        """
        Args:
            model_type: Model type ("pi0", "pi05", "pi0_fast") for image mask handling.
        """
        self.model_type = model_type.lower()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform DROID data format to PyTorch tensor format.

        Expected input format (after repack):
        - observation/image: main exterior camera image (mapped from observation/exterior_image_1_left)
        - observation/wrist_image: wrist camera image (mapped from observation/wrist_image_left)
        - observation/joint_position: joint positions
        - observation/gripper_position: gripper position
        - actions: action sequence (mapped from action)
        - prompt: task description
        """
        # Process state - concatenate joint and gripper positions like OpenPI
        state = self._process_state(data)

        # Process images - handle both tensor and numpy inputs
        base_image = self._process_image(data["observation/image"])
        wrist_image = self._process_image(data["observation/wrist_image"])

        # Create right wrist placeholder (same device as base image)
        right_wrist_image = torch.zeros_like(base_image)

        # Create inputs dict in format expected by PI0 data collator
        # Use OpenPI's camera naming convention with dictionaries
        # For PI0_FAST, all masks True; for PI0/PI05, padding images masked False
        right_mask = np.True_ if self.model_type == "pi0_fast" else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Handle actions - pad to model action dimension if needed
        if "actions" in data:
            actions = self._process_actions(data["actions"])
            inputs["actions"] = actions

        # PI0 collator expects 'task' key, not 'prompt'
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Pass through RL-specific fields (for value learning)
        for rl_key in [
            "return",
            "reward",
            "done",
            "is_failed",
            "task_idx",
            "subtask_idx",
        ]:
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

        # Pass through RL keys (rewards, returns, action chunks, padding flags)
        for key in data:
            if (
                key.startswith(("reward_", "return_", "action_", "history_"))
                and key not in inputs
            ):
                inputs[key] = data[key]
            elif key.endswith("_is_pad") and key not in inputs:
                inputs[key] = data[key]

        return inputs

    def _process_state(self, data: dict[str, Any]) -> torch.Tensor:
        """Process state by concatenating joint and gripper positions, matching OpenPI logic."""
        # Try to get joint and gripper positions
        joint_pos = data["observation/joint_position"]
        gripper_pos = data["observation/gripper_position"]

        # Convert to tensors if needed
        if joint_pos is not None and not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        if gripper_pos is not None and not isinstance(gripper_pos, torch.Tensor):
            gripper_pos = torch.tensor(gripper_pos, dtype=torch.float32)

        # Concatenate joint and gripper positions like OpenPI does
        state = torch.cat([joint_pos, gripper_pos], dim=-1)

        return state

    def _process_actions(self, actions) -> torch.Tensor:
        """Process actions tensor with device-aware padding."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)

        return actions

    def _process_image(self, img) -> torch.Tensor:
        """
        Process image following OpenPI's _parse_image logic but using PyTorch.

        Keeps images as uint8[0,255] and ensures CHW format, matching OpenPI's behavior.
        The conversion to float32[-1,1] will happen later in the image processor.
        """
        # Convert to tensor if needed
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # Follow OpenPI's logic: if floating point, scale to [0,255] and convert to uint8
        if img.dtype.is_floating_point:
            img = (255 * img).to(torch.uint8)

        # Ensure CHW format (standard PyTorch convention) but keep as uint8
        if len(img.shape) == 3:
            if img.shape[-1] == 3:  # HWC -> CHW
                img = img.permute(2, 0, 1)
            # If already CHW (shape[0] == 3), keep as is

        # Ensure uint8 dtype (matching OpenPI's approach)
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        return img


class DroidOutputs(_transforms.DataTransformFn):
    """DROID output transforms that match OpenPI's droid_policy.DroidOutputs."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform model outputs back to DROID format.

        Extract the first 8 actions (matching OpenPI behavior) and ensure proper tensor format.
        """
        actions = data["actions"]
        if isinstance(actions, torch.Tensor):
            # Keep as tensor but extract first 8 dimensions (matching OpenPI)
            actions = actions[..., :8]
        else:
            # Convert to tensor if needed
            actions = torch.tensor(actions, dtype=torch.float32)[..., :8]

        return {"actions": actions}
