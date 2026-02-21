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
Libero-specific data transforms that match the original OpenPI implementation.

This module provides transforms specifically designed for Libero datasets,
converting them to the standard format expected by PI0 models.
"""

from typing import Any

import numpy as np
import torch

# from ..transforms import DataTransformFn
from .. import transforms as _transforms


class LiberoInputs(_transforms.DataTransformFn):
    """Libero input transforms for preprocessing Libero dataset format.

    This matches OpenPI's libero_policy.LiberoInputs behavior.
    The repack transform in config.py converts LeRobot keys (e.g., "image")
    to standard keys (e.g., "observation/image") before this transform runs.
    """

    def __init__(self, mask_padding: bool = True, model_type: str = "pi05"):
        self.mask_padding = mask_padding
        self.model_type = model_type

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform Libero data format to model input format.

        Expected input format (after repack):
        - observation/image: main camera image (tensor)
        - observation/wrist_image: wrist camera image (tensor)
        - observation/state: robot proprioceptive state (tensor)
        - actions: action sequence (tensor)
        - prompt: task description (string)

        This matches OpenPI's libero_policy.py:52-53 which reads from
        data["observation/image"] and data["observation/wrist_image"].
        """
        # Process images - handle both tensor and numpy inputs
        # Keys match OpenPI's libero_policy.LiberoInputs expectations
        base_image = self._process_image(data["observation/image"])
        wrist_image = self._process_image(data["observation/wrist_image"])

        # Create right wrist placeholder (same device as base image)
        right_wrist_image = torch.zeros_like(base_image)

        # Create inputs dict matching OpenPI's libero_policy.py output format EXACTLY
        # OpenPI uses "image" (singular) and "image_mask" (singular)
        inputs = {
            "state": data["observation/state"],
            # Image dict with standard camera keys
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            # Mask dict - matching OpenPI's logic for padding images
            # For PI0_FAST, all masks are True; for PI0/PI05, padding images are masked False
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type.lower() == "pi0_fast"
                else np.False_,
            },
        }

        # Handle actions - pad to model action dimension if needed
        if "actions" in data:
            inputs["actions"] = data["actions"]

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
        # Next observation is handled by RL dataset (applies same VLA transforms)
        for key in data:
            if (
                key.startswith(("history_", "reward_", "return_", "action_"))
                and key not in inputs
            ):
                inputs[key] = data[key]
            elif key.endswith("_is_pad") and key not in inputs:
                inputs[key] = data[key]

        # Pass through ECOT-required indices for reasoning transforms
        for ecot_key in ["episode_index", "frame_index"]:
            if ecot_key in data:
                inputs[ecot_key] = data[ecot_key]

        return inputs

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


class LiberoOutputs(_transforms.DataTransformFn):
    """Libero output transforms - extracts correct action dimensions."""

    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform model outputs back to Libero format.

        Extract the first 7 actions (remove any padding).
        Handles both numpy arrays and tensors.
        """
        if "actions" in data:
            actions = data["actions"]
            # Extract first action_dim dimensions (rest is padding)
            # Keep the same type as input (numpy or tensor)
            if isinstance(actions, np.ndarray):
                actions = actions[..., : self.action_dim]
            elif isinstance(actions, torch.Tensor):
                actions = actions[..., : self.action_dim]
            else:
                actions = np.asarray(actions)[..., : self.action_dim]

            return {"actions": actions}

        return data
