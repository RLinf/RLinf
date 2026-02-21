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
ALOHA-specific data transforms that match the original OpenPI implementation.

This module provides transforms specifically designed for ALOHA datasets,
converting them to the standard format expected by PI0 models.
"""

from typing import Any, ClassVar

import numpy as np
import torch

# from ..transforms import DataTransformFn
from .. import transforms as _transforms


class AlohaInputs(_transforms.DataTransformFn):
    """ALOHA input transforms that match OpenPI's aloha_policy.AlohaInputs."""

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __init__(self, adapt_to_pi: bool = True, model_type: str = "pi05"):
        """
        Args:
            adapt_to_pi: If true, this will convert the joint and gripper values from the standard
                        Aloha space to the space used by the pi internal runtime which was used to
                        train the base model.
            model_type: Model type ("pi0", "pi05", "pi0_fast") for consistency.
        """
        self.adapt_to_pi = adapt_to_pi
        self.model_type = model_type.lower()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform ALOHA data format to PyTorch tensor format.

        Expected input format (after repack):
        - state: robot proprioceptive state [14] (tensor)
        - images: dict with camera keys (cam_high, cam_low, cam_left_wrist, cam_right_wrist)
        - actions: action sequence (tensor)
        - prompt: task description (string)
        """
        # Decode ALOHA data (handle special ALOHA transformations)
        data = self._decode_aloha(data)

        # Process state - pad from 14 to action dimension like OpenPI
        state = self._process_state(data["state"])

        # Process images - handle ALOHA camera configuration
        images, image_masks = self._process_images(data["images"])

        # Create inputs dict in format expected by PI0 data collator
        inputs = {
            "state": state,
            "images": images,
            "image_masks": image_masks,
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

    def _decode_aloha(self, data: dict[str, Any]) -> dict[str, Any]:
        """Decode ALOHA data following OpenPI's _decode_aloha logic."""
        result = data.copy()

        # Process state with ALOHA-specific transformations
        if "state" in data:
            state = data["state"]
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            state = self._decode_state(state)
            result["state"] = state

        # Process images - convert from potential CHW to HWC and ensure uint8
        if "images" in data:
            images_dict = {}
            for name, img in data["images"].items():
                images_dict[name] = self._convert_image(img)
            result["images"] = images_dict

        return result

    def _decode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Decode state following OpenPI's _decode_state logic."""
        # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
        # dim sizes: [6, 1, 6, 1] = 14 total

        if self.adapt_to_pi:
            # Flip the joints using the joint flip mask
            joint_flip_mask = self._joint_flip_mask()
            state = joint_flip_mask * state

            # Reverse the gripper transformation that is being applied by the Aloha runtime
            # Grippers are at indices 6 and 13
            state[6] = self._gripper_to_angular(state[6])
            state[13] = self._gripper_to_angular(state[13])

        return state

    def _convert_image(self, img) -> torch.Tensor:
        """Convert image following OpenPI's convert_image logic."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # Convert to uint8 if using float images
        if img.dtype.is_floating_point:
            img = (255 * img).to(torch.uint8)

        # Convert from CHW to HWC for processing, then back to CHW
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
            img = img.permute(1, 2, 0)  # CHW -> HWC

        # Ensure uint8 dtype
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        # Convert back to CHW for PyTorch convention
        if len(img.shape) == 3 and img.shape[-1] == 3:  # HWC -> CHW
            img = img.permute(2, 0, 1)

        return img

    def _process_state(self, state: torch.Tensor) -> torch.Tensor:
        """Process state tensor with device-aware padding."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        return state

    def _process_images(
        self, images: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Process images following OpenPI's camera mapping logic."""
        # Check that all input cameras are expected
        if set(images.keys()) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain cameras from {self.EXPECTED_CAMERAS}, got {tuple(images.keys())}"
            )

        # Assume that base image (cam_high) always exists
        base_image = images["cam_high"]

        # Start with base camera
        result_images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": torch.tensor(True, dtype=torch.bool),
        }

        # Add the extra images with OpenPI mapping
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }

        for dest, source in extra_image_names.items():
            if source in images:
                result_images[dest] = images[source]
                image_masks[dest] = torch.tensor(True, dtype=torch.bool)
            else:
                result_images[dest] = torch.zeros_like(base_image)
                image_masks[dest] = torch.tensor(False, dtype=torch.bool)

        return result_images, image_masks

    def _process_actions(self, actions) -> torch.Tensor:
        """Process actions tensor with device-aware padding and ALOHA encoding."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)

        # Apply ALOHA-specific action encoding (inverse of decoding)
        actions = self._encode_actions_inv(actions)

        # Ensure proper shape
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add time dimension

        return actions

    def _joint_flip_mask(self) -> torch.Tensor:
        """Used to convert between aloha and pi joint angles."""
        return torch.tensor(
            [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=torch.float32
        )

    def _normalize(self, x, min_val, max_val):
        """Normalize value to [0, 1] range."""
        return (x - min_val) / (max_val - min_val)

    def _unnormalize(self, x, min_val, max_val):
        """Unnormalize value from [0, 1] range."""
        return x * (max_val - min_val) + min_val

    def _gripper_to_angular(self, value):
        """Convert gripper position to angular space following OpenPI logic."""
        # Aloha transforms the gripper positions into a linear space. The following code
        # reverses this transformation to be consistent with pi0 which is pretrained in
        # angular space.

        # These values are coming from the Aloha code:
        # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
        value = self._unnormalize(value, min_val=0.01844, max_val=0.05800)

        # This is the inverse of the angular to linear transformation inside the Interbotix code.
        def linear_to_radian(linear_position, arm_length=0.036, horn_radius=0.022):
            value_calc = (horn_radius**2 + linear_position**2 - arm_length**2) / (
                2 * horn_radius * linear_position
            )
            return torch.arcsin(torch.clamp(value_calc, -1.0, 1.0))

        # The constants are taken from the Interbotix code.
        value = linear_to_radian(value)

        # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
        # There are 4096 total encoder counts and aloha uses a zero of 2048.
        # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
        return self._normalize(value, min_val=0.5476, max_val=1.6296)

    def _gripper_from_angular_inv(self, value):
        """Directly inverts the gripper_from_angular function."""
        # This is used for action encoding
        value = self._unnormalize(value, min_val=-0.6213, max_val=1.4910)
        return value - 0.5476

    def _encode_actions_inv(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions following OpenPI's _encode_actions_inv logic."""
        if self.adapt_to_pi:
            actions = self._joint_flip_mask() * actions
            # Process gripper positions at indices 6 and 13
            if len(actions.shape) == 2:  # [time, action_dim]
                actions[:, 6] = self._gripper_from_angular_inv(actions[:, 6])
                actions[:, 13] = self._gripper_from_angular_inv(actions[:, 13])
            else:  # [action_dim]
                actions[6] = self._gripper_from_angular_inv(actions[6])
                actions[13] = self._gripper_from_angular_inv(actions[13])

        return actions


class AlohaOutputs(_transforms.DataTransformFn):
    """ALOHA output transforms that match OpenPI's aloha_policy.AlohaOutputs."""

    def __init__(self, adapt_to_pi: bool = True):
        """
        Args:
            adapt_to_pi: If true, this will convert the joint and gripper values from the standard
                        Aloha space to the space used by the pi internal runtime which was used to
                        train the base model.
        """
        self.adapt_to_pi = adapt_to_pi

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform model outputs back to ALOHA format.

        Extract the first 14 actions (matching OpenPI behavior) and ensure proper tensor format.
        """
        actions = data["actions"]
        if isinstance(actions, torch.Tensor):
            # Keep as tensor but extract first 14 dimensions (matching OpenPI)
            actions = actions[..., :14]
        else:
            # Convert to tensor if needed
            actions = torch.tensor(actions, dtype=torch.float32)[..., :14]

        # Apply ALOHA-specific action encoding
        actions = self._encode_actions(actions)

        return {"actions": actions}

    def _joint_flip_mask(self) -> torch.Tensor:
        """Used to convert between aloha and pi joint angles."""
        return torch.tensor(
            [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=torch.float32
        )

    def _normalize(self, x, min_val, max_val):
        """Normalize value to [0, 1] range."""
        return (x - min_val) / (max_val - min_val)

    def _gripper_from_angular(self, value):
        """Convert from the gripper position used by pi0 to the gripper position that is used by Aloha."""
        # We do not scale the output since the trossen model predictions are already in radians.
        # See the comment in _gripper_to_angular for a derivation of the constant
        value = value + 0.5476

        # These values are coming from the Aloha code:
        # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
        return self._normalize(value, min_val=-0.6213, max_val=1.4910)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions following OpenPI's _encode_actions logic."""
        if self.adapt_to_pi:
            # Flip the joints
            actions = self._joint_flip_mask() * actions
            # Process gripper positions at indices 6 and 13
            if len(actions.shape) == 2:  # [time, action_dim]
                actions[:, 6] = self._gripper_from_angular(actions[:, 6])
                actions[:, 13] = self._gripper_from_angular(actions[:, 13])
            else:  # [action_dim]
                actions[6] = self._gripper_from_angular(actions[6])
                actions[13] = self._gripper_from_angular(actions[13])

        return actions
