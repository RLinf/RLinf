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
Franka robot policy transforms for VLA training.

This module provides I/O transforms for Franka robot data collected with the realRL framework.
Uses 6D rotation representation to avoid singularity issues with quaternions/euler angles.

Matches OpenPI's franka_policy.py implementation.

State format (10D):
    - Position: [x, y, z] (3D)
    - Rotation: rot6d representation (6D) - first two columns of rotation matrix
    - Gripper: [gripper_pose] (1D)

Action format (10D):
    - Delta position: [dx, dy, dz] (3D)
    - Delta rotation: rot6d of delta rotation matrix (6D)
    - Gripper: [gripper] (1D) - absolute gripper target [0, 1]
"""

from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .. import transforms as _transforms

# ============================================================================
# Rotation Conversion Utilities
# ============================================================================


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    r = Rotation.from_quat(quat)  # scipy uses [x, y, z, w] format
    return r.as_matrix()


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    r = Rotation.from_matrix(R)
    return r.as_quat()  # scipy returns [x, y, z, w] format


def euler_to_matrix(euler: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Convert euler angles [rx, ry, rz] to 3x3 rotation matrix."""
    r = Rotation.from_euler(seq, euler)
    return r.as_matrix()


def matrix_to_euler(R: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Convert 3x3 rotation matrix to euler angles [rx, ry, rz]."""
    r = Rotation.from_matrix(R)
    return r.as_euler(seq)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation.

    Takes the first two columns of the rotation matrix and flattens them.
    This provides a continuous representation without singularities.

    Args:
        R: 3x3 rotation matrix

    Returns:
        6D vector [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
    """
    return R[:, :2].T.flatten()  # [col1, col2] flattened


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrix.

    Args:
        rot6d: 6D vector [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]

    Returns:
        3x3 rotation matrix
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1)


def quaternion_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 6D rotation representation."""
    R = quaternion_to_matrix(quat)
    return matrix_to_rot6d(R)


def rot6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to quaternion [qx, qy, qz, qw]."""
    R = rot6d_to_matrix(rot6d)
    return matrix_to_quaternion(R)


def euler_delta_to_rot6d(euler_delta: np.ndarray) -> np.ndarray:
    """Convert euler angle delta [drx, dry, drz] to 6D rotation representation.

    The euler delta represents a small rotation from identity.
    We convert it to a rotation matrix and take the rot6d representation.

    Args:
        euler_delta: 3D euler angle delta in radians

    Returns:
        6D rotation representation of the delta rotation
    """
    R_delta = euler_to_matrix(euler_delta)
    return matrix_to_rot6d(R_delta)


def rot6d_to_euler_delta(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation back to euler angle delta.

    Args:
        rot6d: 6D rotation representation

    Returns:
        3D euler angle delta [drx, dry, drz]
    """
    R_delta = rot6d_to_matrix(rot6d)
    return matrix_to_euler(R_delta)


# ============================================================================
# State/Action Conversion Functions
# ============================================================================


def convert_state_to_rot6d(
    tcp_pose: np.ndarray, gripper_pose: np.ndarray
) -> np.ndarray:
    """Convert state from quaternion representation to rot6d representation.

    Args:
        tcp_pose: [x, y, z, qx, qy, qz, qw] (7D)
        gripper_pose: [gripper] (1D)

    Returns:
        state: [x, y, z, rot6d(6), gripper] (10D)
    """
    position = tcp_pose[:3]
    quaternion = tcp_pose[3:7]
    rot6d = quaternion_to_rot6d(quaternion)

    return np.concatenate([position, rot6d, gripper_pose])


def convert_action_to_rot6d(action: np.ndarray) -> np.ndarray:
    """Convert action from euler delta to rot6d representation.

    Args:
        action: [dx, dy, dz, drx, dry, drz, gripper] (7D)
            - Position delta: [dx, dy, dz]
            - Rotation delta: euler angles [drx, dry, drz] (already scaled)
            - Gripper: absolute target [0, 1]

    Returns:
        action: [dx, dy, dz, delta_rot6d(6), gripper] (10D)
    """
    delta_pos = action[:3]
    euler_delta = action[3:6]
    gripper = action[6:7]

    delta_rot6d = euler_delta_to_rot6d(euler_delta)

    return np.concatenate([delta_pos, delta_rot6d, gripper])


def convert_action_from_rot6d(action: np.ndarray) -> np.ndarray:
    """Convert action from rot6d representation back to euler delta.

    Args:
        action: [dx, dy, dz, delta_rot6d(6), gripper] (10D)

    Returns:
        action: [dx, dy, dz, drx, dry, drz, gripper] (7D)
    """
    delta_pos = action[:3]
    delta_rot6d = action[3:9]
    gripper = action[9:10]

    euler_delta = rot6d_to_euler_delta(delta_rot6d)

    return np.concatenate([delta_pos, euler_delta, gripper])


# ============================================================================
# Image Processing
# ============================================================================


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # Handle CHW -> HWC conversion
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


# ============================================================================
# OpenPi Transform Classes
# ============================================================================


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy.

    Used for testing and verification.
    """
    return {
        "observation/state/tcp_pose": np.random.rand(7).astype(np.float32),
        "observation/state/gripper_pose": np.random.rand(1).astype(np.float32),
        "observation/images/front_cam": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/images/wrist_cam": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "pick up the object",
    }


class FrankaInputs(_transforms.DataTransformFn):
    """Transform Franka robot inputs to model format with rot6d representation.

    Converts:
        - State: tcp_pose (7D quat) + gripper (1D) → 10D with rot6d
        - Action: 7D with euler delta → 10D with rot6d delta
        - Images: Maps front_cam and wrist_cam to model image inputs
    """

    def __init__(self, model_type: str = "pi05"):
        """
        Args:
            model_type: Model type ("pi0", "pi05", "pi0_fast") for image mask handling
        """
        self.model_type = model_type.lower()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Extract and convert state to rot6d representation
        tcp_pose = np.asarray(data["observation/state/tcp_pose"], dtype=np.float32)
        gripper_pose = np.asarray(
            data["observation/state/gripper_pose"], dtype=np.float32
        )
        if gripper_pose.ndim == 0:
            gripper_pose = gripper_pose[np.newaxis]

        # Convert state: position + rot6d + gripper = 10D
        state = convert_state_to_rot6d(tcp_pose, gripper_pose)

        # Parse images
        front_image = _parse_image(data["observation/images/front_cam"])
        wrist_image = _parse_image(data["observation/images/wrist_cam"])

        # Convert to tensor CHW format (matching LiberoInputs)
        front_image = self._process_image(front_image)
        wrist_image = self._process_image(wrist_image)

        # Set up image inputs based on model type
        # For PI0_FAST, all masks are True; for PI0/PI05, padding images are masked False
        if self.model_type == "pi0_fast":
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (front_image, torch.zeros_like(front_image), wrist_image)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            # PI0 and PI05
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (front_image, wrist_image, torch.zeros_like(front_image))
            image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": torch.from_numpy(state).float(),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Convert actions to rot6d representation (only during training)
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            # Handle action sequence (action_horizon, action_dim) or single action
            if actions.ndim == 1:
                converted_actions = convert_action_to_rot6d(actions)
            else:
                # Apply conversion to each action in the sequence
                converted_actions = np.stack(
                    [
                        convert_action_to_rot6d(actions[i])
                        for i in range(actions.shape[0])
                    ],
                    axis=0,
                )
            inputs["actions"] = torch.from_numpy(converted_actions).float()

        # Pass through the prompt
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

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
                key.startswith(("reward_", "return_", "action_", "history_"))
                and key not in inputs
            ):
                inputs[key] = data[key]
            elif key.endswith("_is_pad") and key not in inputs:
                inputs[key] = data[key]

        return inputs

    def _process_image(self, img) -> torch.Tensor:
        """Process image to CHW uint8 tensor format."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # Convert floating point to uint8
        if img.dtype.is_floating_point:
            img = (255 * img).to(torch.uint8)

        # Ensure CHW format
        if len(img.shape) == 3:
            if img.shape[-1] == 3:  # HWC -> CHW
                img = img.permute(2, 0, 1)

        # Ensure uint8 dtype
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        return img


class FrankaOutputs(_transforms.DataTransformFn):
    """Transform model outputs back to Franka robot format.

    Converts:
        - Action: 10D with rot6d delta → 7D with euler delta

    The output action format matches the original data format:
        [dx, dy, dz, drx, dry, drz, gripper]
    """

    def __init__(self, action_dim: int = 7):
        """
        Args:
            action_dim: Output action dimension (default 7 for Franka)
        """
        self.action_dim = action_dim

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            return data

        actions = data["actions"]

        # Convert to numpy for processing
        if isinstance(actions, torch.Tensor):
            device = actions.device
            dtype = actions.dtype
            actions = actions.cpu().numpy()
            is_tensor = True
        else:
            actions = np.asarray(actions)
            is_tensor = False

        # Handle action sequence or single action
        if actions.ndim == 1:
            # Single action: (10,) → (7,)
            output_actions = convert_action_from_rot6d(actions[:10])
        else:
            # Action sequence: (horizon, 10+) → (horizon, 7)
            output_actions = np.stack(
                [
                    convert_action_from_rot6d(actions[i, :10])
                    for i in range(actions.shape[0])
                ],
                axis=0,
            )

        # Convert back to tensor if needed
        if is_tensor:
            output_actions = torch.from_numpy(output_actions).to(
                device=device, dtype=dtype
            )

        return {"actions": output_actions}


# ============================================================================
# Utility Functions for Testing
# ============================================================================


def test_rotation_conversions():
    """Test that rotation conversions are invertible."""
    print("Testing rotation conversions...")

    # Test quaternion → rot6d → quaternion
    quat_original = np.array([0.1, 0.2, 0.3, 0.9])
    quat_original = quat_original / np.linalg.norm(quat_original)  # Normalize

    rot6d = quaternion_to_rot6d(quat_original)
    quat_recovered = rot6d_to_quaternion(rot6d)

    # Quaternions can differ by sign
    if np.dot(quat_original, quat_recovered) < 0:
        quat_recovered = -quat_recovered

    print(f"Quaternion original: {quat_original}")
    print(f"Quaternion recovered: {quat_recovered}")
    print(f"Quaternion error: {np.linalg.norm(quat_original - quat_recovered):.6f}")

    # Test euler delta → rot6d → euler delta
    euler_original = np.array([0.1, -0.05, 0.15])  # Small angles

    rot6d = euler_delta_to_rot6d(euler_original)
    euler_recovered = rot6d_to_euler_delta(rot6d)

    print(f"\nEuler original: {euler_original}")
    print(f"Euler recovered: {euler_recovered}")
    print(f"Euler error: {np.linalg.norm(euler_original - euler_recovered):.6f}")

    # Test identity case
    identity_euler = np.array([0.0, 0.0, 0.0])
    rot6d_identity = euler_delta_to_rot6d(identity_euler)
    expected_identity = np.array([1, 0, 0, 0, 1, 0])  # First two columns of I

    print(f"\nIdentity rot6d: {rot6d_identity}")
    print(f"Expected identity: {expected_identity}")
    print(f"Identity error: {np.linalg.norm(rot6d_identity - expected_identity):.6f}")

    # Test full state/action conversion
    print("\n--- Testing full state/action conversion ---")
    tcp_pose = np.array([0.5, 0.1, 0.3, 0.1, 0.2, 0.3, 0.9])
    tcp_pose[3:7] = tcp_pose[3:7] / np.linalg.norm(tcp_pose[3:7])
    gripper_pose = np.array([0.5])

    state_10d = convert_state_to_rot6d(tcp_pose, gripper_pose)
    print(f"State 10D shape: {state_10d.shape}")
    print(f"State 10D: {state_10d}")

    action_7d = np.array([0.01, -0.02, 0.015, 0.05, -0.03, 0.02, 0.8])
    action_10d = convert_action_to_rot6d(action_7d)
    action_7d_recovered = convert_action_from_rot6d(action_10d)

    print(f"\nAction 7D original: {action_7d}")
    print(f"Action 10D: {action_10d}")
    print(f"Action 7D recovered: {action_7d_recovered}")
    print(
        f"Action recovery error: {np.linalg.norm(action_7d - action_7d_recovered):.6f}"
    )

    print("\n✓ All rotation conversion tests passed!")


if __name__ == "__main__":
    test_rotation_conversions()
