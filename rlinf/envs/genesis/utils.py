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

from __future__ import annotations

import numpy as np
import torch


def extract_robot_state(
    robot,
    eef_link,
    num_motor_dofs: int,
    num_finger_dofs: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract robot proprioceptive state as a flat tensor.

    Returns a tensor of shape ``(B, state_dim)`` where ``state_dim`` is
    ``3 (eef_pos) + 4 (eef_quat) + num_finger_dofs (gripper)``.

    Args:
        robot: Genesis robot entity.
        eef_link: Genesis link object for the end-effector.
        num_motor_dofs: Number of motor (arm) DOFs.
        num_finger_dofs: Number of finger (gripper) DOFs.
        device: Target torch device for the output tensor.
    """
    eef_pos = eef_link.get_pos()  # (B, 3)
    eef_quat = eef_link.get_quat()  # (B, 4)
    gripper = robot.get_dofs_position()[..., num_motor_dofs : num_motor_dofs + num_finger_dofs]  # (B, F)
    state = torch.cat([eef_pos, eef_quat, gripper], dim=-1).float()
    if state.device != torch.device(device):
        state = state.to(device)
    return state


def _to_uint8_rgb(data) -> np.ndarray:
    """Convert a raw render output to a uint8 RGB numpy array.

    Handles torch tensors, RGBA -> RGB stripping, and float -> uint8
    conversion.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    # Strip alpha channel if present.
    if data.ndim >= 3 and data.shape[-1] == 4:
        data = data[..., :3]
    # Ensure uint8.
    if data.dtype != np.uint8:
        if data.max() <= 1.0:
            data = np.clip(data * 255, 0, 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    return data


def camera_render_rgb(
    camera,
    num_envs: int,
    scene=None,
    camera_base_pos: tuple[float, ...] | np.ndarray = (3.5, 0.0, 2.5),
    camera_base_lookat: tuple[float, ...] | np.ndarray = (0.0, 0.0, 0.5),
) -> np.ndarray:
    """Render RGB images from a Genesis camera sensor.

    When the scene uses ``BatchRenderer`` (the recommended backend for RL
    training), ``camera.render(rgb=True)`` natively returns a tensor of
    shape ``(n_envs, H, W, 3)``.  This function converts that output to
    a numpy uint8 array.

    A fallback per-env rendering loop is provided for the Rasterizer
    backend, but ``BatchRenderer`` should be preferred for performance.

    Args:
        camera: Genesis camera sensor object.
        num_envs: Number of parallel environments (batch dimension).
        scene: The built ``genesis.Scene``.  Required for the
            Rasterizer fallback when ``num_envs > 1``.
        camera_base_pos: Camera position relative to the env origin
            (used only by the Rasterizer fallback).
        camera_base_lookat: Camera look-at target relative to the env
            origin (used only by the Rasterizer fallback).

    Returns:
        RGB images as ``np.ndarray`` with shape ``(num_envs, H, W, 3)``
        and dtype ``uint8``.
    """
    # BatchRenderer path: render() returns (n_envs, H, W, C) directly.
    data = camera.render(rgb=True)
    if isinstance(data, tuple):
        data = data[0]
    raw = _to_uint8_rgb(data)

    if raw.ndim == 4 and raw.shape[0] == num_envs:
        return raw

    # Single-env case.
    if num_envs <= 1:
        if raw.ndim == 3:
            raw = raw[np.newaxis, ...]
        return raw

    # Rasterizer fallback: render per-env by repositioning the camera.
    if scene is None:
        raise ValueError(
            "camera_render_rgb requires `scene` when num_envs > 1 "
            "and the camera backend does not produce batched output. "
            "Consider using gs.renderers.BatchRenderer for batched "
            "rendering."
        )

    base_pos = np.asarray(camera_base_pos, dtype=np.float64)
    base_lookat = np.asarray(camera_base_lookat, dtype=np.float64)

    batch_imgs: list[np.ndarray] = []
    for i in range(num_envs):
        offset = np.asarray(scene.envs_offset[i], dtype=np.float64)
        camera.set_pose(pos=offset + base_pos, lookat=offset + base_lookat)
        frame = camera.render()
        if isinstance(frame, tuple):
            frame = frame[0]
        batch_imgs.append(_to_uint8_rgb(frame))

    return np.stack(batch_imgs, axis=0)
