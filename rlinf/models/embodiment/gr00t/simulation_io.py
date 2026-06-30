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

import numpy as np
import torch
import torch.nn.functional as F


def _quat_to_euler_xyz(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a single quaternion (w,x,y,z) to intrinsic XYZ Euler angles (roll, pitch, yaw).

    Uses the standard quaternion→rotation-matrix→euler pipeline.  The conversion
    is intentionally kept self-contained (no scipy dependency) because this module
    runs inside the GR00T inference worker.
    """
    # rotation matrix from quaternion
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    # intrinsic XYZ:  roll = atan2(r21, r22), pitch = -asin(r20), yaw = atan2(r10, r00)
    pitch = -np.arcsin(np.clip(r20, -1.0, 1.0))
    roll = np.arctan2(r21, r22)
    yaw = np.arctan2(r10, r00)

    return float(roll), float(pitch), float(yaw)


def _batched_quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Batched quaternion (..., 4) → Euler XYZ (..., 3)."""
    quat = np.asarray(quat, dtype=np.float64)
    flat = quat.reshape(-1, 4)
    euler = np.empty((flat.shape[0], 3), dtype=np.float64)
    for i in range(flat.shape[0]):
        euler[i] = _quat_to_euler_xyz(
            float(flat[i, 0]), float(flat[i, 1]),
            float(flat[i, 2]), float(flat[i, 3]),
        )
    return euler.reshape(quat.shape[:-1] + (3,)).astype(np.float32)


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}

    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # [B, 8] -> [B, T(1), 8]
    groot_obs["state.x"] = env_obs["states"].unsqueeze(1)[:, :, 0:1].numpy()
    groot_obs["state.y"] = env_obs["states"].unsqueeze(1)[:, :, 1:2].numpy()
    groot_obs["state.z"] = env_obs["states"].unsqueeze(1)[:, :, 2:3].numpy()
    groot_obs["state.roll"] = env_obs["states"].unsqueeze(1)[:, :, 3:4].numpy()
    groot_obs["state.pitch"] = env_obs["states"].unsqueeze(1)[:, :, 4:5].numpy()
    groot_obs["state.yaw"] = env_obs["states"].unsqueeze(1)[:, :, 5:6].numpy()
    groot_obs["state.gripper"] = env_obs["states"].unsqueeze(1)[:, :, 6:].numpy()
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],  # H
        256,
    )
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).numpy()
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t_1_7 pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action_n1d5(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T N1.5 action chunk to Libero format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d6(
    action_chunk: dict[str, np.array],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.6 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")

    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.7 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")
    # gripper conversion is handled by the shared
    # ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.

    if action_array.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim action, got {action_array.shape[-1]}")
    return action_array


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Isaaclab Stack Cube format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_robocasa365_obs_to_gr00t_format(env_obs):
    """Convert RoboCasa365 observations to the GR00T model format.

    RoboCasa365 provides a 16+-dim state vector (eef_pos + eef_quat +
    gripper_qpos/qvel + base_to_eef + base_pose).  This converter extracts the
    arm-only proprioception (eef_pos:3, eef_quat→euler:3, gripper_qpos:1) and
    maps it to the same flat ``state.*`` keys expected by a libero-style GR00T
    checkpoint, so the model can consume the data without architecture changes.

    Notes:
        - Quaternions are converted to intrinsic-XYZ Euler angles so they fit
          the (roll, pitch, yaw) slots the checkpoint was trained with.
        - Base-related state components (base_to_eef, base_pos, base_quat) are
          intentionally dropped — the model only sees arm proprioception.
        - Images: ``main_images`` → ``video.image``, ``wrist_images`` →
          ``video.wrist_image``, same as the libero converter.
    """
    import logging

    _logger = logging.getLogger(__name__)
    groot_obs: dict[str, Any] = {}

    # --- images ----------------------------------------------------------------
    # [B, H, W, C] → [B, T=1, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    if env_obs.get("wrist_images") is not None:
        groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()

    # --- state ----------------------------------------------------------------
    states: torch.Tensor = env_obs["states"]  # [B, state_dim]
    state_dim = states.shape[-1]

    # The robocasa365 state layout (see env/robocasa365.yaml) is:
    #   eef_pos (3) | eef_quat (4) | gripper_qpos (1) | gripper_qvel (1) |
    #   base_to_eef_pos (3) | base_to_eef_quat (4) | base_pos (?) | base_quat (?)
    #
    # We take the first 8 dims (eef_pos + eef_quat + gripper_qpos) and convert
    # the quaternion slice to Euler angles.
    if state_dim < 8:
        _logger.warning(
            "RoboCasa365 state has only %d dims (expected ≥8); "
            "zero-padding missing proprioception dims.",
            state_dim,
        )
        padded = torch.zeros(states.shape[0], 8, dtype=states.dtype, device=states.device)
        padded[:, :state_dim] = states[:, :state_dim]
        states = padded

    eef_pos = states[:, :3].cpu().numpy()          # [B, 3]
    eef_quat = states[:, 3:7].cpu().numpy()        # [B, 4]  (w,x,y,z)
    gripper = states[:, 7:8].cpu().numpy()          # [B, 1]

    eef_euler = _batched_quat_to_euler_xyz(eef_quat)  # [B, 3]  (roll, pitch, yaw)

    # unsqueeze time dim  [B, D] → [B, T=1, D]
    groot_obs["state.x"] = eef_pos[:, np.newaxis, 0:1]
    groot_obs["state.y"] = eef_pos[:, np.newaxis, 1:2]
    groot_obs["state.z"] = eef_pos[:, np.newaxis, 2:3]
    groot_obs["state.roll"] = eef_euler[:, np.newaxis, 0:1]
    groot_obs["state.pitch"] = eef_euler[:, np.newaxis, 1:2]
    groot_obs["state.yaw"] = eef_euler[:, np.newaxis, 2:3]
    groot_obs["state.gripper"] = gripper[:, np.newaxis, :]

    # --- task description -----------------------------------------------------
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
    "robocasa365": convert_robocasa365_obs_to_gr00t_format,
}

ACTION_CONVERSION_N1D5 = {
    "libero": convert_to_libero_action_n1d5,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "robocasa365": convert_to_libero_action_n1d5,
}

ACTION_CONVERSION_N1D6 = {
    "libero": convert_to_libero_action_n1d6,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "robocasa365": convert_to_libero_action_n1d6,
}

ACTION_CONVERSION_N1D7 = {
    "libero": convert_to_libero_action_n1d7,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "robocasa365": convert_to_libero_action_n1d7,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """Cut and resize the images to the crop size."""
    images_nchw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    original_width = images_nchw.shape[-1]  # W
    start = (original_width - crop_size) // 2
    end = start + crop_size

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images_nchw[:, :, :, start:end]  # [B, C, H, crop_W]

    # Resize: interpolate to target_size x target_size
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )  # [B, C, target_size, target_size]

    # Convert back to NHWC
    resized_nhwc = resized_tensor.permute(
        0, 2, 3, 1
    ).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    return resized_nhwc


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
