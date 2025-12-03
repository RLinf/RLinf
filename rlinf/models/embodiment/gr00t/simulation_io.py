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

import numpy as np
import torch
import torch.nn.functional as F


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by the GR00T model.
    The data format is determined by the modality_config and meta/info.json following LeRobot format.
    Considering that we don't have a unified data inferface, we use direct logic here.
    """
    groot_obs = {}

    # [B, C, H, W] -> [B, T(1), C, H, W] -> [B, T, H, W, C]
    groot_obs["video.image"] = (
        env_obs["images"].unsqueeze(1).numpy().transpose(0, 1, 3, 4, 2)
    )
    groot_obs["video.wrist_image"] = (
        env_obs["wrist_images"].unsqueeze(1).numpy().transpose(0, 1, 3, 4, 2)
    )
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
    Convert the observation to the format expected by the GR00T model.
    The data format is determined by the modality_config and meta/info.json following LeRobot format.
    Considering that we don't have a unified data inferface, we use direct logic here.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["images"] = cut_and_resize_images(
        env_obs["images"], env_obs["images"].shape[-2], 256
    )
    # [B, C, H, W] -> [B, T(1), C, H, W] -> [B, T, H, W, C]
    images = env_obs["images"].unsqueeze(1).numpy()
    groot_obs["video.ego_view"] = np.transpose(images, (0, 1, 3, 4, 2))
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Libero format.

    Args:
        action_chunk: Dictionary of action components from GR00T policy
        chunk_size: Number of action steps to consider from the chunk

    Returns:
        7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
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


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    # Accord to gr1 definition, action.left_arm happens to be 7 dims, matching the demand of maniskill.

    return action_chunk["action.left_arm"][:, :chunk_size]


# TODO: we need a unified embodiement data.
OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
}

ACTION_CONVERSION = {
    "libero": convert_to_libero_action,
    "maniskill": convert_to_maniskill_action,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """
    Cut and resize the images to the crop size.
    """
    original_width = images.shape[-1]  # 640
    start = (original_width - crop_size) // 2  # (640-480)/2 = 80
    end = start + crop_size  # 80 + 480 = 560

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images[:, :, :, start:end]  # Shape: (2, 3, 480, 480)
    # Step 2: Resize the cropped 480x480 tensor to 256x256
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )
    return resized_tensor


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].

    Normalization formula: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
