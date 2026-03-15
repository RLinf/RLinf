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

"""Utility functions for Robocasa environments."""

import logging
import os
from typing import Optional, Union

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# SPACES #############

# STATE #####

ROBOCASA_STATES = {  # for daixianjie/robocasa_lerobot dataset
    "robot0_eef_pos": np.arange(0, 3),
    "robot0_eef_quat": np.arange(3, 7),
    "robot0_gripper_qpos": np.arange(7, 9),
    "robot0_gripper_qvel": np.arange(9, 11),
    "robot0_base_to_eef_pos": np.arange(11, 14),
    "robot0_base_to_eef_quat": np.arange(14, 18),
    "robot0_base_pos": np.arange(18, 21),
    "robot0_base_quat": np.arange(21, 25),
}

STATE_SPACE_STR_MAPPING = {
    # NOTE: see https://github.com/robocasa/robocasa/issues/11, in Robocasa paper the authors use 16d state as input.
    "16d": [
        "robot0_base_to_eef_pos",  # 11-14， 3
        "robot0_base_to_eef_quat",  # 14-18， 4
        "robot0_base_pos",  # 18-21，3
        "robot0_base_quat",  # 21-25， 4
        "robot0_gripper_qpos",  # 7-9， 2
    ],  # add up to 16
    "25d": list(ROBOCASA_STATES.keys()),  # add up to 25
}


def get_state_space(state_space: Union[str, list]) -> list:
    if isinstance(state_space, str):
        state_space_str = state_space
        state_space = STATE_SPACE_STR_MAPPING.get(state_space)
        if state_space is None:
            logging.warning(
                f"String-format state_space {state_space_str} property of RoboCasaInputs is not registered in STATE_SPACE_STR_MAPPING {list(STATE_SPACE_STR_MAPPING.keys())}"
            )

    return state_space


def _check_state_space(state_space: list) -> bool:
    check_ret = True
    for state_name in state_space:
        if state_name not in ROBOCASA_STATES.keys():
            check_ret = False
            break
    return check_ret


def get_state_ids(state_space: list) -> list:
    all_state_ids = []
    for robocasa_state_name in state_space:
        state_ids = ROBOCASA_STATES[robocasa_state_name]
        all_state_ids.extend(state_ids)

    return all_state_ids


# IMAGE #####

# Env-level observation keys used for images.
OBS_KEY_IMAGES = [
    "observation/image",
    "observation/wrist_image",
    "observation/extra_view_image",
]

# Mapping from env-level observation keys to robocasa camera names.
OBS_KEY_ROBOCASA_IMAGE_MAPPING = {
    "main_images": "robot0_agentview_left_image",
    "wrist_images": "robot0_eye_in_hand_image",
    "extra_view_images": "robot0_agentview_right_image",
}

OBS_KEY_CAMERA_NAME_MAPPING = {
    "observation/image": "robot0_agentview_left",
    "observation/wrist_image": "robot0_eye_in_hand",
    "observation/extra_view_image": "robot0_agentview_right",
}

DEFAULT_ROBOCASA_IMAGE_SIZE = (224, 224, 3)

# Env-level image space mapping: preset name -> list of observation keys.
IMAGE_SPACE_STR_MAPPING = {
    "2views": [
        "observation/image",
        "observation/wrist_image",
    ],
    "3views": [
        "observation/image",
        "observation/wrist_image",
        "observation/extra_view_image",
    ],
}


def get_image_space(image_space: Union[str, list]) -> list:
    """Resolve image_space into a list of observation keys."""
    if isinstance(image_space, str):
        image_space_str = image_space
        image_space = IMAGE_SPACE_STR_MAPPING.get(image_space)
        if image_space is None:
            logging.warning(
                f"String-format image_space {image_space_str} property of RoboCasaInputs is not registered in IMAGE_SPACE_STR_MAPPING {list(IMAGE_SPACE_STR_MAPPING.keys())}"
            )
    return image_space


def _check_image_space(image_space: list) -> bool:
    """Validate that all obs_keys in image_space are known image observation keys."""
    if not isinstance(image_space, list):
        return False
    for obs_key in image_space:
        if obs_key not in OBS_KEY_IMAGES:
            return False
    return True


# ACTIONS #####


ROBOCASA_ALL_ACTION_DIM = 12

ROBOCASA_ACTIONS = {  # for daixianjie/robocasa_lerobot dataset
    "rel_pose_6d": np.arange(0, 6),  # corresponding to "right" _action_split_indices
    "gripper": np.arange(6, 7),
    "base": np.arange(7, 10),
    "torso": np.arange(10, 11),
    "base_mode": np.arange(
        11, 12
    ),  # NOTE: https://github.com/robocasa/robocasa/issues/141
}

ROBOCASA_DEFAULT_ACTION = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
)

ACTION_SPACE_STR_MAPPING = {
    # NOTE: see https://github.com/robocasa/robocasa/issues/11, align with paper
    "7d": [
        "rel_pose_6d",  # 0-6， 6
        "gripper",  # 6-7, 1
    ],  # add up to 7
    "12d": list(ROBOCASA_ACTIONS.keys()),  # add up to 12
}


def get_action_space(action_space: Union[str, list]) -> list:
    if isinstance(action_space, str):
        action_space_str = action_space
        action_space = ACTION_SPACE_STR_MAPPING.get(action_space)
        if action_space is None:
            logging.warning(
                f"String-format action_space {action_space_str} property of RoboCasaInputs is not registered in ACTION_SPACE_STR_MAPPING {list(ACTION_SPACE_STR_MAPPING.keys())}"
            )
    return action_space


def _check_action_space(action_space: list) -> bool:
    check_ret = True
    for action_name in action_space:
        if action_name not in ROBOCASA_ACTIONS.keys():
            check_ret = False
            break
    return check_ret


def get_action_ids(action_space: list) -> list:
    all_action_ids = []
    for robocasa_action_name in action_space:
        action_ids = ROBOCASA_ACTIONS[robocasa_action_name]
        all_action_ids.extend(action_ids)

    return all_action_ids


# VIDEO ###########
def tile_images(
    images: list[Union[np.ndarray, torch.Tensor]], nrows: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Copied from maniskill https://github.com/haosulab/ManiSkill
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns: list[list[Union[np.ndarray, torch.Tensor]]] = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image


def put_text_on_image(
    image: np.ndarray, lines: list[str], max_width: int = 200
) -> np.ndarray:
    """
    Put text lines on an image with automatic line wrapping.

    Args:
        image: Input image as numpy array
        lines: List of text lines to add
        max_width: Maximum width for text wrapping
    """
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    new_lines = []
    for line in lines:
        words = line.split()
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            test_width = font.getlength(test_line)

            if test_width <= max_width:
                current_line.append(word)
            else:
                new_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            new_lines.append(" ".join(current_line))

    y = -10
    for line in new_lines:
        bbox = draw.textbbox((0, 0), text=line)
        textheight = bbox[3] - bbox[1]
        y += textheight + 10
        x = 10
        draw.text((x, y), text=line, fill=(0, 0, 0))
    return np.array(image)


def put_info_on_image(
    image: np.ndarray,
    info: dict[str, float],
    extras: Optional[list[str]] = None,
    overlay: bool = True,
) -> np.ndarray:
    """
    Put information dictionary and extra lines on an image.

    Args:
        image: Input image
        info: Dictionary of key-value pairs to display
        extras: Additional text lines to display
        overlay: Whether to overlay text on image
    """
    lines = [
        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in info.items()
    ]
    if extras is not None:
        lines.extend(extras)
    return put_text_on_image(image, lines)


def save_rollout_video(
    rollout_images: list[np.ndarray], output_dir: str, video_name: str, fps: int = 30
) -> None:
    """
    Saves an MP4 replay of an episode.

    Args:
        rollout_images: List of images from the episode
        output_dir: Directory to save the video
        video_name: Name of the output video file
        fps: Frames per second for the video
    """
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = os.path.join(output_dir, f"{video_name}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
