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
Function Description:
This script processes a single large .pkl file containing continuous robot demonstration data.
It performs the following main tasks:
1. Segments the continuous data stream into individual episodes.
2. Cleans the data by filtering out frames with negligible actions (i.e., when the robot is stationary).
3. Converts data formats (e.g., float images to uint8) for efficiency and compatibility.
4. Restructures the data and saves each episode into a separate .pkl file, formatted for use
   in downstream training pipelines.

Usage:
Run the script from the command line, providing the path to the input .pkl file as the
sole argument.

Example:
    python process_realworld_data.py /path/to/your/input_data.pkl

The script will automatically create a directory named 'collected_data' in the same
location as the input file and save the processed episodes there.
"""

import argparse
import os
import pickle
from typing import Any, Union

import numpy as np
import torch
from tqdm import tqdm

# --- Configuration Area ---
# These settings control the data processing logic.

# Convert image data from float (0-1) to uint8 (0-255) for smaller file sizes.
TO_UINT8 = True
# Filter out frames where the robot action is negligible (i.e., the robot is stationary).
REMOVE_ZERO_ACTIONS = True
# Threshold to consider an action as "zero" or stationary.
ZERO_ACTION_THRESHOLD = 1e-5


def ensure_dir(path: str):
    """
    Ensures a directory exists. If it does, a warning is printed.
    If not, it is created.
    """
    if os.path.exists(path):
        print(f"Warning: Output directory '{path}' already exists.")
        print(
            "The script will overwrite files with the same name but will not clear the directory."
        )
        print(
            "For a fresh start, please manually delete the directory: rm -rf collected_data"
        )
    os.makedirs(path, exist_ok=True)


def process_image(img_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Converts an image tensor to a NumPy array on the CPU.
    Optionally converts the data type from float to uint8.
    """
    if isinstance(img_tensor, torch.Tensor):
        img_data = img_tensor.detach().cpu().numpy()
    else:
        img_data = img_tensor  # Already a NumPy array

    if TO_UINT8 and img_data.dtype != np.uint8:
        if np.issubdtype(img_data.dtype, np.floating):
            # Assumes float data is in the [0.0, 1.0] range
            img_data = (img_data * 255.0).round().astype(np.uint8)

    return img_data


def save_episode(
    episode_buffer: dict[str, list[Any]], episode_id: int, output_dir: str
) -> tuple[bool, int]:
    """
    Saves a single processed episode to a .pkl file.

    Returns:
        A tuple (bool, int) indicating if the episode was saved and the number of success frames.
    """
    if not episode_buffer["actions"]:
        # Do not save empty episodes that were filtered out completely.
        return False, 0

    # Check if the episode contains any success frames
    success_flags = [info["is_obj_placed"] for info in episode_buffer["infos"]]
    is_success_episode = any(success_flags)
    success_count = sum(success_flags)

    label = "success" if is_success_episode else "fail"
    # Mimic the filename format of the target project
    filename = f"rank_0_env_0_episode_{episode_id}_{label}.pkl"
    save_path = os.path.join(output_dir, filename)

    # Construct the final data structure to be saved
    episode_data = {
        "mode": "train",
        "rank": 0,
        "env_idx": 0,
        "episode_id": episode_id,
        "success": is_success_episode,
        "observations": episode_buffer["observations"],
        "actions": episode_buffer["actions"],
        "rewards": episode_buffer["rewards"],
        "terminated": episode_buffer["terminated"],
        "truncated": episode_buffer["truncated"],
        "infos": episode_buffer["infos"],
    }

    with open(save_path, "wb") as f:
        pickle.dump(episode_data, f)

    return True, success_count


def main(args):
    """
    Main function to load, process, and save robot demonstration data.
    """
    input_pkl_path = args.input_pkl_path

    # --- Dynamic Path Setup ---
    # The output directory will be created in the same location as the input file.
    base_dir = os.path.dirname(os.path.abspath(input_pkl_path))
    output_dir = os.path.join(base_dir, "collected_data")

    try:
        print(f"Loading raw data from: {input_pkl_path}")
        with open(input_pkl_path, "rb") as f:
            raw_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_pkl_path}'")
        return

    print(f"Loaded {len(raw_data)} transitions. Starting processing...")

    ensure_dir(output_dir)

    # Initialize a buffer to hold data for the current episode
    current_episode: dict[str, list[Any]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "infos": [],
    }

    episode_count = 0
    total_success_frames = 0
    total_frames_kept = 0

    for i, transition in enumerate(tqdm(raw_data, desc="Processing Transitions")):
        # --- 1. Get original termination flags ---
        raw_term = transition["terminations"].item()
        raw_trunc = transition["truncations"].item()
        # An episode ends if terminated, truncated, or it's the last transition in the file.
        is_raw_end_of_episode = raw_term or raw_trunc or (i == len(raw_data) - 1)

        # --- 2. Decide whether to keep the current frame ---
        should_keep_frame = True

        action = transition["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu()

        if REMOVE_ZERO_ACTIONS:
            # If the action magnitude is negligible, we consider it a stationary frame.
            # We skip it unless it's the final, successful frame of an episode.
            if torch.sum(torch.abs(action)) < ZERO_ACTION_THRESHOLD and not raw_term:
                should_keep_frame = False

        # --- 3. If keeping the frame, process and add to buffer ---
        if should_keep_frame:
            # Process Observation
            raw_obs = transition["transitions"]["obs"]
            img = process_image(raw_obs["main_images"])
            state = raw_obs["states"]
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu()

            # Process Reward and generate a boolean success label
            raw_reward = transition["rewards"]
            reward_val = (
                raw_reward.item()
                if isinstance(raw_reward, torch.Tensor)
                else raw_reward
            )

            # Map reward to a boolean flag: reward > 0.5 is considered a success
            is_obj_placed = reward_val > 0.5

            info = {"is_obj_placed": is_obj_placed}

            # Append processed data to the current episode buffer
            current_episode["observations"].append(
                {"main_images": img, "states": state}
            )
            current_episode["actions"].append(action)
            current_episode["rewards"].append(torch.tensor(reward_val))
            # Append False for now; will be corrected if this is the last frame of the episode.
            current_episode["terminated"].append(False)
            current_episode["truncated"].append(False)
            current_episode["infos"].append(info)

            total_frames_kept += 1

        # --- 4. At the end of a raw episode, save the buffered data ---
        if is_raw_end_of_episode:
            # Only process and save if the buffer is not empty (i.e., we kept at least one frame)
            if len(current_episode["actions"]) > 0:
                # CRITICAL STEP: Correct the termination flags for the *last kept frame*.
                # Since we might have dropped stationary frames at the end, the true end
                # of our cleaned episode is the last frame currently in the buffer.
                if raw_term:
                    current_episode["terminated"][-1] = True
                if raw_trunc:
                    current_episode["truncated"][-1] = True

                # Save the completed episode
                saved, s_count = save_episode(
                    current_episode, episode_count, output_dir
                )
                if saved:
                    total_success_frames += s_count
                    episode_count += 1
            else:
                # This can happen if an entire episode consisted of "zero" actions.
                print(
                    f"Warning: Episode {episode_count} was empty after filtering and has been skipped."
                )

            # Reset the buffer for the next episode
            current_episode = {k: [] for k in current_episode}

    print("-" * 50)
    print("Processing Complete!")
    print(f"Data has been saved to: {os.path.abspath(output_dir)}")
    print(f"Total episodes saved: {episode_count}")
    print(f"Total valid frames kept: {total_frames_kept}")
    print(f"Total success frames (is_obj_placed=True): {total_success_frames}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a single large .pkl file of robot demonstrations into multiple per-episode .pkl files."
    )
    parser.add_argument(
        "input_pkl_path",
        type=str,
        help="Path to the input .pkl file containing the raw demonstration data.",
    )

    args = parser.parse_args()
    main(args)
