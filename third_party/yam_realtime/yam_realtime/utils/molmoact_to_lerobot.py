# To train lerobot model, use
    # python lerobot/scripts/train.py --dataset.repo_id=/home/sean/Desktop/YAM/lerobot/lerobot/common/datasets/yam_dp_dataset --policy.type=diffusion
# or to resume from a checkpoint for training
    # python lerobot/scripts/train.py --dataset.repo_id=/home/sean/Desktop/YAM/lerobot/lerobot/common/datasets/yam_dp_dataset --policy.type=diffusion 
    # --output_dir=/home/sean/Desktop/YAM/lerobot/outputs/train/2025-11-12/01-53-36_diffusion/checkpoints/010000/pretrained_model/config.json --resume=true
# note that repo_id is requred and can be either the absolute path of the local dir or huggingface repo
# We can collect data using the script collect_data_with_policy.py and then convert to lerobot format using this script molmoact_to_lerobot.py
# Then, this converted dataset can be used to train lerobot model.


#!/usr/bin/env python
"""
Script to convert molmoact_test dataset to LeRobot format
For training diffusion models
Usage:
python convert_molmoact_test_to_lerobot.py --data_dir /path/to/molmoact_test --output_dir /path/to/molmoact_test_lerobot
"""
import sys
sys.path.append('/home/sean/Desktop/YAM')
sys.path.append('/home/sean/Desktop/YAM/lerobot')
import argparse
import json
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def load_molmoact_test_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load molmoact_test dataset from episode-first layout:
    
    data_dir/
    ├── 000001/
    │   ├── 000001.json
    │   ├── left_rgb/
    │   ├── right_rgb/
    │   └── front_rgb/
    ├── 000002/
    │   ├── 000002.json
    │   ├── left_rgb/
    │   ├── right_rgb/
    │   └── front_rgb/
    └── ...
    """

    episodes = []
    data_path = Path(data_dir)

    # Find all episode directories (e.g., 000001, 000002, etc.)
    episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Found {len(episode_dirs)} episodes under {data_dir}")

    for ep_dir in episode_dirs:
        episode_id = ep_dir.name
        json_path = ep_dir / f"{episode_id}.json"

        if not json_path.exists():
            print(f"Skipping {episode_id}: missing JSON file {json_path}")
            continue

        # Load the JSON data (which acts like your pickle info)
        with open(json_path, 'r') as f:
            episode_data = json.load(f)

        # Task description — if present in JSON
        task_description = episode_data[0].get("task", f"task_{episode_id}") if episode_data else f"task_{episode_id}"

        # Simulate pickle structure — assuming your JSON has arrays for qpos and action
        try:
            left_joint = np.array([json.loads(frame["left_joint"]) for frame in episode_data], dtype=np.float32)
            right_joint = np.array([json.loads(frame["right_joint"]) for frame in episode_data], dtype=np.float32)
            # left_actions = np.array([json.loads(frame["left_delta_action"]) for frame in episode_data], dtype=np.float32)
            # right_actions = np.array([json.loads(frame["right_delta_action"]) for frame in episode_data], dtype=np.float32)
            left_actions = np.array([json.loads(frame["left_joint"]) for frame in episode_data], dtype=np.float32)
            right_actions = np.array([json.loads(frame["right_joint"]) for frame in episode_data], dtype=np.float32)

            qpos = np.concatenate([left_joint, right_joint], axis=1)
            actions = np.concatenate([left_actions, right_actions], axis=1)

            language_instruction = episode_data[0].get("language_instruction", ["no_instruction"])
        except Exception as e:
            print(f"Error parsing episode {episode_id}: {e}")
            continue

        # Prepare episode info
        episode_info = {
            "task_name": "default_task",
            "episode_id": episode_id,
            "task_description": task_description,
            "qpos": qpos,
            "actions": actions,
            "language_instruction": language_instruction,
            "episode_length": len(actions),
            "images": []
        }

        # Find camera image folders (left_rgb, right_rgb, front_rgb)
        for camera_dir in [ep_dir / "left_rgb", ep_dir / "right_rgb", ep_dir / "front_rgb"]:
            if not camera_dir.exists():
                continue

            camera_name = camera_dir.name.replace("_rgb", "")
            image_files = sorted([f for f in camera_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            camera_images = []
            for img_file in image_files:
                try:
                    img = Image.open(img_file)
                    camera_images.append(img)
                except Exception as e:
                    print(f"  Warning: Failed to load image {img_file}: {e}")
            episode_info["images"].append({
                "camera_name": camera_name,
                "images": camera_images
            })

        episodes.append(episode_info)

    print(f"Loaded {len(episodes)} total episodes")
    return episodes

def get_completed_episodes(output_dir: str) -> set:
    """Get indices of completed episodes"""
    completed = set()
    data_dir = Path(output_dir) / "data"
    if data_dir.exists():
        for chunk_dir in data_dir.iterdir():
            if chunk_dir.is_dir():
                for parquet_file in chunk_dir.glob("episode_*.parquet"):
                    # Extract episode index
                    episode_idx = int(parquet_file.stem.split('_')[1])
                    completed.add(episode_idx)
    return completed
def create_lerobot_dataset(episodes: List[Dict[str, Any]], output_dir: str, fps: int = 30):
    """
    Convert molmoact_test data to LeRobot format
    Args:
        episodes: Loaded episode data
        output_dir: Output directory
        fps: Data collection frame rate
    """
    # Define feature structure - based on molmoact_test data
    features = {
        # Robot joint positions (required)
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),  # lowdim_qpos dimension
            "names": ["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6", "left_gripper", "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6", "right_gripper"] # left joint, right joint
        },
        # Actions (required) (delta)
        # "action": { 
        #     "dtype": "float32",
        #     "shape": (16,),  # action dimension
        #     "names": ["left_dx", "left_dy", "left_dz", "left_w", "left_wx", "left_wy", "left_wz", "left_gripper", "right_dx", "right_dy", "right_dz", "right_w", "right_wx", "right_wy", "right_wz", "right_gripper"]
        # },
        # Actions (required) (joint)
        "action": { 
            "dtype": "float32",
            "shape": (14,),  # action dimension
            "names": ["left_m1", "left_m2", "left_m3", "left_m4", "left_m5", "left_m6", "left_m7", "right_m8", "right_m9", "right_m3", "right_m4", "right_m5", "right_m6", "right_m7"]
        },
        # # End-effector position (optional)
        # "observation.ee_pos": {
        #     "dtype": "float32",
        #     "shape": (7,),  # lowdim_ee dimension
        #     "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]
        # },
        # Image observations (optional)
        "observation.images.camera_left": { 
            "dtype": "image",
            "shape": (480, 640, 3),  # Adjust according to actual image size
            "names": ["height", "width", "channels"]
        },
        "observation.images.camera_right": {
            "dtype": "image",
            "shape": (480, 640, 3),  # Adjust according to actual image size
            "names": ["height", "width", "channels"]
        },
        "observation.images.camera_front": {
            "dtype": "image",
            "shape": (480, 640, 3),  # Adjust according to actual image size
            "names": ["height", "width", "channels"]
        }
    }
    # Check completed episodes
    completed_episodes = get_completed_episodes(output_dir)
    print(f"Found {len(completed_episodes)} completed episodes")
    if completed_episodes:
        print(f"Completed episode range: {min(completed_episodes)} - {max(completed_episodes)}")
        response = input("Continue processing incomplete episodes? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Conversion cancelled")
            return
    # Create or load LeRobot dataset
    if completed_episodes:
        print(f"Loading existing dataset: {output_dir}")
        dataset = LeRobotDataset(output_dir)
    else:
        print(f"Creating new dataset: {output_dir}")
        dataset = LeRobotDataset.create(
            repo_id="molmoact_test_dataset",
            fps=fps,
            root=output_dir,
            features=features,
            use_videos=True  # Use video encoding to save space.
        )
    print(f"Features: {list(features.keys())}")
    import tqdm
    from tqdm import trange
    # Add data
    for episode_idx, episode_data in enumerate(tqdm.tqdm(episodes)):
        # Skip completed episodes
        if episode_idx in completed_episodes:
            print(f"Skipping completed episode {episode_idx}")
            continue
        # print(f"Processing episode {episode_idx}: {episode_data['task_name']}/{episode_data['episode_id']}")
        qpos = episode_data['qpos']
        # ee_pos = episode_data['ee_pos']
        actions = episode_data['actions']
        task_description = episode_data['task_description']
        episode_length = episode_data['episode_length']
        # Get image data
        camera_images = {}
        for cam_data in episode_data['images']:
            camera_name = cam_data['camera_name']
            images = cam_data['images']
            camera_images[camera_name] = images
        for frame_idx in trange(episode_length):
            if frame_idx < 5:
                continue #do not process the first 5 frames
            frame_data = {
                # Robot state
                "observation.state": qpos[frame_idx].astype(np.float32),
                # Action
                "action": actions[frame_idx].astype(np.float32),
                # End-effector position
                # "observation.ee_pos": ee_pos[frame_idx],
                # Task
                "task": task_description
                # Timestamp - let LeRobot generate automatically
                # "timestamp": np.array([frame_idx / fps], dtype=np.float32)
            }
            # Add images (if available)
            for cam_idx, (camera_name, images) in enumerate(camera_images.items()):
                if frame_idx < len(images):
                    frame_data[f"observation.images.camera_{camera_name}"] = images[frame_idx]
            dataset.add_frame(frame_data)
        # Save episode
        dataset.save_episode()

        # print(f"Saved episode {episode_idx}, length: {episode_length}")
    print(f"Dataset creation completed!")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")
    print(f"Output directory: {output_dir}")
def main():
    parser = argparse.ArgumentParser(description="Convert molmoact_test data to LeRobot format")
    parser.add_argument("--data_dir", type=str,
                       default="/path/to/molmoact_test",
                       help="Input data directory")
    parser.add_argument("--output_dir", type=str,
                       default="/path/to/molmoact_test_lerobot",
                       help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Data collection frame rate")
    args = parser.parse_args()
    # Check input directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Input directory does not exist: {args.data_dir}")
        return
    # Check output directory (supports resume)
    if os.path.exists(args.output_dir):
        print(f"Output directory already exists: {args.output_dir}")
        print("Will check completed data and continue processing...")
    # Load data
    print("Loading molmoact_test data...")
    episodes = load_molmoact_test_data(args.data_dir)
    print(f"Loaded {len(episodes)} episodes")
    if len(episodes) == 0:
        print("Error: No episode data found")
        return
    # Statistics
    total_frames = sum(ep['episode_length'] for ep in episodes)
    tasks = set(ep['task_name'] for ep in episodes)
    print(f"Total frames: {total_frames}")
    print(f"Task types: {tasks}")
    # Convert to LeRobot format
    print("Converting to LeRobot format...")
    create_lerobot_dataset(episodes, args.output_dir, args.fps)
    print("Conversion completed!")
if __name__ == "__main__":
    main()