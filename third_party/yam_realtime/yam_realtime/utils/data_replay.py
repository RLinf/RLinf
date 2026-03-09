import ast
import concurrent
import json
import pickle
import time
from turtle import right
import torch
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, Any, Optional
from yam_realtime.envs.robot_env import RobotEnv

# Import centralized logging utilities
from yam_realtime.utils.logging_utils import log_data_utils, log_replay, log_demo_data_info
import glob
from PIL import Image

# Add matplotlib imports for visualization
import matplotlib.pyplot as plt
from collections import deque

from yam_realtime.utils.data_utils import action_preprocessing
from yam_realtime.utils.logging_utils import log_collect_demos

class DataReplayer():
    """
    DataReplayer class for replaying episodes.

    Args:
        save_format: Format of the saved data (json or npy)
        old_format: Whether the data is in the old format (True) or new format (False)
    """
    def __init__(self, save_format, old_format):
        self.save_format = save_format
        self.old_format = old_format
        self.demo = None

        self.left_camera_key = "left_rgb"
        self.right_camera_key = "right_rgb"
        self.front_camera_key = "front_rgb"

        # for json format only
        # self.main_rgb_paths = None
        # self.wrist_rgb_paths = None
        self.left_rgb_paths = None
        self.right_rgb_paths = None
        self.front_rgb_paths = None
        
        # Visualization setup
        self.fig = None
        self.axs = None
        self.joint_line = None
        self.action_line = None
        self.history_len = 300
        self.joint_history = deque(maxlen=self.history_len)  # Keep last 100 joint positions
        self.action_history = deque(maxlen=self.history_len)  # Keep last 100 actions
        self.step_history = deque(maxlen=self.history_len)    # Keep last 100 step indices
        
        # Initialize matplotlib backend for real-time plotting
        plt.ion()  # Turn on interactive mode
    
    def load_episode(self, root_dir, episode_number):
        # convention: 6 digits for episode number
        episode_number = f"{int(episode_number):06d}"
        try:
            # Check if file exists
            if self.save_format == "json":
                if not self.old_format:
                    demo_dir = f"{root_dir}/{episode_number}/{episode_number}.json"
                else:
                    demo_dir = f"{root_dir}_pickle/{episode_number}.pkl"
                if not os.path.exists(demo_dir):
                    log_data_utils(f"Episode file not found: {demo_dir}", "error")
                log_replay(f"Found episode file: {demo_dir}", "info")
                
                # Load demo data
                log_data_utils(f"Loading demo data from: {demo_dir}", "info")
                try:
                    with open(demo_dir, 'rb') as f:
                        demo = json.load(f)
                    # preprocess the demo data
                    # Log demo structure
                    log_data_utils(f"Demo type: {type(demo)}", "info")
                    if isinstance(demo, dict):
                        log_data_utils(f"Demo keys: {list(demo.keys())}", "info")
                        for key, value in demo.items():
                            if isinstance(value, np.ndarray):
                                log_data_utils(f"  {key}: shape={value.shape}, dtype={value.dtype}", "info")
                            else:
                                log_data_utils(f"  {key}: type={type(value)}, value={value}", "info")
                    elif isinstance(demo, list):
                        log_data_utils(f"Demo length: {len(demo)}", "info")
                        if len(demo) > 0:
                            log_data_utils(f"First item type: {type(demo[0])}", "info")
                            if isinstance(demo[0], dict):
                                log_data_utils(f"First item keys: {list(demo[0].keys())}", "info")
                except Exception as e:
                    demo = {}
                    log_data_utils(f"Failed to load demo data: {str(e)}", "error")
                
                # # Set camera keys if not provided
                # if main_camera_key is None:
                #     # Try to find camera keys in the demo
                #     if isinstance(demo, dict):
                #         camera_keys = [key for key in demo.keys() if 'rgb' in key.lower() or 'camera' in key.lower()]
                #         if camera_keys:
                #             main_camera_key = camera_keys[0]
                #             log_data_utils(f"Auto-detected main camera key: {main_camera_key}", "info")
                #         else:
                #             main_camera_key = "main_camera"
                #             log_data_utils(f"No camera keys found, using default: {main_camera_key}", "warning")
                #     else:
                #         main_camera_key = "main_camera"
                #         log_data_utils(f"Demo is not a dict, using default camera key: {main_camera_key}", "warning")
                
                # if wrist_camera_key is None:
                #     wrist_camera_key = "wrist_camera"
                #     log_data_utils(f"Using default wrist camera key: {wrist_camera_key}", "info")
                
                # self.main_camera_key = main_camera_key
                # self.wrist_camera_key = wrist_camera_key

                left_camera_dir = f"{root_dir}/{episode_number}/{self.left_camera_key}"
                right_camera_dir = f"{root_dir}/{episode_number}/{self.right_camera_key}"
                front_camera_dir = f"{root_dir}/{episode_number}/{self.front_camera_key}"

                log_data_utils(f"Looking for images in: {left_camera_dir}", "info")
                log_data_utils(f"Looking for images in: {right_camera_dir}", "info")
                log_data_utils(f"Looking for images in: {front_camera_dir}", "info")
                
                self.left_rgb_paths = sorted(glob.glob(os.path.join(left_camera_dir, "*.png")))
                self.right_rgb_paths = sorted(glob.glob(os.path.join(right_camera_dir, "*.png")))
                self.front_rgb_paths = sorted(glob.glob(os.path.join(front_camera_dir, "*.png")))

                log_data_utils(f"Found {len(self.left_rgb_paths)} left camera images", "info")
                log_data_utils(f"Found {len(self.right_rgb_paths)} right camera images", "info")
                log_data_utils(f"Found {len(self.front_rgb_paths)} front camera images", "info")

                if len(self.left_rgb_paths) == 0:
                    log_data_utils(f"WARNING: No left camera images found in {left_camera_dir}", "warning")
                if len(self.right_rgb_paths) == 0:
                    log_data_utils(f"WARNING: No right camera images found in {right_camera_dir}", "warning")
                if len(self.front_rgb_paths) == 0:
                    log_data_utils(f"WARNING: No front camera images found in {front_camera_dir}", "warning")
                
                # # Try to load image paths
                # if not self.old_format:
                #     main_camera_dir = f"{root_dir}/{self.save_format}/{main_camera_key}/{episode_number}"
                #     wrist_camera_dir = f"{root_dir}/{self.save_format}/{wrist_camera_key}/{episode_number}"
                # else:
                #     main_camera_dir = f"{root_dir}/{main_camera_key}/{episode_number}"
                #     wrist_camera_dir = f"{root_dir}/{wrist_camera_key}/{episode_number}"
                    
                
                # log_data_utils(f"Looking for images in: {main_camera_dir}", "info")
                # log_data_utils(f"Looking for images in: {wrist_camera_dir}", "info")
                
                # self.main_rgb_paths = sorted(glob.glob(os.path.join(main_camera_dir, "*.png")))
                # self.wrist_rgb_paths = sorted(glob.glob(os.path.join(wrist_camera_dir, "*.png")))
                
                # log_data_utils(f"Found {len(self.main_rgb_paths)} main camera images", "info")
                # log_data_utils(f"Found {len(self.wrist_rgb_paths)} wrist camera images", "info")
                
                # if len(self.main_rgb_paths) == 0:
                #     log_data_utils(f"WARNING: No main camera images found in {main_camera_dir}", "warning")
                # if len(self.wrist_rgb_paths) == 0:
                #     log_data_utils(f"WARNING: No wrist camera images found in {wrist_camera_dir}", "warning")
                
                # Process actions (this code doesn't do anything)
                if isinstance(demo, dict) and "left_raw_action" in demo and "right_raw_action" in demo:
                    actions = {"left": demo[0]["left_raw_action"], "right": demo[0]["right_raw_action"]}
                    log_data_utils(f"Processing actions with shape: {actions.shape}", "info")
                    if self.old_format:
                        log_data_utils("Processing old format actions", "info")
                        # NOTE: The value of the demo["action"] is the raw vr controller delta act, so need to preprocess.
                        actions = demo["action"]
                        actions = action_preprocessing(demo, actions)  # delta action
                        demo["action"] = actions
                        log_data_utils(f"Processed actions with shape: {actions.shape}", "info")
                else:
                    log_data_utils("No actions found in demo or demo is not a dict", "warning")

                def convert_to_int_list(value):
                    # Helper function to convert stringified lists to integer lists or string numbers to integers.
                    if isinstance(value, str):
                        # Check if the string is a stringified list
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                # Try to evaluate the string as a list
                                result = ast.literal_eval(value)
                                if isinstance(result, list):
                                    # Convert all items of the list to integers if possible
                                    return [float(x) if isinstance(x, (int, float)) else 0 for x in result]
                                else:
                                    # Return the original value if it's not a list
                                    return value
                            except (ValueError, SyntaxError) as e:
                                # If we fail to parse the stringified list, log the error
                                log_data_utils(f"Error converting stringified list '{value}' to list: {e}", "warning")
                                return value  # Return the original value if conversion fails
                        else:
                            # If it's just a number in string format, convert it to an integer
                            try:
                                return int(value)
                            except ValueError:
                                # If it's not a valid number, return it as is
                                return value
                    return value

                resolved_demo = []
    
                for step_data in demo:
                    resolved_step = {}
                    
                    # Iterate over each key-value pair in the current step data
                    for key, value in step_data.items():
                        resolved_step[key] = convert_to_int_list(value)  # Apply conversion function to each value
                    
                    resolved_demo.append(resolved_step)

                self.demo = resolved_demo

                # Function to load and convert image to numpy array
                def load_image(path):
                    try:
                        return np.array(Image.open(path))
                    except Exception as e:
                        log_data_utils(f"Error processing image {path}: {str(e)}", "error")
                        return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Use map to load all images concurrently for each path
                    left_images = list(executor.map(load_image, self.left_rgb_paths))
                    right_images = list(executor.map(load_image, self.right_rgb_paths))
                    front_images = list(executor.map(load_image, self.front_rgb_paths))

                # Insert images into demo (ensure each image corresponds to the correct demo step)
                for i, demo_step in enumerate(self.demo):
                    if left_images[i] is not None:
                        demo_step["image_left_rgb"] = left_images[i]
                    if right_images[i] is not None:
                        demo_step["image_right_rgb"] = right_images[i]
                    if front_images[i] is not None:
                        demo_step["image_front_rgb"] = front_images[i]
                # self.demo[self.main_camera_key] = np.stack([np.array(Image.open(path)) for path in self.main_rgb_paths])
                # self.demo[self.wrist_camera_key] = np.stack([np.array(Image.open(path)) for path in self.wrist_rgb_paths])
                log_data_utils(f"Replaying episode from: {demo_dir}", "info")

                # Log comprehensive data information
                log_demo_data_info(self.demo[0], demo_dir)
            elif self.save_format == "npy":
                demo_dir = os.path.join(root_dir, self.save_format, f"{episode_number}.npy")
                with open(demo_dir, 'rb') as f:
                    self.demo = pickle.load(f)
                log_demo_data_info(self.demo, demo_dir)
        except Exception as e:
            log_data_utils(f"Error loading demo data: {str(e)}", "error")
            return False
        
        return True
    
    def get_demo_length(self):
        if self.demo is None:
            log_data_utils("No demo data loaded. Please load a demo first.", "error")
            return 0
        return len(self.demo)
    
    def get_observation(self, step_idx: int) -> Dict[str, Any]:
        """
        Format the observation dictionary.
        """
        # Create observation dictionary from demo data
        obs = {}
        # print(self.demo)
        # for key in self.demo.keys():
        #     obs[key] = self.demo[key][step_idx]
        # return obs
        for key in self.demo[step_idx].keys():
            obs[key] = self.demo[step_idx][key]
        return obs
        # step_data = self.demo[step_idx]  # get the dict for the current timestep

    # Convert stringified lists to actual lists
        # for key, value in step_data.items():
        #     if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        #         obs[key] = ast.literal_eval(value)
        #     else:
        #         obs[key] = value

        # return obs
    
    def get_instruction(self):
        if self.demo is None:
            log_data_utils("No demo data loaded. Please load a demo first.", "error")
            return None
        if "language_instruction" not in self.demo:
            log_data_utils("No language instruction found in demo.", "error")
            return None
        return self.demo["language_instruction"]
    
    # def get_episode_all_obs(self):
    #     if self.demo is None:
    #         log_data_utils("No demo data loaded. Please load a demo first.", "error")
    #         return None
    #     state = [np.concatenate([self.demo[i]["left_joint"], self.demo[i]["right_joint"]]) for i in range(len(self.demo))]
        
    #     left_rgb = [self.demo[i]["image_left_rgb"] for i in range(len(self.demo))]
    #     front_rgb = [self.demo[i]["image_front_rgb"] for i in range(len(self.demo))]
    #     right_rgb = [self.demo[i]["image_right_rgb"] for i in range(len(self.demo))]
    #     obs_dict = {
    #         "observation.state": state,
    #         "observation.images.camera_0": left_rgb,
    #         "observation.images.camera_1": right_rgb,
    #         "observation.images.camera_2": front_rgb,
    #     }
    #     return obs_dict
 
    def replay(self, env: RobotEnv, visual: bool = False, robot_trajectory: bool = True):
        if self.demo is None:
            log_data_utils("No demo data loaded. Please load a demo first.", "error")
            return
        
        # demo_length = self.demo["left_raw_action"].shape[0]
        demo_length = len(self.demo)
        actions = [
            {
                "left": {"pos": np.array(ast.literal_eval(self.demo[i]["left_raw_action"]) if isinstance(self.demo[i]["left_raw_action"], str) else self.demo[i]["left_raw_action"], dtype=float)},
                "right": {"pos": np.array(ast.literal_eval(self.demo[i]["right_raw_action"]) if isinstance(self.demo[i]["right_raw_action"], str) else self.demo[i]["right_raw_action"], dtype=float)},
            }
            for i in range(demo_length)
        ]
        # print(actions)
        input(f"Press Enter to replay the episode or Ctrl+C to exit...")
        
        log_data_utils(f"Starting replay of {demo_length} steps...", "info")
        # log_data_utils(f"Demo keys: {list(self.demo.keys())}", "info")
        log_data_utils(f"Demo keys: {list(self.demo[0].keys())}", "info")
        log_data_utils(f"Left camera key: {self.left_camera_key}", "info")
        log_data_utils(f"Right camera key: {self.right_camera_key}", "info")
        log_data_utils(f"Front camera key: {self.front_camera_key}", "info")
        # log_data_utils(f"Main camera key: {self.main_camera_key}", "info")
        # log_data_utils(f"Wrist camera key: {self.wrist_camera_key}", "info")

        # if self.main_camera_key is not None:
        #     log_data_utils(f"Main camera images: {len(self.main_rgb_paths)}", "info")
        # if self.wrist_camera_key is not None:
        #     log_data_utils(f"Wrist camera images: {len(self.wrist_rgb_paths)}", "info")
        if self.left_camera_key is not None:
            log_data_utils(f"Left camera images: {len(self.left_rgb_paths)}", "info")
        if self.right_camera_key is not None:
            log_data_utils(f"Right camera images: {len(self.right_rgb_paths)}", "info")
        if self.front_camera_key is not None:
            log_data_utils(f"Front camera images: {len(self.front_rgb_paths)}", "info")
        
        try:
            def smooth_move_while_inference_envstep(env: RobotEnv, action):
                current_left_joint = env.robot("left").get_joint_pos()
                current_right_joint = env.robot("right").get_joint_pos()

                target_left_joint = action["left"]["pos"]
                target_right_joint = action["right"]["pos"]

                steps = 10
                obs = None
                for i in range(steps + 1):
                    alpha = i / steps  # Interpolation factor
                    target_pos_left = (1 - alpha) * current_left_joint + alpha * target_left_joint  # Linear interpolation
                    target_pos_right = (1 - alpha) * current_right_joint + alpha * target_right_joint
                    obs = env.step({"left" : {"pos" : target_pos_left}, "right" : {"pos": target_pos_right}})
                    time.sleep(0.5 / steps)

                return obs

            if robot_trajectory:
                for step_idx in tqdm(range(demo_length), desc="Replaying episode"):
                    act = actions[step_idx]

                    obs = self.get_observation(step_idx)
                    
                    # Log step information
                    if step_idx % 15 == 0:  # Log every 10 steps to avoid spam
                        log_data_utils(f"Step {step_idx}/{demo_length}: action={act['left']['pos'][:3]}, gripper={act['left']['pos'][-1]:.3f}", "data_info")
                        log_data_utils(f"Step {step_idx}/{demo_length}: action={act['right']['pos'][:3]}, gripper={act['right']['pos'][-1]:.3f}", "data_info")
                    
                    # # Visualize episode if requested
                    # if visual:
                    #     self.visualize_episode(obs, step_idx, act)
                    
                    # env.step(act)
                    obs = smooth_move_while_inference_envstep(env, act)
                    
                    # Handle window events for visualization
                    # if visual:
                    #     plt.pause(0.001)  # Small pause to allow matplotlib to update
                    #     if plt.waitforbuttonpress(timeout=0.001):  # Check for key press
                    #         key = plt.gcf().canvas.get_key()
                    #         if key == 'q':  # Press 'q' to quit
                    #             log_data_utils("Replay interrupted by user (pressed 'q')", "warning")
                    #             break
                    #         elif key == 'p':  # Press 'p' to pause
                    #             log_data_utils("Replay paused. Press any key to continue...", "info")
                                # plt.waitforbuttonpress()
            
            # Initialize visualization if requested
            if visual:
                self._init_visualization()
                for step_idx in tqdm(range(demo_length), desc="Replaying episode"):
                    act = actions[step_idx]

                    obs = self.get_observation(step_idx)
                    
                    # Log step information
                    if step_idx % 15 == 0:  # Log every 10 steps to avoid spam
                        log_data_utils(f"Step {step_idx}/{demo_length}: action={act['left']['pos'][:3]}, gripper={act['left']['pos'][-1]:.3f}", "data_info")
                        log_data_utils(f"Step {step_idx}/{demo_length}: action={act['right']['pos'][:3]}, gripper={act['right']['pos'][-1]:.3f}", "data_info")
                    
                    # Visualize episode if requested
                    if visual:
                        self.visualize_episode(obs, step_idx, act)
                    
                    # env.step(act)
                    
                    # Handle window events for visualization
                    if visual:
                        plt.pause(0.001)  # Small pause to allow matplotlib to update
                        if plt.waitforbuttonpress(timeout=0.001):  # Check for key press
                            key = plt.gcf().canvas.get_key()
                            if key == 'q':  # Press 'q' to quit
                                log_data_utils("Replay interrupted by user (pressed 'q')", "warning")
                                break
                            elif key == 'p':  # Press 'p' to pause
                                log_data_utils("Replay paused. Press any key to continue...", "info")
                                plt.waitforbuttonpress()
                
                log_data_utils("Episode replay completed successfully!", "success")
            
        except Exception as e:
            log_data_utils(f"Error during replay: {str(e)}", "error")
            raise
        finally:
            # Clean up visualization
            if visual:
                self._cleanup_visualization()


    def visualize_episode(self, obs: Dict[str, Any], step_idx: int, action: np.ndarray):
        """
        Main visualization function that orchestrates all visualization components.
        
        Args:
            obs: Observation dictionary containing image data and metadata
            step_idx: Current step index
            action: Current action being executed
        """
        # Update history for real-time plotting
        self.step_history.append(step_idx)
        
        # Visualize images
        self._visualize_image(obs)
        
        # # Visualize metadata
        # self._visualize_metadata(obs, step_idx)
        
        # # Visualize state (joint positions)
        # if "lowdim_qpos" in obs:
        #     self._visualize_joint_state(obs)
        
        # # Visualize action
        # self._visualize_action_data(action)
        
        # Update the plot
        plt.tight_layout()
        plt.draw()

    def _init_visualization(self):
        """Initialize matplotlib figure and subplots for visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout: 2 rows, 3 columns
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
        
        # Image subplots (top row)
        self.ax_images = []
        for i in range(3):
            ax = self.fig.add_subplot(gs[0, i])
            self.ax_images.append(ax)
        
        # # Metadata subplot (bottom left)
        # self.ax_metadata = self.fig.add_subplot(gs[1, 0])
        
        # # Joint positions subplot (bottom middle)
        # self.ax_joints = self.fig.add_subplot(gs[1, 1])
        
        # # Actions subplot (bottom right)
        # self.ax_actions = self.fig.add_subplot(gs[1, 2])
        
        # Set titles
        # self.ax_images[0].set_title("Main Camera")
        # self.ax_images[1].set_title("Wrist Camera")
        # self.ax_images[2].set_title("Depth Camera (if available)")
        # self.ax_metadata.set_title("Episode Metadata")
        # self.ax_joints.set_title("Joint Positions (Real-time)")
        # self.ax_actions.set_title("Actions (Real-time)")

        self.ax_images[0].set_title("Left Camera")
        self.ax_images[1].set_title("Front Camera")
        self.ax_images[2].set_title("Right Camera")
        
        # # Initialize joint and action plots
        # self._init_joint_plot()
        # self._init_action_plot()
        
        plt.tight_layout()

    def _cleanup_visualization(self):
        """Clean up matplotlib resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None

    def _visualize_image(self, obs: Dict[str, Any]):
        """
        Visualize RGB and depth images using matplotlib.
        
        Args:
            obs: Observation dictionary containing image data
        """
        # Clear previous images
        for ax in self.ax_images:
            ax.clear()
        
        # Debug logging
        log_data_utils(f"Visualizing images for keys: {list(obs.keys())}", "debug")
        
        # Find image keys
        rgb_keys = [key for key in obs.keys() if 'rgb' in key.lower()]
        # depth_keys = [key for key in obs.keys() if 'depth' in key.lower()]
        
        log_data_utils(f"Found RGB keys: {rgb_keys}", "debug")
        # log_data_utils(f"Found depth keys: {depth_keys}", "debug")
        
        # Handle RGB images
        if rgb_keys:
            for i, key in enumerate(rgb_keys):
                if i < len(self.ax_images):
                    img_data = obs[key]
                    
                    # Skip if image is None
                    if img_data is None:
                        self.ax_images[i].text(0.5, 0.5, f"No {key} image", 
                                             transform=self.ax_images[i].transAxes,
                                             ha='center', va='center',
                                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                        self.ax_images[i].set_title(f"{key} (Not Available)")
                        self.ax_images[i].axis('off')
                        continue

                    # Convert PIL Image to numpy array if needed
                    if isinstance(img_data, Image.Image):
                        img_data = np.array(img_data)
                    
                    # Ensure img_data is a numpy array with proper dtype
                    if not isinstance(img_data, np.ndarray):
                        log_data_utils(f"Warning: img_data is not a numpy array, type: {type(img_data)}", "warning")
                        continue
                    
                    # Convert object dtype to proper numeric dtype if needed
                    if img_data.dtype == np.dtype('object'):
                        try:
                            img_data = img_data.astype(np.float32)
                        except:
                            log_data_utils(f"Warning: Could not convert image data to float32", "warning")
                            continue
                    
                    # Handle different image formats
                    if len(img_data.shape) == 3:
                        if img_data.shape[2] == 3:
                            # RGB image
                            self.ax_images[i].imshow(img_data)
                        elif img_data.shape[2] == 4:
                            # RGBA image, convert to RGB
                            self.ax_images[i].imshow(img_data[:, :, :3])
                    else:
                        # Grayscale image
                        self.ax_images[i].imshow(img_data, cmap='gray')
                    
                    self.ax_images[i].set_title(f"{key}")
                    self.ax_images[i].axis('off')
        
        # # Handle depth images
        # if depth_keys and len(rgb_keys) < 2:
        #     for i, key in enumerate(depth_keys):
        #         if i + len(rgb_keys) < len(self.ax_images):
        #             img_data = obs[key]
                    
        #             # Convert PIL Image to numpy array if needed
        #             if isinstance(img_data, Image.Image):
        #                 img_data = np.array(img_data)
                    
        #             # Normalize depth for visualization
        #             if img_data.dtype == np.uint16:
        #                 img_display = img_data / 65535.0
        #             else:
        #                 img_display = img_data.astype(float) / img_data.max()
                    
        #             self.ax_images[i + len(rgb_keys)].imshow(img_display, cmap='viridis')
        #             self.ax_images[i + len(rgb_keys)].set_title(f"{key}")
        #             self.ax_images[i + len(rgb_keys)].axis('off')
        
        # # Hide unused subplots
        # for i in range(len(rgb_keys) + len(depth_keys), len(self.ax_images)):
        #     self.ax_images[i].set_visible(False)
        for i in range(len(rgb_keys), len(self.ax_images)):
            self.ax_images[i].set_visible(False)

    # def _visualize_metadata(self, obs: Dict[str, Any], step_idx: int):
    #     """
    #     Visualize episode metadata including language instruction and teleop device.
        
    #     Args:
    #         obs: Observation dictionary containing metadata
    #         step_idx: Current step index
    #     """
    #     self.ax_metadata.clear()
    #     self.ax_metadata.set_title("Episode Metadata")
        
    #     # Create text content
    #     text_content = []
    #     text_content.append(f"Step: {step_idx}")
        
    #     # Add language instruction if available
    #     if "language_instruction" in obs:
    #         instruction = obs["language_instruction"]
    #         if isinstance(instruction, (list, np.ndarray)):
    #             instruction = instruction[0] if len(instruction) > 0 else "N/A"
    #         text_content.append(f"Task: {instruction}")
        
    #     # Add teleop device if available
    #     if "teleop_device" in obs:
    #         device = obs["teleop_device"]
    #         if isinstance(device, (list, np.ndarray)):
    #             device = device[0] if len(device) > 0 else "N/A"
    #         text_content.append(f"Device: {device}")
        
    #     # Add action information
    #     if "action" in obs:
    #         action = obs["action"]
    #         if isinstance(action, np.ndarray):
    #             text_content.append(f"Action shape: {action.shape}")
    #             text_content.append(f"Action range: [{action.min():.4f}, {action.max():.4f}]")
        
    #     # Add joint information
    #     if "lowdim_qpos" in obs:
    #         qpos = obs["lowdim_qpos"]
    #         if isinstance(qpos, np.ndarray):
    #             text_content.append(f"Joints: {qpos.shape}")
        
    #     # Display text
    #     text_str = "\n".join(text_content)
    #     self.ax_metadata.text(0.05, 0.95, text_str, transform=self.ax_metadata.transAxes,
    #                          fontsize=10, verticalalignment='top',
    #                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    #     self.ax_metadata.axis('off')

    # def _init_joint_plot(self):
    #     """Initialize the joint positions plot."""
    #     self.ax_joints.clear()
    #     self.ax_joints.set_title("Joint Positions (Real-time)")
    #     self.ax_joints.set_xlabel("Time Step")
    #     self.ax_joints.set_ylabel("Joint Position (rad)")
    #     self.ax_joints.grid(True, alpha=0.3)
        
    #     # Initialize empty lines for each joint
    #     self.joint_lines = []
    #     joint_names = [f"Joint {i+1}" for i in range(8)]  # Assuming 8 joints
    #     colors = plt.cm.tab10(np.linspace(0, 1, 8))
        
    #     for i, (name, color) in enumerate(zip(joint_names, colors)):
    #         line, = self.ax_joints.plot([], [], label=name, color=color, linewidth=1.5)
    #         self.joint_lines.append(line)
        
    #     self.ax_joints.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # def _visualize_joint_state(self, obs: Dict[str, Any]):
    #     """
    #     Visualize joint positions in real-time using a 2D graph.
        
    #     Args:
    #         obs: Observation dictionary containing joint position data
    #     """
    #     if "lowdim_qpos" not in obs:
    #         return
        
    #     qpos = obs["lowdim_qpos"]
    #     if not isinstance(qpos, np.ndarray):
    #         return
        
    #     # Update joint history
    #     self.joint_history.append(qpos)
        
    #     # Update plot data
    #     if len(self.joint_history) > 1:
    #         steps = list(range(len(self.joint_history)))
    #         joint_data = np.array(list(self.joint_history))
            
    #         # Update each joint line
    #         for i in range(min(len(self.joint_lines), joint_data.shape[1])):
    #             self.joint_lines[i].set_data(steps, joint_data[:, i])
            
    #         # Update axis limits
    #         self.ax_joints.set_xlim(0, len(self.joint_history))
    #         if joint_data.size > 0:
    #             y_min, y_max = joint_data.min(), joint_data.max()
    #             margin = (y_max - y_min) * 0.1
    #             self.ax_joints.set_ylim(y_min - margin, y_max + margin)

    # def _init_action_plot(self):
    #     """Initialize the actions plot."""
    #     self.ax_actions.clear()
    #     self.ax_actions.set_title("Actions (Real-time)")
    #     self.ax_actions.set_xlabel("Time Step")
    #     self.ax_actions.set_ylabel("Action Value")
    #     self.ax_actions.grid(True, alpha=0.3)
        
    #     # Initialize empty lines for each action dimension
    #     self.action_lines = []
    #     action_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]
    #     colors = plt.cm.Set1(np.linspace(0, 1, 7))
        
    #     for i, (name, color) in enumerate(zip(action_names, colors)):
    #         line, = self.ax_actions.plot([], [], label=name, color=color, linewidth=1.5)
    #         self.action_lines.append(line)
        
    #     self.ax_actions.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # def _visualize_action_data(self, action: np.ndarray):
    #     """
    #     Visualize actions in real-time using a 2D graph.
        
    #     Args:
    #         action: Current action array
    #     """
    #     if not isinstance(action, np.ndarray):
    #         return
        
    #     # Update action history
    #     self.action_history.append(action)
        
    #     # Update plot data
    #     if len(self.action_history) > 1:
    #         steps = list(range(len(self.action_history)))
    #         action_data = np.array(list(self.action_history))
            
    #         # Update each action dimension line
    #         for i in range(min(len(self.action_lines), action_data.shape[1])):
    #             self.action_lines[i].set_data(steps, action_data[:, i])
            
    #         # Update axis limits
    #         self.ax_actions.set_xlim(0, len(self.action_history))
    #         if action_data.size > 0:
    #             y_min, y_max = action_data.min(), action_data.max()
    #             margin = (y_max - y_min) * 0.1
    #             self.ax_actions.set_ylim(y_min - margin, y_max + margin)

    # def _visualize_state(self, state: Dict[str, Any]):
    #     """
    #     Visualize state information (joint positions) in real-time.
        
    #     Args:
    #         state: State dictionary containing joint position data
    #     """
    #     if "lowdim_qpos" in state:
    #         self._visualize_joint_state(state)

    # def _visualize_action(self, action: np.ndarray):
    #     """
    #     Visualize action information in real-time.
        
    #     Args:
    #         action: Action array to visualize
    #     """
    #     self._visualize_action_data(action)

def run_replay_mode(cfg, env: RobotEnv, data_replayer: DataReplayer, logger_func=log_collect_demos):
    """
    Run interactive replay mode that can be used by both policy_eval.py and replay.py
    
    Args:
        cfg: Configuration object
        env: Robot environment
        data_replayer: DataReplayer instance
    """
    logger_func("Starting replay mode", "important")
    base_dir = cfg.storage.base_dir
    last_task_directory = None
    last_episode_number = None
    
    while True:
        try:
            env.reset()
            logger_func("Environment reset successfully", "success")

            # Get task directory
            if last_task_directory:
                task_directory = input(f"Enter the record_date/task_directory (e.g. date_723/debug) [last: {last_task_directory}]: ")
                if not task_directory.strip():
                    task_directory = last_task_directory
            else:
                task_directory = input("Enter the record_date/task_directory (e.g. date_723/debug): ")
            
            # Get episode number
            if last_episode_number:
                episode_number = input(f"Enter the episode number [last: {last_episode_number}]: ")
                if not episode_number.strip():
                    episode_number = last_episode_number
            else:
                episode_number = input("Enter the episode number: ")
            
            # Store for next iteration
            last_task_directory = task_directory
            last_episode_number = episode_number
            
            root_dir = f"{base_dir}/{task_directory}"

            # Ask user for visualization options
            visual_choice = input("Enable image visualization? (y/n): ").lower().strip()
            visual = visual_choice in ['y', 'yes']
            
            data_replayer.load_episode(root_dir, episode_number, main_camera_key=cfg.camera.main_camera_key, wrist_camera_key=cfg.camera.wrist_camera_key)
            data_replayer.replay(env, visual=visual)

            logger_func("Episode replay completed", "success")
            
            # Ask if user wants to repeat the same episode
            repeat = input("Repeat the same episode? (y/n): ").lower().strip()
            if repeat in ['y', 'yes']:
                logger_func("Repeating the same episode...", "info")
                continue
            elif repeat in ['n', 'no']:
                logger_func("Moving to next episode...", "info")
                continue
            else:
                logger_func("Invalid input. Moving to next episode...", "warning")
                continue
                
        except KeyboardInterrupt:
            logger_func("Replay interrupted by user. Exiting...", "info")
            break
        except Exception as e:
            logger_func(f"Error during replay: {str(e)}", "error")
            import traceback
            logger_func(f"Full traceback:\n{traceback.format_exc()}", "error")
            continue

def replay_episode_pickle(demo_dir, env: RobotEnv):
    with open(demo_dir, 'rb') as f:
        demo = pickle.load(f)
   
    actions = demo["action"]

    demo_length = actions.shape[0]
    for step_idx in tqdm(range(demo_length)):
        act = actions[step_idx]
        env.step(act)