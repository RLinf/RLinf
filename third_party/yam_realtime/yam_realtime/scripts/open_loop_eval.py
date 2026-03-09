# Eval diffusion policy for a single task.
from copy import deepcopy
import os
import sys
from tkinter import Y

from PIL import Image
import numpy as np
import torch

sys.path.append('/home/sean/Desktop/YAM')
sys.path.append('/home/sean/Desktop/YAM/yam_realtime')
sys.path.append('/home/sean/Desktop/YAM/lerobot')
"""
Main launch script for YAM realtime robot control environment.
"""

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.factory import LeRobotDatasetMetadata

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R

import tyro

from yam_realtime.utils.data_replay import DataReplayer, log_collect_demos
from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.envs.robot_env import RobotEnv
from yam_realtime.robots.robot import Robot
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.utils.launch_utils import (
    cleanup_processes,
    initialize_agent,
    initialize_robots,
    initialize_sensors,
    setup_can_interfaces,
    setup_logging,
)


@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    collection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_viser_bimanual.yaml",)


DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"


def main(args: Args) -> None:
    """
    Main launch entrypoint.

    1. Load configuration from yaml file
    2. Initialize sensors (cameras, force sensors, etc.)
    3. Setup CAN interfaces (for YAM communication)
    4. Initialize robots (hardware interface)
    5. Initialize agent (e.g. teleoperated control, policy control, etc.)
    6. Create environment
    7. Run control loop
    """
    # Setup logging and get logger
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")

    server_processes = []

    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        # Pop policy config from configs_dict (required for diffusion policy)
        # If you don't pop it, the instantiate function will instantiate the policy config as a dataclass and it will cause error for later policy initialization
        policy_cfg = configs_dict.pop("policy")
        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=rate,
        )

        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')

        logger.info("Starting control loop...")
        ds_meta = LeRobotDatasetMetadata(
            repo_id=policy_cfg["repo_id"]
        )
        policy = DiffusionPolicy.from_pretrained(policy_cfg["checkpoint_path"], dataset_stats=ds_meta.stats)
        policy.to('cuda')
        policy.eval()

        logger.info("Start replay...")
        task = input("Enter task to replay: ")
        episode_number = int(input("Enter episode number to replay: "))

        data_replayer = DataReplayer(save_format=configs_dict['storage']['save_format'], old_format=configs_dict['storage']['old_format'])
        data_replayer.load_episode(configs_dict['storage']['base_dir'] + '/' + task, episode_number)


        # print(policy.config.input_features)
        # print(policy.config.output_features)

        _run_control_loop(env, main_config, policy, data_replayer, agent)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        # Cleanup
        logger.info("Shutting down...")
        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')
        if "env" in locals():
            env.close()
        if "agent" in locals():
            cleanup_processes(agent, server_processes)

TARGET_HEIGHT = 256
TARGET_WIDTH = 342

def preprocess_observation(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Convert raw observations to tensors for diffusion policy.
    Handles multiple camera inputs and normalizes images.
    """
    log_collect_demos("Preparing policy input...", "info")
    # Make numpy arrays contiguous before converting to torch to avoid negative-stride errors
    state_np = np.ascontiguousarray(np.concatenate([observation["left_joint"], observation["right_joint"]]))

    def preprocess_camera_np(img_np: np.ndarray) -> torch.Tensor:
    # Ensure contiguous array
        img_np = np.ascontiguousarray(img_np)
        
        # Convert to PIL
        img_pil = Image.fromarray(img_np)
        
        # Resize
        img_resized = img_pil.resize((TARGET_WIDTH, TARGET_HEIGHT))  # width, height
        
        # Convert back to numpy
        img_resized_np = np.array(img_resized)
    
        return img_resized_np
    cam0_np = np.ascontiguousarray(preprocess_camera_np(observation['image_left_rgb']))
    cam1_np = np.ascontiguousarray(preprocess_camera_np(observation['image_right_rgb']))
    cam2_np = np.ascontiguousarray(preprocess_camera_np(observation['image_front_rgb']))

    input_dict = {
        "observation.state": torch.from_numpy(state_np).float().cuda().unsqueeze(0),  # [1, state_dim])
        # convert images from HWC numpy -> CHW torch
        "observation.images.camera_left": torch.from_numpy(cam0_np).float().permute(2, 0, 1).cuda().unsqueeze(0),  # [1, C, H, W]
        "observation.images.camera_right": torch.from_numpy(cam1_np).float().permute(2, 0, 1).cuda().unsqueeze(0),  # [1, C, H, W]
        "observation.images.camera_front": torch.from_numpy(cam2_np).float().permute(2, 0, 1).cuda().unsqueeze(0),  # [1, C, H, W]
    }
    for k,v in input_dict.items():
        print(f"Input {k}: {v.shape}", "data_info")

    return input_dict 


def reset_robot(agent: Agent, env: RobotEnv, side: str, target_joint_positions: Optional[np.ndarray] = None):
    agent.act({})
    current_pos = env.robot(side).get_joint_pos()
    target_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    steps = 50
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
        env.robot(side).command_joint_pos(target_pos)
        time.sleep(2 / steps)

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

def _run_control_loop(env: RobotEnv, config: LaunchConfig, policy: DiffusionPolicy, data_replayer: DataReplayer, agent: Agent) -> None:
    """
    Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    logger = logging.getLogger(__name__)

    # Init environment and warm up agent
    policy.reset()
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")

    # Main control loop
    demo_length = data_replayer.get_demo_length()
    obs_index = 0
    while obs_index < demo_length:
        obs = data_replayer.get_observation(obs_index)
        input_dict = preprocess_observation(obs)
        log_collect_demos("Running policy inference...", "info")
        start_time = time.time()
        actions = policy.select_action(input_dict)
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")

        print("action.shape:", actions.shape)
        actions = actions.squeeze(0).detach().cpu().numpy()
        delta_act = {'left': actions[:8], 'right': actions[8:]}
        obs['delta_action'] = delta_act
        action = agent.act(obs)
        # env.step(action)
        obs = smooth_move_while_inference_envstep(env, action)
        obs_index += 1

if __name__ == "__main__":
    main(tyro.cli(Args))
