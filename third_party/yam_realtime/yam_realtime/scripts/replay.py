# Collect data for a single task. Pair with oculus viser agent.
import os
import sys

sys.path.append('/home/sean/Desktop/YAM')
sys.path.append('/home/sean/Desktop/YAM/yam_realtime')

from yam_realtime.utils.data_replay import DataReplayer
from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.launch_utils import cleanup_processes, initialize_agent, initialize_robots, initialize_sensors, setup_can_interfaces, setup_logging

import tqdm
from yam_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from yam_realtime.agents.teleoperation.oculus_viser_agent import OculusViserAgent
from yam_realtime.envs.robot_env import RobotEnv
from yam_realtime.utils.data_saver import DataSaver

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.robots.robot import Robot
import logging
import time
import tyro
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    config_path: Tuple[str, ...] = ("~/yam_realtime/configs/yam_record_replay.yaml",)
    

def main(args: Args):
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")

    server_processes = []
    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        # logger.info("Initializing sensors...")
        # camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict={},
            control_rate_hz=rate,
        )

        logger.info("Start replay...")
        task = input("Enter task to replay: ")
        episode_number = int(input("Enter episode number to replay: "))
        robot_trajectory = input("Replay robot trajectory? (y/n): ")
        if robot_trajectory == 'y':
            robot_trajectory = True
        else:
            robot_trajectory = False
        camera_trajectory = input("Replay camera trajectory? (y/n): ")
        if camera_trajectory == 'y':
            camera_trajectory = True
        else:
            camera_trajectory = False

        data_replayer = DataReplayer(save_format=configs_dict['storage']['save_format'], old_format=configs_dict['storage']['old_format'])
        data_replayer.load_episode(configs_dict['storage']['base_dir'] + '/' + task, episode_number)
        reset_robot(agent, env, 'left', data_replayer.demo[0]['left_raw_action'])
        reset_robot(agent, env, 'right', data_replayer.demo[0]['right_raw_action'])
        data_replayer.replay(env, camera_trajectory, robot_trajectory)
        

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


    logger.info(f"Finished collecting data")

# slowly move the robot back to original position (or target joint positions)
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

if __name__ == "__main__":
    main(tyro.cli(Args))

