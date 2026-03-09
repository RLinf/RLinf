#!/usr/bin/env python3
# Collect data for a single task. Pair with oculus viser agent.

# # CRITICAL: Redirect stderr at OS level to filter OpenSSL warnings
# import sys
# import os
# import io

# # Set environment variable to disable RDRAND
# os.environ['OPENSSL_ia32cap'] = '~0x200000200000000'

# # Create a filtering wrapper for stderr at the file descriptor level
# class StderrFilter:
#     def __init__(self, fd):
#         self.fd = fd
#         self.buffer = b''
        
#     def write(self, data):
#         if isinstance(data, str):
#             data = data.encode('utf-8')
#         # Filter out OpenSSL warnings
#         lines = data.split(b'\n')
#         filtered_lines = []
#         for line in lines:
#             if b'CPU random generator' not in line and b'RDRND generated' not in line:
#                 filtered_lines.append(line)
#         if filtered_lines:
#             filtered_data = b'\n'.join(filtered_lines)
#             os.write(self.fd, filtered_data)
#         return len(data)
    
#     def fileno(self):
#         return self.fd

# # Redirect stderr file descriptor through our filter
# original_stderr_fd = sys.stderr.fileno()
# pipe_read, pipe_write = os.pipe()

# # Fork a thread to filter stderr
# import threading
# def filter_stderr():
#     while True:
#         try:
#             data = os.read(pipe_read, 4096)
#             if not data:
#                 break
#             # Filter the data
#             if b'CPU random generator' not in data and b'RDRND generated' not in data:
#                 os.write(original_stderr_fd, data)
#         except:
#             break

# # Save original stderr and redirect to pipe
# saved_stderr = os.dup(2)
# os.dup2(pipe_write, 2)
# os.close(pipe_write)

# # Start filtering thread
# filter_thread = threading.Thread(target=filter_stderr, daemon=True)
# filter_thread.start()

import sys
import os
import io
import threading

import warnings
from tkinter import Y

# Suppress warnings
warnings.filterwarnings('ignore')

# Monkey-patch socket to suppress "Bad file descriptor" errors during cleanup
import socket
_original_socket_close = socket.socket._real_close
def _patched_socket_close(self):
    try:
        _original_socket_close(self)
    except OSError as e:
        if e.errno not in [9, 88]:  # Suppress "Bad file descriptor" and "Socket operation on non-socket"
            raise
socket.socket._real_close = _patched_socket_close

# Add paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
yam_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
yam_realtime_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(yam_root)
sys.path.append(yam_realtime_root)

import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R

from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.launch_utils import cleanup_processes, initialize_agent, initialize_robots, initialize_sensors, setup_can_interfaces, setup_logging
from yam_realtime.utils.camera_thread import EpisodeSaverThread, LeRobotSaverThread

import tqdm
from yam_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from yam_realtime.agents.teleoperation.oculus_viser_agent import OculusViserAgent
from yam_realtime.envs.robot_env import RobotEnv
from yam_realtime.utils.data_saver import DataSaver
from yam_realtime.utils.lerobot_data_saver import LeRobotDataSaver
from yam_realtime.utils.video_data_saver import VideoDataSaver

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.robots.robot import Robot
import logging
import time
import tyro
from scipy.spatial.transform import Rotation as R

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
    config_path: Tuple[str, ...] = ("~/thomas/YAM/yam_realtime/configs/yam_record_replay.yaml",)
    

def main(args: Args):
    # Monkey-patch CAN bus to suppress cleanup errors (must be done after imports)
    try:
        import can.bus
        import can.notifier
        import can.interfaces.socketcan.socketcan
        import can.exceptions
        import logging as can_logging
        
        # Save reference to exception class
        CanOperationError = can.exceptions.CanOperationError
        
        # Suppress BusABC.__del__ warnings
        _original_bus_del = can.bus.BusABC.__del__
        def _patched_bus_del(self):
            try:
                logger = can_logging.getLogger('can.bus')
                old_level = logger.level
                logger.setLevel(can_logging.ERROR)
                try:
                    _original_bus_del(self)
                finally:
                    logger.setLevel(old_level)
            except (OSError, Exception):
                pass
        can.bus.BusABC.__del__ = _patched_bus_del
        
        # Suppress SocketcanBus._recv_internal errors during shutdown
        _original_recv_internal = can.interfaces.socketcan.socketcan.SocketcanBus._recv_internal
        def _patched_recv_internal(self, timeout=None):
            try:
                return _original_recv_internal(self, timeout)
            except (OSError, CanOperationError):
                return None, False  # Return empty result on socket errors
        can.interfaces.socketcan.socketcan.SocketcanBus._recv_internal = _patched_recv_internal
    except ImportError:
        pass  # CAN not available
    
    # Change to yam_realtime directory for relative paths to work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yam_realtime_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    os.chdir(yam_realtime_dir)
    
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")
    logger.info(f"Working directory: {os.getcwd()}")

    server_processes = []
    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        # setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")
        logger.info(f"camera_dict: {camera_dict}")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=rate,
        )

        # Reset robots (only reset the ones that are configured)
        if 'left' in robots:
            reset_robot(agent, env, 'left')
        if 'right' in robots:
            reset_robot(agent, env, 'right')
        

        logger.info("Starting control loop...")
        
        # Get storage configuration
        storage_cfg = configs_dict['storage']
        task_name = storage_cfg.get('task_name', 'default_task')
        task_directory = task_name.lower().replace(' ', '_').replace('.', '').replace(',', '')
        save_dir = storage_cfg.get('save_dir', './data')
        data_format = storage_cfg.get('format', 'json')
        
        # Convert save_dir to absolute path based on config file location
        from pathlib import Path
        config_dir = Path(main_config.config_path).parent if hasattr(main_config, 'config_path') else Path.cwd()
        save_dir_abs = (config_dir / save_dir).resolve() if not Path(save_dir).is_absolute() else Path(save_dir).resolve()
        
        # Initialize data saver based on format
        if data_format == 'video':
            # Video format - only record camera feeds (no robot data)
            camera_names = list(camera_dict.keys()) if camera_dict else []
            logger.info(f"Using video format - cameras: {camera_names}")
            
            # Video format uses: output/{task_name}/{timestamp}/
            # save_dir is ignored for video format, always use "output" in current directory
            output_dir = Path.cwd() / 'output'
            
            data_saver = VideoDataSaver(
                task_name=task_directory,
                save_dir=str(output_dir),
                fps=int(storage_cfg.get('fps', 30)),
                camera_names=camera_names,
            )
            
            logger.info(f"Video saver initialized: '{task_name}' -> {output_dir / task_directory}")
        elif data_format == 'lerobot':
            camera_names = list(camera_dict.keys()) if camera_dict else []
            repo_id = task_directory  # Just "pick_up_the_cloth"
            
            # Dataset path: data/lerobot/pick_up_the_cloth
            lerobot_root = save_dir_abs / 'lerobot'
            dataset_path = lerobot_root / repo_id
            
            if dataset_path.exists():
                # # Temporarily restore stderr for clean user input
                # sys.stderr.flush()
                # sys.stdout.flush()
                # os.dup2(saved_stderr, 2)
                
                # Clear screen and show prominent message
                print("\n" + "="*80)
                print("WARNING: Dataset directory already exists!")
                print("="*80)
                print(f"Path: {dataset_path}")
                print("\n" + "="*80)
                print("Choose an option:")
                print("="*80)
                print("  [d] Delete existing directory")
                print("  [r] Rename existing directory (add timestamp)")
                print("  [q] Exit program")
                print("="*80)
                
                try:
                    choice = input("\nYour choice [d/r/q]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = 'q'
                
                print("="*80 + "\n")
                
                if choice == 'd':
                    import shutil
                    print(f"Deleting {dataset_path}...")
                    shutil.rmtree(dataset_path)
                    print("✓ Directory deleted\n")
                elif choice == 'r':
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_path = Path(str(dataset_path) + f"_backup_{timestamp}")
                    print(f"Renaming {dataset_path} → {new_path.name}...")
                    dataset_path.rename(new_path)
                    print("✓ Directory renamed\n")
                else:
                    print("Exiting...\n")
                    sys.exit(0)
                
                # # Re-enable stderr filtering after user input
                # sys.stderr.flush()
                # sys.stdout.flush()
                # # Note: We don't redirect stderr back to pipe because it might cause issues
                # # The filter thread is already running and will handle new output
            
            logger.info(f"Creating LeRobot dataset: {repo_id}")
            # Get total episodes to determine batch encoding size
            total_episodes = storage_cfg.get('episodes', 10)
            
            data_saver = LeRobotDataSaver(
                repo_id=repo_id,  # "pick_up_the_cloth" - used for HF repo name
                root=str(dataset_path),  # Full path: "data/lerobot/pick_up_the_cloth"
                fps=int(storage_cfg.get('fps', 10)),
                task_name=task_name,
                robot_type="yam",
                camera_names=camera_names,
                use_videos=True,
                image_writer_processes=4,
                image_writer_threads=len(camera_names) * 2 if camera_names else 4,
                hf_user=storage_cfg.get('hf_user'),
                auto_upload=storage_cfg.get('auto_upload', True),
                # IMPORTANT: Batch encode videos to avoid blocking
                # Videos will be encoded in one batch at the end
                batch_encoding_size=total_episodes,  # Encode all at once at the end
            )
            
            logger.info(f"Dataset initialized: '{task_name}' -> {dataset_path}")
        else:
            logger.info("Using legacy JSON format")
            data_saver = DataSaver(
                save_dir=os.path.join(save_dir, 'json'),
                task_directory=task_directory, 
                language_instruction=task_name
            )
        
        _run_control_loop(env, agent, main_config, configs_dict, data_saver)
        # Normal completion - allow upload
        should_upload = True

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        should_upload = False  # Don't upload incomplete data
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        should_upload = False  # Don't upload on error
        raise e
    finally:
        # Cleanup in correct order
        logger.info("Shutting down...")
        
        # 1. Stop agent first (stop sending commands)
        if "agent" in locals():
            try:
                cleanup_processes(agent, server_processes)
            except Exception as e:
                logger.warning(f"Error cleaning up agent: {e}")
        
        # 2. Close environment (this closes cameras)
        if "env" in locals():
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
        
        # 3. Manually disable motors (auto cleanup doesn't disable right arm)
        logger.info("Manually disabling all motors...")
        try:
            from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
            import time
            import logging as motor_logging
            
            # Temporarily disable CAN interface warnings during shutdown
            can_logger = motor_logging.getLogger('root')
            old_level = can_logger.level
            can_logger.setLevel(motor_logging.ERROR)
            
            try:
                for channel in ['can_left', 'can_right']:
                    can = None
                    try:
                        can = DMSingleMotorCanInterface(channel=channel, bustype='socketcan', bitrate=1000000)
                        for motor_id in [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]:
                            try:
                                can.motor_off(motor_id)
                                time.sleep(0.02)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    finally:
                        if can is not None:
                            try:
                                can.shutdown()
                            except Exception:
                                pass
            finally:
                can_logger.setLevel(old_level)
        except Exception as e:
            logger.warning(f"Error during motor shutdown: {e}")
        
        # 4. Finalize data saver (only upload if normal completion)
        if "data_saver" in locals() and "should_upload" in locals():
            try:
                if hasattr(data_saver, 'finalize'):
                    if should_upload:
                        logger.info("Finalizing dataset and uploading to Hugging Face...")
                        data_saver.finalize()
                    else:
                        logger.info("Skipping upload (interrupted/error), but saving metadata...")
                        # Save metadata without uploading
                        if hasattr(data_saver, '_save_metadata'):
                            data_saver._save_metadata()
            except Exception as e:
                logger.warning(f"Error finalizing data saver: {e}")
        
        logger.info("Shutdown complete.")



# slowly move the robot back to original position
def reset_robot(agent: Agent, env: RobotEnv, side: str):
    agent.act({})
    current_pos = env.robot(side).get_joint_pos()
    target_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    steps = 50
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
        env.robot(side).command_joint_pos(target_pos)
        time.sleep(2 / steps)

def sum_delta_action(prev_delta, curr_delta):
    prev_delta_pos_left = prev_delta["left"]["delta"][:3]
    prev_delta_quat_left = np.concatenate([prev_delta["left"]["delta"][4:], [prev_delta["left"]["delta"][3]]])

    curr_delta_pos_left = curr_delta["left"]["delta"][:3]
    curr_delta_quat_left = np.concatenate([curr_delta["left"]["delta"][4:], [curr_delta["left"]["delta"][3]]])

    sum_delta_pos_left = curr_delta_pos_left + prev_delta_pos_left
    sum_delta_quat_left = (R.from_quat(curr_delta_quat_left) * R.from_quat(prev_delta_quat_left)).as_quat()

    prev_delta_pos_right = prev_delta["right"]["delta"][:3]
    prev_delta_quat_right = np.concatenate([prev_delta["right"]["delta"][4:], [prev_delta["right"]["delta"][3]]])

    curr_delta_pos_right = curr_delta["right"]["delta"][:3]
    curr_delta_quat_right = np.concatenate([curr_delta["right"]["delta"][4:], [curr_delta["right"]["delta"][3]]])

    sum_delta_pos_right = curr_delta_pos_right + prev_delta_pos_right
    sum_delta_quat_right = (R.from_quat(curr_delta_quat_right) * R.from_quat(prev_delta_quat_right)).as_quat()

    delta_sum = {
        "left": {"delta" : np.concatenate([sum_delta_pos_left, [sum_delta_quat_left[3]], sum_delta_quat_left[:3]])},
        "right": {"delta" : np.concatenate([sum_delta_pos_right, [sum_delta_quat_right[3]], sum_delta_quat_right[:3]])}
    }
    return delta_sum

def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig, configs_dict: Dict, data_saver) -> None:
    """
    Run the main control loop.
    Supports legacy JSON format (DataSaver), LeRobot format (LeRobotDataSaver), and Video format (VideoDataSaver).
    """
    # Check data saver type
    is_lerobot = isinstance(data_saver, LeRobotDataSaver)
    is_video = isinstance(data_saver, VideoDataSaver)
    
    # Only use background thread for legacy JSON format
    if not is_lerobot and not is_video:
        # Legacy format - use threaded saver
        previous_action = agent.act({})
        saver_thread = EpisodeSaverThread(data_saver)
        saver_thread.start()

    logger = logging.getLogger(__name__)
    num_traj = 1

    # Init environment and warm up agent
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")
    last_save_time = time.time()
    hz = configs_dict['storage'].get('fps', 10)
    
    if not is_lerobot and not is_video:
        delta_cumulative = {
            "left": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
            "right": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
        }

    # Main control loop
    while num_traj <= configs_dict['storage']['episodes']:
        obs = env.reset()
        
        # Reset agent state between episodes by calling act with empty obs
        # This resets IK position and clears prev_joints to prevent cumulative drift
        agent.act({})
        
        if not is_lerobot and not is_video:
            data_saver.reset_buffer()
        

        logger.info(f"Press 'A' to start collecting data: ")

        while True:
            info = agent.get_info()
            if info["success"]:
                logger.info(f"Successfully pressed 'A', starting to collect data")
                time.sleep(2)
                break
        logger.info(f"Press 'A' to save the data, press 'B' to discard the data")

        for _ in tqdm.tqdm(range(configs_dict['collection']['max_episode_length']), desc=f"Collecting data {num_traj}/{configs_dict['storage']['episodes']}"):
            info = agent.get_info()
            
            # Check for manual motor disable (press 'X' button)
            if info.get("manual_disable", False):
                logger.warning("Manual motor disable triggered!")
                # Disable all motors (only for robots that exist)
                try:
                    if 'left' in env._robot_dict:
                        env.robot('left').motor_off()
                    if 'right' in env._robot_dict:
                        env.robot('right').motor_off()
                    logger.info("All motors disabled")
                except Exception as e:
                    logger.error(f"Error disabling motors: {e}")
                break
            
            while (not info["success"] and not info["failure"]) and not info.get("movement_enabled", {}).get('left', False) and not info.get("movement_enabled", {}).get('right', False):
                info = agent.get_info()
            
            save = False
            if info["success"]:
                save = True
                break
            elif info["failure"]:
                break

            # within agent, we use 7 point representation for action (x, y, z, w, x, y, z) for both left and right arm
            # but for robot to actually move, we use 6 value joints position instead. So the act function in agent converts 
            # the 7 points action. Also, there are some nasty conversion for quat from (w, x, y, z) to (x, y, z, w),
            # the robot and viser uses (w, x, y, z) for quat. but python scipy uses (x, y, z, w) for quat.

            # currently in the json for traning, the delta acton is actually a 7 point action without gripper. (from calc_delta_action)
            # Also make sure that gripper is either 0 or 1. 


            # note that we didn't really use obs (this is the outside world joints state containing pos, gripper, eff, vel)
            # we use viser agent to compute the action in 3D viser space, which is then solved by ik to get the joints position.
            # The action that the env.step() uses is the viser joints (the action returned by agent.act()).
            if info.get("movement_enabled", {}).get('left', False) or info.get("movement_enabled", {}).get('right', False):
                try:
                    act = agent.act(obs)
                except Exception as e:
                    logger.error(f"Error in agent.act: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Build action dict only for available robots
                action = {}
                if 'left' in env._robot_dict and 'pos' in act.get('left', {}):
                    action['left'] = {'pos': act['left']['pos']}
                if 'right' in env._robot_dict and 'pos' in act.get('right', {}):
                    action['right'] = {'pos': act['right']['pos']}
                
                # Save data at specified frequency
                if time.time() - last_save_time > (1/hz):
                    if is_lerobot:
                        # LeRobot format
                        data_saver.add_frame(
                            observation=obs,
                            action=act,
                            timestamp=time.time()
                        )
                    elif is_video:
                        # Video format - only save camera observations
                        data_saver.add_frame(observation=obs)
                    else:
                        # Legacy JSON format
                        delta_cumulative = sum_delta_action(delta_cumulative, act)
                        act["left"]["delta"] = delta_cumulative["left"]["delta"]
                        act["right"]["delta"] = delta_cumulative["right"]["delta"]
                        data_saver.add_observation(obs, act)
                        delta_cumulative = {
                            "left": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
                            "right": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
                        }
                    last_save_time = time.time()
                
                try:
                    next_obs = env.step(action)
                except Exception as e:
                    logger.error(f"Error in env.step: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                # Even when not moving, step with empty action to update camera observations
                # This prevents duplicate frames in the video at the end
                try:
                    next_obs = env.step({})
                except Exception as e:
                    logger.error(f"Error in env.step (empty): {e}")
                    import traceback
                    traceback.print_exc()
                    break
            obs = next_obs.copy()
        
        if save:
            if is_lerobot:
                # LeRobot format - save directly (video encoding happens in background automatically)
                if data_saver.episode_started:
                    logger.info(f"Saving LeRobot episode {num_traj}...")
                    data_saver.save_episode()
                    num_traj += 1
                    logger.info(f"Episode saved! (Video encoding continues in background)")
                else:
                    logger.info(f"No data collected, skipping save")
            elif is_video:
                # Video format - save directly
                if data_saver.episode_started:
                    logger.info(f"Saving video episode {num_traj}...")
                    data_saver.save_episode()
                    num_traj += 1
                    logger.info(f"Video episode saved!")
                else:
                    logger.info(f"No data collected, skipping save")
            else:
                # Legacy JSON format - use thread
                if data_saver.buffer == []:
                    logger.info(f"No data collected, skipping save")
                    continue
                saver_thread.save_episode(data_saver.buffer.copy())
                num_traj += 1
                logger.info(f"Successfully collected data")
        else:
            logger.info(f"Failure")
        
        # Reset robots (only reset the ones that are configured)
        robot_dict = configs_dict.get('robots', {})
        if 'left' in robot_dict:
            reset_robot(agent, env, 'left')
        if 'right' in robot_dict:
            reset_robot(agent, env, 'right')

    # Stop saver thread (only for legacy format)
    if not is_lerobot and not is_video:
        logger.info("Stopping background saver thread...")
        saver_thread.stop()
        logger.info("Waiting for all pending saves to complete...")
        saver_thread.join(timeout=300)
        if saver_thread.is_alive():
            logger.warning("Background saver thread did not finish in time!")
        else:
            logger.info("All pending saves completed successfully")

    env.reset()
    logger.info(f"Finished collecting data")

if __name__ == "__main__":
    main(tyro.cli(Args))
