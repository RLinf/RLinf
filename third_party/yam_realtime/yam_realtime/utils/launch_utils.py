"""
Utilities for launching and configuring robots, sensors, and agents.
"""

import logging
import subprocess
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import omegaconf
import portal

from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.robot import ROBOT_PROTOCOL_METHODS, Robot
from yam_realtime.utils.portal_utils import (
    Client,
    RemoteServer,
    launch_remote_get_local_handler,
)

# Create logger for this module
logger = logging.getLogger(__name__)


def setup_logging() -> logging.Logger:
    """
    Setup logging configuration.

    Returns:
        Logger instance for the main module
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
        force=True,  # Force reconfiguration of logging
    )

    # Create and return a logger for the main module
    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(logging.INFO)

    return main_logger


def setup_can_interfaces():
    """Setup CAN interfaces for robot communication."""
    logger.info("Setting up CAN interfaces...")
    subprocess.run(["bash", "yam_realtime/scripts/reset_all_can.sh"], check=True)
    time.sleep(0.5)
    logger.info("CAN interfaces ready")


def initialize_sensors(
    sensors_cfg: Optional[Dict[str, Any]], server_processes: List[Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Initialize sensors (cameras) from configuration.

    Args:
        sensors_cfg: Sensor configuration dictionary
        server_processes: List to track server processes

    Returns:
        Tuple of (camera_dict, camera_info)
    """
    camera_dict = {}
    camera_info = {}

    if sensors_cfg is None:
        logger.info("No sensors configured")
        return camera_dict, camera_info

    _launch_remote_get_local_handler = partial(
        launch_remote_get_local_handler,
        launch_remote=True,
        process_pool=server_processes,
    )

    for sensor_name, sensor_cfg in sensors_cfg.items():
        if sensor_name == "cameras" and sensor_cfg is not None:
            for camera_name, camera_config in sensor_cfg.items():
                logger.info(f"Initializing camera: {camera_name}")
                camera_config["camera"]["name"] = camera_name
                _, client = _launch_remote_get_local_handler(camera_config)
                camera_dict[camera_name] = client

                if (
                    hasattr(client, "supported_remote_methods")
                    and "get_camera_info" in client.supported_remote_methods  # type: ignore
                ):  # type: ignore
                    camera_info[camera_name] = client.get_camera_info()
                else:
                    raise AttributeError(f"Camera {camera_name} does not implement 'get_camera_info'!")

    logger.info(f"Initialized {len(camera_dict)} cameras")
    return camera_dict, camera_info


def initialize_robots(robots_cfg: Dict[str, Union[str, Robot]], server_processes: List[Any]) -> Dict[str, Robot]:
    """
    Initialize robots from configuration.

    Args:
        robots_cfg: Robot configuration dictionary
        server_processes: List to track server processes

    Returns:
        Dictionary of initialized robots
    """
    robots = {}

    for robot_name, robot_path_or_robot in robots_cfg.items():
        logger.info(f"Initializing robot: {robot_name}")
        robot_client = _create_robot_client(robot_path_or_robot, server_processes)
        robots[robot_name] = robot_client

    logger.info(f"Initialized {len(robots)} robots")
    return robots


def _create_robot_client(robot_path_or_robot: Union[str, Robot, List[str]], server_processes: List[Any]) -> Robot:
    """
    Create a robot client from configuration or robot instance.

    Args:
        robot_path_or_robot: Robot configuration path(s) or robot instance
        server_processes: List to track server processes

    Returns:
        Robot client instance
    """
    if isinstance(robot_path_or_robot, (str, omegaconf.listconfig.ListConfig, list)):
        # Handle configuration file path(s)
        if isinstance(robot_path_or_robot, omegaconf.listconfig.ListConfig):
            robot_path_or_robot = list(robot_path_or_robot)

        try:
            robot_dict = DictLoader.load(robot_path_or_robot)
        except Exception as e:
            logger.error(f"Failed to load robot config: {robot_path_or_robot}")
            raise

        if "Client" in robot_dict["_target_"]:
            return instantiate(robot_dict)
        else:
            _, robot_client = launch_remote_get_local_handler(
                robot_dict,
                process_pool=server_processes,
                custom_remote_methods=ROBOT_PROTOCOL_METHODS,
            )
            return robot_client  # type: ignore

    elif isinstance(robot_path_or_robot, Robot):
        # Handle robot instance - create remote server
        port = portal.free_port()

        def _launch_robot_server(robot: Any, port: int) -> None:
            remote_server = RemoteServer(robot, port, custom_remote_methods=ROBOT_PROTOCOL_METHODS)
            remote_server.serve()

        process = portal.Process(partial(_launch_robot_server, robot=robot_path_or_robot, port=port), start=True)
        server_processes.append(process)
        return Client(port)  # type: ignore

    else:
        raise ValueError(f"Invalid robot configuration: {robot_path_or_robot}")


def initialize_agent(agent_cfg: Dict[str, Any], server_processes: List[Any]) -> Any:
    """
    Initialize agent from configuration.

    Args:
        agent_cfg: Agent configuration dictionary
        server_processes: List to track server processes

    Returns:
        Agent instance or client
    """
    logger.info("Initializing agent...")

    if "Client" in agent_cfg["_target_"]:
        agent = instantiate(agent_cfg)
    else:
        # Define the agent methods that need to be remotely accessible
        agent_remote_methods = {
            "act": False,
            "reset": False,
            "close": False,
        }
        _, agent = launch_remote_get_local_handler(agent_cfg, custom_remote_methods=agent_remote_methods)
        server_processes.append(_)  # Track the server process

    logger.info("Agent initialized")
    return agent


def cleanup_processes(agent: Any, server_processes: List[Any]) -> None:
    """
    Clean up all processes and connections.

    Args:
        agent: Agent instance to close
        server_processes: List of server processes to terminate
    """
    logger.info("Cleaning up processes...")

    try:
        agent.close()
    except Exception as e:
        logger.warning(f"Error closing agent: {e}")

    # Terminate server processes and wait for them to finish
    import time
    for server_process in server_processes:
        try:
            if server_process and server_process.is_alive():
                server_process.terminate()  # Send SIGTERM first
                server_process.join(timeout=2)  # Wait up to 2 seconds
                
                # If still alive, force kill
                if server_process.is_alive():
                    server_process.kill()
                    server_process.join(timeout=1)  # Wait for kill to complete
        except Exception as e:
            logger.warning(f"Error terminating server process: {e}")

    logger.info("Cleanup complete")
