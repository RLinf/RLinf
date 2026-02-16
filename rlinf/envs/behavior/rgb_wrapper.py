"""
RGB Wrapper for BEHAVIOR environments to use higher resolution cameras.

This is a minimal adaptation of OpenPI-Comet's RGBWrapper for VectorEnvironment.
"""

from omnigibson.envs import VectorEnvironment
from omnigibson.learning.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    ROBOT_CAMERA_NAMES,
)
from rlinf.utils.logging import get_logger

__all__ = ["apply_rgb_wrapper"]


def apply_rgb_wrapper(env: VectorEnvironment, use_high_res: bool = True):
    """
    Apply RGB wrapper to modify camera resolutions.
    
    Based on openpi-comet/src/behavior/learning/wrappers/rgb_wrapper.py
    Adapted for VectorEnvironment (handles multiple sub-environments).
    
    Args:
        env: VectorEnvironment instance
        use_high_res: If True, use 720x720 (head) and 480x480 (wrist)
    """
    logger = get_logger()
    
    if not use_high_res:
        logger.info("RGB wrapper disabled, using default camera resolutions")
        return env
    
    logger.info("Applying RGB wrapper with high-resolution cameras...")
    
    # For VectorEnvironment, modify each sub-environment
    # This is the ONLY difference from openpi-comet's version
    for sub_env in env.envs:
        robot = sub_env.robots[0]
        
        # Update robot sensors (same logic as openpi-comet, with one safety check)
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]  # Same as openpi-comet line 25
            
            # Handle dynamic robot instance names (e.g., robot_lfwumz vs robot_r1)
            if sensor_name not in robot.sensors:
                # Robot instance name may vary, find the matching sensor
                # An example of robot.sensors.keys() is
                # ['robot_ycoexm:left_realsense_link:Camera:0', 
                # 'robot_ycoexm:right_realsense_link:Camera:0', 
                # 'robot_ycoexm:zed_link:Camera:0']
                matching_sensors = [s for s in robot.sensors.keys() if s.endswith(sensor_name.split(':', 1)[1])]
                if matching_sensors:
                    sensor_name = matching_sensors[0]
                else:
                    logger.warning(f"Sensor not found: {sensor_name}. Available: {list(robot.sensors.keys())}")
                    continue
            
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
                logger.info(f"Set {sensor_name} to {HEAD_RESOLUTION}")
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
                logger.info(f"Set {sensor_name} to {WRIST_RESOLUTION}")
        
        # Reload observation space (same as openpi-comet line 38)
        sub_env.load_observation_space()
    
    logger.info("RGB wrapper applied successfully!")
    return env
