import logging
import os
from typing import Optional, Dict, Any
import numpy as np

class LogColors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def setup_logger(name: str, level: int = logging.INFO, log_to_file: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and optional file logging.
    
    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to also log to file
        log_file: Path to log file (if None, uses default naming)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        if log_file is None:
            log_file = f"logs/{name}.log"
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger

# Global loggers for different components
def get_collect_demos_logger() -> logging.Logger:
    """Get logger for data collection."""
    return setup_logger("collect_demos")

def get_replay_logger() -> logging.Logger:
    """Get logger for replay functionality."""
    return setup_logger("replay")

def get_data_utils_logger() -> logging.Logger:
    """Get logger for data utilities."""
    return setup_logger("data_utils")

def get_data_saver_logger() -> logging.Logger:
    """Get logger for data saver."""
    return setup_logger("data_saver")

def get_robot_env_logger() -> logging.Logger:
    """Get logger for robot environment."""
    return setup_logger("robot_env")

def get_franka_logger() -> logging.Logger:
    """Get logger for Franka robot."""
    return setup_logger("franka")

def get_camera_logger() -> logging.Logger:
    """Get logger for camera operations."""
    return setup_logger("camera")

def get_policy_eval_logger() -> logging.Logger:
    """Get logger for policy evaluation."""
    return setup_logger("policy_eval")

def get_molmoact_logger() -> logging.Logger:
    """Get logger for MolmoAct policy."""
    return setup_logger("molmoact")

# Generic logging functions that work with any logger
def log_info(logger: logging.Logger, msg: str):
    """Log info message with cyan color."""
    logger.info(f"{LogColors.CYAN}{msg}{LogColors.END}")

def log_warning(logger: logging.Logger, msg: str):
    """Log warning message with yellow color."""
    logger.warning(f"{LogColors.YELLOW}{msg}{LogColors.END}")

def log_success(logger: logging.Logger, msg: str):
    """Log success message with green color."""
    logger.info(f"{LogColors.GREEN}{msg}{LogColors.END}")

def log_error(logger: logging.Logger, msg: str):
    """Log error message with red color."""
    logger.error(f"{LogColors.RED}{msg}{LogColors.END}")

def log_data_info(logger: logging.Logger, msg: str):
    """Log data information with blue color."""
    logger.info(f"{LogColors.BLUE}{msg}{LogColors.END}")

def log_config(logger: logging.Logger, msg: str):
    """Log configuration message with cyan color."""
    logger.info(f"{LogColors.CYAN}{msg}{LogColors.END}")

def log_connect(logger: logging.Logger, msg: str):
    """Log connection message with blue color."""
    logger.info(f"{LogColors.BLUE}{msg}{LogColors.END}")

def log_instruction(logger: logging.Logger, msg: str):
    """Log instruction message with yellow color."""
    logger.info(f"{LogColors.YELLOW}{msg}{LogColors.END}")

def log_failure(logger: logging.Logger, msg: str):
    """Log failure message with red color."""
    logger.info(f"{LogColors.RED}{msg}{LogColors.END}")

def log_important(logger: logging.Logger, msg: str):
    """Log important message with bold header color."""
    logger.info(f"{LogColors.BOLD}{LogColors.HEADER}{msg}{LogColors.END}")

# Convenience functions for specific loggers
def log_collect_demos(msg: str, level: str = "info"):
    """Log message for data collection."""
    logger = get_collect_demos_logger()
    if level == "info":
        log_info(logger, msg)
    elif level == "warning":
        log_warning(logger, msg)
    elif level == "success":
        log_success(logger, msg)
    elif level == "error":
        log_error(logger, msg)
    elif level == "config":
        log_config(logger, msg)
    elif level == "connect":
        log_connect(logger, msg)
    elif level == "instruction":
        log_instruction(logger, msg)
    elif level == "failure":
        log_failure(logger, msg)
    elif level == "important":
        log_important(logger, msg)

def log_replay(msg: str, level: str = "info"):
    """Log message for replay functionality."""
    logger = get_replay_logger()
    if level == "info":
        log_info(logger, msg)
    elif level == "warning":
        log_warning(logger, msg)
    elif level == "success":
        log_success(logger, msg)
    elif level == "error":
        log_error(logger, msg)
    elif level == "data_info":
        log_data_info(logger, msg)

def log_data_utils(msg: str, level: str = "info"):
    """Log message for data utilities."""
    logger = get_data_utils_logger()
    if level == "info":
        log_info(logger, msg)
    elif level == "warning":
        log_warning(logger, msg)
    elif level == "success":
        log_success(logger, msg)
    elif level == "error":
        log_error(logger, msg)
    elif level == "data_info":
        log_data_info(logger, msg)

def log_robot_env(msg: str, level: str = "info"):
    """Log message for robot environment."""
    logger = get_robot_env_logger()
    if level == "info":
        log_info(logger, msg)
    elif level == "warning":
        log_warning(logger, msg)
    elif level == "success":
        log_success(logger, msg)
    elif level == "error":
        log_error(logger, msg)

def log_policy_eval(msg: str, level: str = "info"):
    """Log message for policy evaluation."""
    logger = get_policy_eval_logger()
    if level == "info":
        log_info(logger, msg)
    elif level == "warning":
        log_warning(logger, msg)
    elif level == "success":
        log_success(logger, msg)
    elif level == "error":
        log_error(logger, msg)
    elif level == "config":
        log_config(logger, msg)
    elif level == "connect":
        log_connect(logger, msg)
    elif level == "instruction":
        log_instruction(logger, msg)
    elif level == "failure":
        log_failure(logger, msg)
    elif level == "important":
        log_important(logger, msg)
    elif level == "data_info":
        log_data_info(logger, msg)

def log_molmoact(msg: str, level: str = "info"):
    """Log message for MolmoAct policy."""
    logger = get_molmoact_logger()
    if level == "info":
        log_info(logger, msg)
    elif level == "warning":
        log_warning(logger, msg)
    elif level == "success":
        log_success(logger, msg)
    elif level == "error":
        log_error(logger, msg)
    elif level == "config":
        log_config(logger, msg)
    elif level == "data_info":
        log_data_info(logger, msg)

def log_demo_data_info(demo: Dict[str, Any], demo_dir: str) -> None:
    """
    Log comprehensive information about the demo data.
    
    Args:
        demo: The loaded demo dictionary
        demo_dir: Path to the demo file
    """
    log_data_utils("=" * 60, "data_info")
    log_data_utils(f"DEMO DATA ANALYSIS: {os.path.basename(demo_dir)}", "data_info")
    log_data_utils("=" * 60, "data_info")
    
    # Basic file info
    file_size = os.path.getsize(demo_dir) / (1024 * 1024)  # MB
    log_data_utils(f"File size: {file_size:.2f} MB", "data_info")
    
    # Available keys
    log_data_utils(f"Available keys: {list(demo.keys())}", "data_info")
    
    # Episode length
    if "action" in demo:
        episode_length = len(demo["action"])
        log_data_utils(f"Episode length: {episode_length} steps", "data_info")
        
        # Action statistics
        actions = demo["action"]
        if isinstance(actions, np.ndarray):
            log_data_utils(f"Action shape: {actions.shape}", "data_info")
            log_data_utils(f"Action dtype: {actions.dtype}", "data_info")
            log_data_utils(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]", "data_info")
            log_data_utils(f"Action mean: {actions.mean():.4f}", "data_info")
            log_data_utils(f"Action std: {actions.std():.4f}", "data_info")
    
    # Low-dimensional state info
    if "lowdim_ee" in demo:
        lowdim_ee = demo["lowdim_ee"]
        log_data_utils(f"End-effector state shape: {lowdim_ee.shape}", "data_info")
        log_data_utils(f"End-effector state range: [{lowdim_ee.min():.4f}, {lowdim_ee.max():.4f}]", "data_info")
    
    if "lowdim_qpos" in demo:
        lowdim_qpos = demo["lowdim_qpos"]
        log_data_utils(f"Joint positions shape: {lowdim_qpos.shape}", "data_info")
        log_data_utils(f"Joint positions range: [{lowdim_qpos.min():.4f}, {lowdim_qpos.max():.4f}]", "data_info")
    
    # Image data info
    image_keys = [key for key in demo.keys() if any(img_type in key for img_type in ['rgb', 'depth'])]
    if image_keys:
        log_data_utils(f"Image keys found: {image_keys}", "data_info")
        for key in image_keys:
            img_data = demo[key]
            if isinstance(img_data, np.ndarray):
                log_data_utils(f"  {key}: shape={img_data.shape}, dtype={img_data.dtype}", "data_info")
                if 'rgb' in key:
                    log_data_utils(f"  {key}: value range=[{img_data.min()}, {img_data.max()}]", "data_info")
                elif 'depth' in key:
                    log_data_utils(f"  {key}: value range=[{img_data.min()}, {img_data.max()}]", "data_info")
    
    # Language instruction
    if "language_instruction" in demo:
        log_data_utils(f"Language instruction: {demo['language_instruction'][0]}", "data_info")
    
    # Teleop device
    if "teleop_device" in demo:
        log_data_utils(f"Teleop device: {demo['teleop_device'][0]}", "data_info")
    