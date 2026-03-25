"""Dual-arm robot state snapshot for DOSW1."""

import time
from dataclasses import dataclass, field

import numpy as np

_NUM_JOINTS = 6


@dataclass
class DOSW1RobotState:
    """Snapshot of the dual-arm DOSW1 robot at a single timestep.

    Each arm has 6 revolute joints (radians) and 1 gripper value in metres.
    Control and state are joint-space only.
    """

    left_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(_NUM_JOINTS)
    )
    left_gripper: float = 0.0
    right_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(_NUM_JOINTS)
    )
    right_gripper: float = 0.0
    timestamp: float = field(default_factory=time.time)
