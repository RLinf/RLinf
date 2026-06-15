# Copyright 2026 The RLinf Authors.
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

"""SO101 6-DOF robot environment powered by LeRobot.

This environment wraps LeRobot's :class:`~lerobot.robots.so_follower.SO101Follower`
to provide a standard ``gym.Env`` interface for RLinf training, data collection,
and evaluation.

Key features:
- **No distributed controller needed** — LeRobot's synchronous API is used directly.
- **Joint-space control** — actions are absolute joint positions in degrees.
- **LeRobot calibration** — uses the same calibration JSON as the LeRobot CLI.
- **Leader teleop** — optional leader arm for bilateral teleoperation data collection.
- **Dummy mode** — for offline training and testing without hardware.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .so101_robot_state import SO101RobotState

# Default joint limits for SO101 in degrees (Feetech STS3215 range).
# These are conservative defaults; actual limits are set during calibration.
_DEFAULT_JOINT_LIMIT_LOW_DEG = np.array([-150.0, -90.0, -150.0, -90.0, -150.0, 0.0])
_DEFAULT_JOINT_LIMIT_HIGH_DEG = np.array([150.0, 90.0, 150.0, 90.0, 150.0, 90.0])

# LeRobot motor names for SO101 in bus-ID order.
_SO101_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class SO101RobotConfig:
    """Configuration for :class:`SO101Env`.

    Hardware connection fields (``port``, ``leader_port``, ``camera_serials``,
    etc.) are populated automatically from :class:`~rlinf.scheduler.SO101HWInfo`
    when ``None``.
    """

    # ── Hardware connection ────────────────────────────────────────────────
    port: Optional[str] = None
    """Serial port for the Feetech motor bus (e.g. ``"/dev/ttyACM0"``).
    Auto-filled from :class:`SO101HWInfo`."""

    leader_port: Optional[str] = None
    """Serial port for the leader arm (teleoperation).
    Only needed for data collection with bilateral teleop."""

    calibration_id: str = "default"
    """LeRobot calibration ID. Calibration JSON is stored at
    ``~/.cache/lerobot/calibration/robots/so_follower/{id}.json``."""

    arm_variant: str = "so101"
    """Arm variant: ``"so101"`` or ``"so100"``."""

    # ── Camera ─────────────────────────────────────────────────────────────
    camera_serials: Optional[list] = None
    """Camera serial numbers or indices. ``None`` or ``[]`` = no cameras."""

    camera_type: Optional[str] = None
    """Camera backend: ``"opencv"``, ``"realsense"``, or ``"zed"``."""

    # ── Control ────────────────────────────────────────────────────────────
    use_degrees: bool = True
    """Whether joint angles use degrees (LeRobot default) or radians."""

    max_relative_target: Optional[float] = None
    """Maximum per-step joint movement in degrees for safety.
    ``None`` disables relative clamping (use with caution)."""

    step_frequency: float = 30.0
    """Maximum environment steps per second (SO101 default is 30 Hz)."""

    # ── Joint limits ───────────────────────────────────────────────────────
    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW_DEG.copy()
    )
    """Lower joint limits ``(6,)`` in degrees."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH_DEG.copy()
    )
    """Upper joint limits ``(6,)`` in degrees."""

    # ── Task ───────────────────────────────────────────────────────────────
    max_num_steps: int = 200
    """Episode truncation horizon."""

    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0, -45.0, 90.0, 0.0, 0.0, 45.0]
    )
    """Joint configuration (degrees) to move to on reset.
    Default is a neutral pose."""

    target_joint_qpos: Optional[np.ndarray] = None
    """Target joint configuration for reward computation.
    When ``None``, a sparse 0/1 reward based on task completion is used."""

    reward_threshold_deg: float = 5.0
    """Per-joint tolerance in degrees for success."""

    # ── Gripper ────────────────────────────────────────────────────────────
    binary_gripper_threshold: float = 45.0
    """Gripper position threshold (degrees) for open/close decision.
    Positions above this are considered 'open'."""

    # ── Modes ──────────────────────────────────────────────────────────────
    is_dummy: bool = False
    """When ``True``, skip all hardware calls (useful for offline training)."""

    enable_teleop: bool = False
    """When ``True``, use leader arm for teleoperation instead of policy."""

    enable_video_player: bool = True
    """Display a live camera window during episodes."""

    save_video_path: Optional[str] = None
    """Path to save episode videos. ``None`` disables saving."""

    def __post_init__(self):
        if self.joint_limit_low is not None:
            self.joint_limit_low = np.array(self.joint_limit_low, dtype=np.float64)
        if self.joint_limit_high is not None:
            self.joint_limit_high = np.array(self.joint_limit_high, dtype=np.float64)
        if self.target_joint_qpos is not None:
            self.target_joint_qpos = np.array(self.target_joint_qpos, dtype=np.float64)


class SO101Env(gym.Env):
    """SO101 6-DOF robot environment with joint-space actions.

    This environment uses LeRobot's :class:`~lerobot.robots.so_follower.SO101Follower`
    for hardware control. No distributed controller Worker is needed — LeRobot's
    synchronous Python API handles all motor communication on the Feetech bus.

    **Action space**: ``Box((7,))`` — ``[q1, ..., q6, gripper]`` in degrees.
    The first six dimensions are absolute joint position targets, clamped to
    the configured joint limits. The seventh dimension is the gripper position.

    **Observation space**: ``Dict{state: Dict{joint_position, gripper_position},
    frames: Dict{...}}``. Camera frames are only present when cameras are
    configured.

    The six joints correspond to LeRobot motor names: ``shoulder_pan``,
    ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``, ``wrist_roll``,
    ``gripper``.

    Example usage::

        from rlinf.envs.realworld.so101 import SO101Env, SO101RobotConfig

        config = SO101RobotConfig(
            port="/dev/ttyACM0",
            is_dummy=False,
        )
        env = SO101Env(config, worker_info=None, hardware_info=None, env_idx=0)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        config: SO101RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info,
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._state = SO101RobotState()
        self._num_steps = 0
        self._robot = None
        self._leader = None
        self._cameras = {}

        if not self.config.is_dummy:
            self._setup_hardware()

        if self.config.camera_serials is None:
            self.config.camera_serials = []

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Connect to the robot hardware.
        self._connect_robot()

    # ── Hardware setup ─────────────────────────────────────────────────────

    def _setup_hardware(self):
        """Fill connection fields from hardware_info when not set by the user."""
        from rlinf.scheduler.hardware.robots.so101 import SO101HWInfo

        if isinstance(self.hardware_info, SO101HWInfo):
            hw_config = self.hardware_info.config
            if self.config.port is None:
                self.config.port = hw_config.port
            if self.config.leader_port is None:
                self.config.leader_port = getattr(hw_config, "leader_port", None)
            if self.config.camera_serials is None:
                self.config.camera_serials = getattr(hw_config, "camera_serials", [])
            if self.config.camera_type is None:
                self.config.camera_type = getattr(hw_config, "camera_type", "opencv")
            if self.config.calibration_id is None:
                self.config.calibration_id = getattr(
                    hw_config, "calibration_id", "default"
                )

    def _connect_robot(self):
        """Connect to the SO101 follower arm via LeRobot."""
        # Lazy-import lerobot to avoid issues on GPU-only nodes.
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        robot_cfg = SO101FollowerConfig(
            port=self.config.port,
            id=self.config.calibration_id,
            use_degrees=self.config.use_degrees,
            max_relative_target=self.config.max_relative_target,
            disable_torque_on_disconnect=True,
        )

        self._robot = SO101Follower(robot_cfg)
        self._robot.connect(calibrate=True)

        if self.config.enable_teleop and self.config.leader_port:
            self._connect_leader()

        self._state.is_connected = True
        self._logger.info(
            f"SO101 robot connected on port {self.config.port} "
            f"(arm_variant={self.config.arm_variant})"
        )

    def _connect_leader(self):
        """Connect to the SO101 leader arm for teleoperation."""
        from lerobot.teleoperators.so_leader import SO101Leader
        from lerobot.teleoperators.so_leader.config_so100_leader import (
            SO101LeaderConfig,
        )

        leader_cfg = SO101LeaderConfig(
            port=self.config.leader_port,
            id=f"{self.config.calibration_id}_leader",
            use_degrees=self.config.use_degrees,
        )
        self._leader = SO101Leader(leader_cfg)
        self._leader.connect()
        self._logger.info(f"SO101 leader connected on port {self.config.leader_port}")

    # ── Action / observation spaces ────────────────────────────────────────

    def _init_action_obs_spaces(self):
        """Initialise action and observation spaces."""
        self._joint_limit_low = np.array(self.config.joint_limit_low, dtype=np.float64)
        self._joint_limit_high = np.array(
            self.config.joint_limit_high, dtype=np.float64
        )

        # Action: [q1, q2, q3, q4, q5, q6, gripper] in degrees.
        action_low = np.append(self._joint_limit_low, 0.0).astype(np.float32)
        action_high = np.append(self._joint_limit_high, 90.0).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        # Observation: joint positions + optional camera frames.
        num_cameras = len(self.config.camera_serials or [])
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,), dtype=np.float32
                        ),
                        "gripper_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"camera_{k}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(num_cameras)
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # ── Core gym API ───────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Args:
            action: ``(7,)`` float array.
                ``action[:6]`` are absolute joint positions in degrees,
                clamped to the configured joint limits.
                ``action[6]`` is the gripper position in degrees.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            q_target = np.clip(
                action[:6], self._joint_limit_low, self._joint_limit_high
            )
            gripper_target = float(np.clip(action[6], 0.0, 90.0))

            # Build LeRobot-format action dict.
            robot_action = {}
            for i, name in enumerate(_SO101_MOTOR_NAMES):
                if name == "gripper":
                    robot_action[name] = gripper_target
                else:
                    robot_action[name] = float(q_target[i])

            # If in teleop mode, override with leader positions.
            if self.config.enable_teleop and self._leader is not None:
                try:
                    leader_action = self._leader.get_action()
                    robot_action = leader_action
                except Exception as e:
                    self._logger.warning(f"Failed to read leader action: {e}")

            self._robot.send_action(robot_action)

        self._num_steps += 1

        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = reward >= 1.0
        truncated = self._num_steps >= self.config.max_num_steps

        # Maintain step frequency.
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to the rest pose.

        Moves the arm to :attr:`~SO101RobotConfig.reset_joint_qpos` and
        resets the step counter.
        """
        if self.config.is_dummy:
            self._num_steps = 0
            return self._get_observation(), {}

        self._num_steps = 0
        self._go_to_rest()
        self._update_state()
        return self._get_observation(), {}

    def _go_to_rest(self):
        """Move the arm to the configured rest joint configuration."""
        rest_action = {}
        for i, name in enumerate(_SO101_MOTOR_NAMES):
            rest_action[name] = float(self.config.reset_joint_qpos[i])
        self._robot.send_action(rest_action)
        time.sleep(1.0)  # Allow time for the move to complete.

    # ── Observation ────────────────────────────────────────────────────────

    def _update_state(self):
        """Read the latest robot state from hardware."""
        if self.config.is_dummy:
            return
        try:
            obs = self._robot.get_observation()
            # Map LeRobot observation keys to SO101RobotState.
            self._state.joint_position = np.array(
                [obs.get(f"{name}.pos", 0.0) for name in _SO101_MOTOR_NAMES[:6]],
                dtype=np.float64,
            )
            # Velocity may not be available on all motor models; default to 0.
            self._state.joint_velocity = np.array(
                [
                    obs.get(f"{name}.vel", obs.get(f"{name}.velocity", 0.0))
                    for name in _SO101_MOTOR_NAMES[:6]
                ],
                dtype=np.float64,
            )
            gripper_pos = obs.get("gripper.pos", obs.get("gripper.position", 0.0))
            self._state.gripper_position = float(gripper_pos)
            self._state.gripper_open = (
                self._state.gripper_position > self.config.binary_gripper_threshold
            )
        except Exception as e:
            self._logger.warning(f"Failed to read robot observation: {e}")

    def _get_observation(self) -> dict:
        """Return the current observation dict."""
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        self._update_state()

        state = {
            "joint_position": self._state.joint_position.astype(np.float32),
            "gripper_position": np.array(
                [self._state.gripper_position], dtype=np.float32
            ),
        }

        frames = self._get_camera_frames()

        return copy.deepcopy({"state": state, "frames": frames})

    # ── Cameras ────────────────────────────────────────────────────────────

    def _get_camera_frames(self) -> dict:
        """Read camera frames (placeholder — extend for your camera setup)."""
        if not self._cameras:
            return {}
        frames = {}
        for name, camera in self._cameras.items():
            try:
                frame = camera.get_frame()
                frames[name] = frame
            except Exception as e:
                self._logger.warning(f"Failed to read camera '{name}': {e}")
        return frames

    # ── Reward ─────────────────────────────────────────────────────────────

    def _calc_step_reward(self, observation: dict) -> float:
        """Compute reward based on distance to target joint configuration.

        When :attr:`~SO101RobotConfig.target_joint_qpos` is set, computes a
        dense reward from the L2 joint error. Otherwise, returns a sparse
        0 reward (task-specific subclasses should override this).
        """
        if self.config.is_dummy:
            return 0.0

        if self.config.target_joint_qpos is not None:
            joint_pos = observation["state"]["joint_position"]
            error = np.linalg.norm(joint_pos - self.config.target_joint_qpos)
            if error < self.config.reward_threshold_deg:
                return 1.0
            # Dense reward: exponential falloff with distance.
            return float(np.exp(-error / self.config.reward_threshold_deg))
        return 0.0

    # ── Cleanup ────────────────────────────────────────────────────────────

    def close(self):
        """Disconnect from the robot and clean up resources."""
        if self._robot is not None:
            try:
                self._robot.disconnect()
                self._logger.info("SO101 robot disconnected.")
            except Exception as e:
                self._logger.warning(f"Error disconnecting robot: {e}")
            self._robot = None

        if self._leader is not None:
            try:
                self._leader.disconnect()
                self._logger.info("SO101 leader disconnected.")
            except Exception as e:
                self._logger.warning(f"Error disconnecting leader: {e}")
            self._leader = None

        self._state.is_connected = False

    def __del__(self):
        self.close()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def robot(self):
        """The underlying LeRobot :class:`~lerobot.robots.so_follower.SO101Follower` instance."""
        return self._robot

    @property
    def leader(self):
        """The underlying LeRobot :class:`~lerobot.teleoperators.so_leader.SO101Leader` instance."""
        return self._leader
