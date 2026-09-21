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

"""SO-101 environment: absolute joint targets and a continuous gripper.

The SO-101 reports joint positions and nothing else. It has no pose, force,
or torque sensing, and no kinematic model ships with it, so both the
observation and the reward are joint-space. An env that needs Cartesian
quantities has to add forward kinematics of its own.
"""

import copy
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import (
    Camera,
    RobotInfo,
    SO101Config,
    SO101Robot,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.so101 import SO101Arm
from rlinf.robotics.parts.base import PartGroup
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

#: Joint travel of the SO-101, in radians. The servos turn further than this;
#: these are the limits the arm can hold without the links colliding.
_DEFAULT_JOINT_LIMIT_LOW = np.array([-1.91, -1.75, -1.69, -1.66, -2.79])
_DEFAULT_JOINT_LIMIT_HIGH = np.array([1.91, 1.75, 1.69, 1.66, 2.79])

#: Arm joints, matching :pyattr:`SO101Arm.MOTORS`.
_DOF = len(SO101Arm.MOTORS)

# Keep camera reads shorter than the 10 Hz control period. BaseCamera reopens
# the device between attempts, so the environment only needs one bounded read.
_CAMERA_FRAME_TIMEOUT_S = 0.5
_CAMERA_REOPEN_ATTEMPTS = 3


def _zero_observation(space: gym.Space) -> Any:
    """Build a deterministic zero value matching a Gymnasium space."""
    if isinstance(space, gym.spaces.Dict):
        return {key: _zero_observation(value) for key, value in space.spaces.items()}
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"Zero dummy observations do not support {type(space).__name__}.")


@dataclass
class SO101EnvConfig:
    """Task, control, and observation settings for an SO-101 environment."""

    enable_camera_player: bool = True
    """Whether to show captured frames in a viewer window."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    zero_dummy_observation: bool = False
    """Return deterministic zero observations for local integration tests."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    camera_max_age: float = 0.5
    """Maximum age in seconds of a camera frame reused between control steps."""

    reset_joint_qpos: list[float] = field(default_factory=lambda: [0.0] * _DOF)
    """Rest configuration, in radians."""

    reset_gripper_position: float = 0.0
    """Gripper opening at reset, from ``0`` closed to ``1`` open."""

    reset_duration: float = 3.0
    """Minimum reset trajectory duration in seconds."""

    reset_joint_speed: float = np.deg2rad(30.0)
    """Maximum commanded joint speed during reset, in radians per second."""

    reset_on_init: bool = True
    """Whether construction moves the arm to its reset pose."""

    park_joint_qpos: Optional[list[float]] = None
    """Joint configuration used by an explicit :meth:`SO101Env.park` call."""

    park_gripper_position: Optional[float] = None
    """Gripper opening used by an explicit :meth:`SO101Env.park` call."""

    park_duration: float = 3.0
    """Minimum park trajectory duration in seconds."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW.copy()
    )
    """Lower joint bounds, in radians."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH.copy()
    )
    """Upper joint bounds, in radians."""

    max_num_steps: int = 100
    """Steps before an episode is truncated."""

    target_joint_qpos: list[float] = field(default_factory=lambda: [0.0] * _DOF)
    """Goal configuration the reward measures against, in radians."""

    reward_threshold: float = 0.05
    """Per-joint tolerance in radians. Every joint must be within it."""

    use_dense_reward: bool = False
    """Report the negative joint distance instead of a sparse hit."""

    success_hold_steps: int = 1
    """Consecutive successful steps before the episode terminates."""

    gripper_penalty: float = 0.1
    """Reward subtracted when a gripper command changes its state."""

    enable_gripper_penalty: bool = False
    """Whether to charge :pyattr:`gripper_penalty` for gripper motion."""

    def __post_init__(self) -> None:
        """Coerce the array-valued fields, which may arrive as lists."""
        self.joint_limit_low = np.asarray(self.joint_limit_low, dtype=np.float64)
        self.joint_limit_high = np.asarray(self.joint_limit_high, dtype=np.float64)


class SO101Env(gym.Env):
    """SO-101 environment with absolute joint-position actions.

    An action is ``(6,)``: five joint positions in radians followed by a
    gripper opening in ``0..1``. Reward compares the measured joints with
    :pyattr:`SO101EnvConfig.target_joint_qpos`.
    """

    # The leader arm is the same five joints and gripper as this follower.
    TELEOP = ("so101_leader",)
    TELEOP_DEFAULT = "none"
    # Online DAgger and smooth intervention use this flag to mark expert steps.
    TELEOP_MARK_FLAG = True
    # The gripper is continuous, so the one-axis binary wrapper does not fit.
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: SO101EnvConfig,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[SO101Config]]" = None,
        env_idx: int = 0,
    ) -> None:
        self._logger = get_logger()
        self.config = config
        if not np.isfinite(config.step_frequency) or config.step_frequency <= 0:
            raise ValueError("step_frequency must be finite and positive.")
        if not np.isfinite(config.camera_max_age) or config.camera_max_age < 0:
            raise ValueError("camera_max_age must be finite and nonnegative.")
        for name in ("reset_duration", "reset_joint_speed", "park_duration"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self._validate_configured_pose(
            "reset", config.reset_joint_qpos, config.reset_gripper_position
        )
        if (config.park_joint_qpos is None) != (config.park_gripper_position is None):
            raise ValueError(
                "park_joint_qpos and park_gripper_position must be configured together."
            )
        if config.park_joint_qpos is not None:
            self._validate_configured_pose(
                "park", config.park_joint_qpos, config.park_gripper_position
            )
        self.hardware = get_hardware_config(
            SO101Config, robot_info, is_dummy=config.is_dummy
        )
        self.robot_info = robot_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._num_steps = 0
        self._success_hold_counter = 0
        self._last_gripper: Optional[float] = None
        self._last_camera_frame: dict[str, np.ndarray] = {}
        self._gripper_position = np.zeros(1, dtype=np.float32)
        self._joints = np.zeros(_DOF)
        self.robot: Optional[SO101Robot] = None

        if not self.config.is_dummy:
            self._setup_hardware()
            # This view borrows the arm; self.robot owns its connection.
            self._state_parts = PartGroup(arm=self._arm)

        if not self.hardware.camera_serials:
            self._logger.info(
                "No camera serials configured. "
                "Observations will not contain camera frames."
            )

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        if self.config.reset_on_init:
            self.go_to_rest()
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _validate_configured_pose(
        self, name: str, joints: list[float], gripper: Optional[float]
    ) -> None:
        """Validate a configured joint and gripper target."""
        target = np.asarray(joints, dtype=np.float64)
        if target.shape != (_DOF,) or not np.all(np.isfinite(target)):
            raise ValueError(f"{name}_joint_qpos must contain {_DOF} finite values.")
        if np.any(target < self.config.joint_limit_low) or np.any(
            target > self.config.joint_limit_high
        ):
            raise ValueError(f"{name}_joint_qpos must lie within the joint limits.")
        if gripper is None or not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ValueError(f"{name}_gripper_position must be between 0 and 1.")

    # Hardware setup.

    def _setup_hardware(self) -> None:
        """Compose and connect the configured hardware."""
        assert self.env_idx >= 0, "env_idx must be nonnegative."
        hardware = self.hardware
        controller_node_rank = hardware.controller_node_rank
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self.robot = SO101Robot.build(
            port=hardware.serial_port,
            calibration_id=hardware.calibration_id,
            max_relative_target=hardware.max_relative_target,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            cameras={info.name: info for info in self._camera_infos()},
        )
        self.robot.connect()
        # The arm part, for the operations the Arm contract names. Reading
        # and commanding go through the robot, not through this handle.
        self._arm = self.robot.child("arm")

    def _init_action_obs_spaces(self) -> None:
        """Build the joint-space action and observation spaces."""
        self._joint_limit_low = np.asarray(
            self.config.joint_limit_low, dtype=np.float64
        )
        self._joint_limit_high = np.asarray(
            self.config.joint_limit_high, dtype=np.float64
        )

        # Five bounded joints, then a gripper opening in 0..1.
        action_low = np.append(self._joint_limit_low, 0.0).astype(np.float32)
        action_high = np.append(self._joint_limit_high, 1.0).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        spaces: dict[str, gym.Space] = {
            "state": gym.spaces.Dict(
                {
                    "arm_joint_position": gym.spaces.Box(
                        -np.inf, np.inf, shape=(_DOF,)
                    ),
                    "gripper_position": gym.spaces.Box(0, 1, shape=(1,)),
                }
            )
        }
        frames = {
            f"wrist_{index + 1}": gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )
            for index in range(len(self.hardware.camera_serials or []))
        }
        # Gymnasium's env checker rejects an empty Dict space, so an arm with
        # no camera reports no 'frames' key at all rather than an empty one.
        if frames:
            spaces["frames"] = gym.spaces.Dict(frames)

        self.observation_space = gym.spaces.Dict(spaces)
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # Gymnasium API.

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return the joint-position and gripper action parts."""
        return (
            ActionPart("arm", _DOF, ActionKind.JOINT_POSITION),
            ActionPart("end_effector", 1, ActionKind.GRIPPER),
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Args:
            action: ``(6,)`` float array. ``action[:5]`` are absolute joint
                positions in radians, bounded by the configured limits.
                ``action[5]`` is the gripper opening, ``0`` shut to ``1`` open.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.monotonic()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        gripper_moved = False
        if not self.config.is_dummy:
            opening = float(np.clip(action[_DOF], 0.0, 1.0))
            gripper_moved = (
                self._last_gripper is not None
                and abs(opening - self._last_gripper) > self.config.reward_threshold
            )
            self._last_gripper = opening
            # One action for the whole arm: the gripper rides beneath it, so
            # the robot dispatches both without reaching for the driver.
            self.robot.send_action(
                {
                    "arm": {
                        "joint_position": np.clip(
                            action[:_DOF], self._joint_limit_low, self._joint_limit_high
                        ),
                        "end_effector": {"target": np.array([opening])},
                    }
                }
            )

        self._num_steps += 1
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, gripper_moved)

        terminated = (
            reward >= 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        step_time = time.monotonic() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
        return observation, reward, terminated, truncated, {}

    def get_joint_positions(self) -> np.ndarray:
        """Arm joints as ``(1, 5)``, the shape teleop bindings index by arm."""
        return self._joints.reshape(1, -1).copy()

    def get_gripper_position(self) -> np.ndarray:
        """Current gripper opening as ``(1, 1)`` for teleop context."""
        return self._gripper_position.reshape(1, -1).copy()

    @property
    def num_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._num_steps

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return the arm to its rest configuration."""
        # A run with no hardware samples this space instead of reading,
        # so seeding it is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._base_observation_space)
        self._num_steps = 0
        self._success_hold_counter = 0
        self._last_gripper = None
        if self.config.is_dummy:
            return self._get_observation(), {}

        self.go_to_rest()
        return self._get_observation(), {}

    def go_to_rest(self) -> None:
        """Move the joints and gripper to the configured reset state."""
        self._move_to_configured_pose(
            self.config.reset_joint_qpos,
            self.config.reset_gripper_position,
            self.config.reset_duration,
        )

    def park(self) -> None:
        """Move to the configured park state when explicitly requested."""
        if self.config.park_joint_qpos is None:
            raise RuntimeError(
                "Configure park_joint_qpos and park_gripper_position before parking."
            )
        assert self.config.park_gripper_position is not None
        self._move_to_configured_pose(
            self.config.park_joint_qpos,
            self.config.park_gripper_position,
            self.config.park_duration,
        )

    def _move_to_configured_pose(
        self, joints: list[float], gripper: float, duration: float
    ) -> None:
        """Move all six servos to one configured state."""
        self._arm.move_gripper([gripper])
        self._arm.reset_joint(
            joints,
            duration=duration,
            max_velocity=self.config.reset_joint_speed,
        )
        self._arm.wait_for_gripper()

    def get_reset_joint_positions(self) -> np.ndarray:
        """Return the reset joint target for coordinated teleoperation reset."""
        return np.asarray(self.config.reset_joint_qpos, dtype=np.float64)[None, :]

    def get_reset_gripper_position(self) -> np.ndarray:
        """Return the reset gripper target for coordinated teleoperation reset."""
        return np.asarray([[self.config.reset_gripper_position]], dtype=np.float64)

    def get_reset_duration(self) -> float:
        """Return the minimum reset trajectory duration in seconds."""
        return float(self.config.reset_duration)

    def get_reset_joint_speed(self) -> float:
        """Return the maximum reset joint speed in radians per second."""
        return float(self.config.reset_joint_speed)

    def get_park_joint_positions(self) -> Optional[np.ndarray]:
        """Return the configured park joint target."""
        if self.config.park_joint_qpos is None:
            return None
        return np.asarray(self.config.park_joint_qpos, dtype=np.float64)[None, :]

    def get_park_gripper_position(self) -> Optional[np.ndarray]:
        """Return the configured park gripper target."""
        if self.config.park_gripper_position is None:
            return None
        return np.asarray([[self.config.park_gripper_position]], dtype=np.float64)

    def get_park_duration(self) -> float:
        """Return the minimum park trajectory duration in seconds."""
        return float(self.config.park_duration)

    # Reward.

    def _calc_step_reward(
        self, observation: dict[str, Any], gripper_moved: bool = False
    ) -> float:
        """Score the joint distance to the target configuration."""
        measured = np.asarray(observation["state"]["arm_joint_position"], dtype=float)
        target = np.asarray(self.config.target_joint_qpos, dtype=float)
        distance = np.abs(measured - target)

        if self.config.use_dense_reward:
            reward = float(-np.linalg.norm(distance))
        else:
            hit = bool(np.all(distance < self.config.reward_threshold))
            self._success_hold_counter = self._success_hold_counter + 1 if hit else 0
            reward = 1.0 if hit else 0.0

        if gripper_moved and self.config.enable_gripper_penalty:
            reward -= self.config.gripper_penalty
        return reward

    # Observation.

    def _get_observation(self) -> dict[str, Any]:
        """Return the joint state and any camera frames."""
        if self.config.is_dummy:
            if self.config.zero_dummy_observation:
                return _zero_observation(self._base_observation_space)
            return self._base_observation_space.sample()
        reading = self._state_parts.get_observation()["arm"]

        # The driver works in float64; the declared space is float32, and an
        # observation outside its own space fails Gymnasium's env checker.
        self._joints = np.asarray(reading["arm_joint_position"], dtype=float)
        self._gripper_position = np.asarray(
            reading["end_effector"]["state"], dtype=np.float32
        ).reshape(1)
        observation: dict[str, Any] = {
            "state": {
                "arm_joint_position": np.asarray(
                    reading["arm_joint_position"], dtype=np.float32
                ),
                "gripper_position": np.asarray(
                    self._gripper_position, dtype=np.float32
                ),
            }
        }
        # Kept in step with the space, which omits 'frames' when there are none.
        if "frames" in self.observation_space.spaces:
            observation["frames"] = self._get_camera_frames()
        return copy.deepcopy(observation)

    # Cameras.

    def _camera_infos(self) -> list[CameraInfo]:
        """Return declarations for the configured wrist cameras."""
        camera_type = self.hardware.camera_type or "realsense"
        return [
            CameraInfo(
                name=f"wrist_{index + 1}",
                serial_number=serial,
                camera_type=camera_type,
            )
            for index, serial in enumerate(self.hardware.camera_serials or [])
        ]

    def _open_cameras(self) -> None:
        """Take the cameras the robot composed and connected."""
        self._cameras: list[BaseCamera] = list(
            self.robot.parts_of_type(Camera).values()
        )

    def _close_cameras(self) -> None:
        """Drop the camera references; the robot closes what it opened."""
        self._cameras = []

    def _crop_frame(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Centre-crop to a square, then resize to the declared frame size."""
        height, width, _ = frame.shape
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return cv2.resize(frame[top : top + side, left : left + side], size)

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Read camera frames, reusing the latest frame during recovery.

        ``BaseCamera.get_frame`` performs a bounded number of reconnect
        attempts. Once those attempts are exhausted, an existing frame keeps
        the control loop moving; a camera that has never produced a frame
        fails explicitly instead of recursing forever.
        """
        declared = self.observation_space["frames"]
        frames: dict[str, np.ndarray] = {}
        for camera in getattr(self, "_cameras", []):
            camera: BaseCamera
            name = camera.name
            try:
                # Cameras deliver their native resolution; the space fixes one.
                size = declared[name].shape[:2][::-1]
                raw_frame = camera.get_frame(
                    timeout=_CAMERA_FRAME_TIMEOUT_S,
                    attempts=_CAMERA_REOPEN_ATTEMPTS,
                    max_age=self.config.camera_max_age,
                )
            except queue.Empty:
                raw_frame = self._last_camera_frame.get(name)
                if raw_frame is None:
                    raise RuntimeError(
                        f"Camera {name} did not produce a frame after "
                        f"{_CAMERA_REOPEN_ATTEMPTS} attempts and has no cached frame."
                    ) from None
                self._logger.warning(
                    "Camera %s stalled after %d attempts; using the last frame.",
                    name,
                    _CAMERA_REOPEN_ATTEMPTS,
                )

            self._last_camera_frame[name] = np.asarray(raw_frame).copy()
            frames[name] = self._crop_frame(raw_frame, size)

        if hasattr(self, "camera_player"):
            self.camera_player.put_frame(frames)
        return frames

    def close(self) -> None:
        """Release the cameras and disconnect the robot."""
        if hasattr(self, "_cameras"):
            self._close_cameras()
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
