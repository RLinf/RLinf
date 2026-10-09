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

"""UR5 environment: absolute joint targets, no gripper.

The reach env scores a UR5 in joint space, the smallest task that brings a new
arm up end to end. The arm also reports a tool pose, but this env commands and
observes joints only; a Cartesian or grasp task adds what it needs on top.
"""

import copy
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
    UR5Config,
    UR5Robot,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.ur5 import UR5Arm
from rlinf.robotics.parts.base import PartGroup
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

#: Joint travel used for the action space, in radians. The UR5 allows +/- 2*pi;
#: these conservative bench limits keep a reach task clear of self-collision.
_DEFAULT_JOINT_LIMIT_LOW = np.full(6, -np.pi)
_DEFAULT_JOINT_LIMIT_HIGH = np.full(6, np.pi)

#: Arm joints, matching :pyattr:`UR5Arm.DOF`.
_DOF = UR5Arm.DOF


@dataclass
class UR5EnvConfig:
    """Task, control, and observation settings for a UR5 environment."""

    enable_camera_player: bool = True
    """Whether to show captured frames in a viewer window."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    camera_max_age: float = 0.5
    """Maximum age in seconds of a camera frame reused between control steps."""

    reset_joint_qpos: list[float] = field(default_factory=lambda: [0.0] * _DOF)
    """Rest configuration, in radians."""

    reset_duration: float = 3.0
    """Minimum reset trajectory duration in seconds."""

    reset_joint_speed: float = np.deg2rad(30.0)
    """Maximum commanded joint speed during reset, in radians per second."""

    reset_on_init: bool = True
    """Whether construction moves the arm to its reset pose."""

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

    def __post_init__(self) -> None:
        """Coerce the array-valued fields, which may arrive as lists."""
        self.joint_limit_low = np.asarray(self.joint_limit_low, dtype=np.float64)
        self.joint_limit_high = np.asarray(self.joint_limit_high, dtype=np.float64)


class UR5Env(gym.Env):
    """UR5 environment with absolute joint-position actions.

    An action is ``(6,)``: six joint positions in radians. Reward compares the
    measured joints with :pyattr:`UR5EnvConfig.target_joint_qpos`.
    """

    TELEOP = ()
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: UR5EnvConfig,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[UR5Config]]" = None,
        env_idx: int = 0,
    ) -> None:
        self._logger = get_logger()
        self.config = config
        if not np.isfinite(config.step_frequency) or config.step_frequency <= 0:
            raise ValueError("step_frequency must be finite and positive.")
        if not np.isfinite(config.camera_max_age) or config.camera_max_age < 0:
            raise ValueError("camera_max_age must be finite and nonnegative.")
        for name in ("reset_duration", "reset_joint_speed"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.hardware = get_hardware_config(
            UR5Config, robot_info, is_dummy=config.is_dummy
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
        self._joints = np.zeros(_DOF)
        self.robot: Optional[UR5Robot] = None

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

    # Hardware setup.

    def _setup_hardware(self) -> None:
        """Compose and connect the configured hardware."""
        assert self.env_idx >= 0, "env_idx must be nonnegative."
        hardware = self.hardware
        controller_node_rank = hardware.controller_node_rank
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self.robot = UR5Robot.build(
            robot_ip=hardware.robot_ip,
            backend=hardware.backend,
            speed=hardware.speed,
            acceleration=hardware.acceleration,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            cameras={info.name: info for info in self._camera_infos()},
        )
        self.robot.connect()
        # The arm part, for the operations the Arm contract names. Reading and
        # commanding go through the robot, not through this handle.
        self._arm = self.robot.child("arm")

    def _init_action_obs_spaces(self) -> None:
        """Build the joint-space action and observation spaces."""
        self._joint_limit_low = np.asarray(
            self.config.joint_limit_low, dtype=np.float64
        )
        self._joint_limit_high = np.asarray(
            self.config.joint_limit_high, dtype=np.float64
        )

        action_low = self._joint_limit_low.astype(np.float32)
        action_high = self._joint_limit_high.astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        spaces: dict[str, gym.Space] = {
            "state": gym.spaces.Dict(
                {
                    "arm_joint_position": gym.spaces.Box(
                        -np.inf, np.inf, shape=(_DOF,)
                    ),
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
        """Return the joint-position action part."""
        return (ActionPart("arm", _DOF, ActionKind.JOINT_POSITION),)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Args:
            action: ``(6,)`` float array of absolute joint positions in
                radians, bounded by the configured limits.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.monotonic()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            self.robot.send_action(
                {
                    "arm": {
                        "joint_position": np.clip(
                            action[:_DOF],
                            self._joint_limit_low,
                            self._joint_limit_high,
                        ),
                    }
                }
            )

        self._num_steps += 1
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)

        terminated = (
            reward >= 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        step_time = time.monotonic() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
        return observation, reward, terminated, truncated, {}

    def get_joint_positions(self) -> np.ndarray:
        """Arm joints as ``(1, 6)``, the shape teleop bindings index by arm."""
        return self._joints.reshape(1, -1).copy()

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
        if self.config.is_dummy:
            return self._get_observation(), {}

        self.go_to_rest()
        return self._get_observation(), {}

    def go_to_rest(self) -> None:
        """Move to :pyattr:`UR5EnvConfig.reset_joint_qpos`."""
        self._arm.reset_joint(
            self.config.reset_joint_qpos,
            duration=self.config.reset_duration,
            max_velocity=self.config.reset_joint_speed,
        )

    # Reward.

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        """Score the joint distance to the target configuration."""
        measured = np.asarray(observation["state"]["arm_joint_position"], dtype=float)
        target = np.asarray(self.config.target_joint_qpos, dtype=float)
        distance = np.abs(measured - target)

        if self.config.use_dense_reward:
            return float(-np.linalg.norm(distance))
        hit = bool(np.all(distance < self.config.reward_threshold))
        self._success_hold_counter = self._success_hold_counter + 1 if hit else 0
        return 1.0 if hit else 0.0

    # Observation.

    def _get_observation(self) -> dict[str, Any]:
        """Return the joint state and any camera frames."""
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        reading = self._state_parts.get_observation()["arm"]

        # The driver works in float64; the declared space is float32, and an
        # observation outside its own space fails Gymnasium's env checker.
        self._joints = np.asarray(reading["arm_joint_position"], dtype=float)
        observation: dict[str, Any] = {
            "state": {
                "arm_joint_position": np.asarray(
                    reading["arm_joint_position"], dtype=np.float32
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
        """Read one frame per camera, reopening a camera that has stalled."""
        import queue

        declared = self.observation_space["frames"]
        frames: dict[str, np.ndarray] = {}
        for camera in getattr(self, "_cameras", []):
            camera: BaseCamera
            name = camera.name
            try:
                # Cameras deliver their native resolution; the space fixes one.
                size = declared[name].shape[:2][::-1]
                frames[name] = self._crop_frame(
                    camera.get_frame(max_age=self.config.camera_max_age), size
                )
            except queue.Empty:
                self._logger.warning(
                    f"Camera {name} is not producing frames. Waiting 5s and retrying."
                )
                time.sleep(5)
                # Reopen this camera rather than rebuilding the declarations,
                # which would drop the placement the robot gave it.
                camera.reopen()
                return self._get_camera_frames()

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
