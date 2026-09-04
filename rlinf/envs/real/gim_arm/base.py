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

import copy
import queue
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import (
    Camera,
    GimArmConfig,
    GimArmRobot,
    Robot,
    RobotInfo,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.gim_arm import GimArmRobotState
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

# GIM_ARM_XL joint limits from the gim_arm_control SDK, in radians.
# Override them when using the standard GIM_ARM variant.
_DEFAULT_JOINT_LIMIT_LOW = np.array([-1.4, -3.0, 0.0, -1.5, -1.5, -1.88])
_DEFAULT_JOINT_LIMIT_HIGH = np.array([1.4, 0.0, 3.0, 1.5, 1.5, 1.90])


@dataclass
class GimArmRobotConfig:
    """Configuration for :class:`GimArmEnv`.

    Unset hardware connection fields are populated from
    :class:`~rlinf.robotics.RobotInfo`.
    """

    can_interface: Optional[str] = None
    """CAN socket interface name (e.g. ``"can0"``)."""

    arm_variant: Optional[str] = None
    """Arm variant: ``"gim_arm"`` or ``"gim_arm_xl"``."""

    camera_serials: Optional[list[str]] = None
    """Ordered list of camera serial numbers for observations."""

    camera_type: Optional[str] = None
    """Camera backend: ``"realsense"`` or ``"zed"``."""

    enable_gripper: bool = True
    """Whether the gripper is attached."""

    gripper_type: str = "parallel"
    """Gripper mechanical type: ``"parallel"`` or ``"single_side"``."""

    control_mode: str = "momentum_observer"
    """Arm control mode: ``"idle"``, ``"gravity_comp"``, ``"momentum_observer"``, ``"position"``, or ``"torque"``."""

    enable_camera_player: bool = True
    """Display a live camera window during episodes."""

    is_dummy: bool = False
    """When ``True``, skip all hardware calls (useful for offline training)."""

    use_dense_reward: bool = False
    """Use distance-based dense reward instead of binary 0/1."""

    step_frequency: float = 10.0
    """Maximum environment steps per second."""

    # Target and reset poses.
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    """Target end-effector pose ``[x, y, z, rx, ry, rz]`` (m / Euler XYZ).
    Used only for reward computation, not motion control."""

    reset_joint_qpos: list[float] = field(default_factory=lambda: [0.0] * 6)
    """Joint configuration to move to at the start of each episode."""

    joint_reset_qpos: list[float] = field(default_factory=lambda: [0.0] * 6)
    """Joint configuration for a full periodic joint reset."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_LOW.copy()
    )
    """Lower joint limits ``(6,)`` in radians used to clamp actions.
    Default values are for the GIM_ARM_XL variant."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _DEFAULT_JOINT_LIMIT_HIGH.copy()
    )
    """Upper joint limits ``(6,)`` in radians used to clamp actions.
    Default values are for the GIM_ARM_XL variant."""

    max_num_steps: int = 100
    """Episode truncation horizon."""

    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Per-axis tolerances ``[x, y, z, rx, ry, rz]`` for the success check.

    Only the XYZ entries are currently consulted; the orientation entries
    are accepted for Franka-API parity and reserved for future use.
    """

    binary_gripper_threshold: float = 0.5
    """Action magnitude threshold for open/close gripper transitions."""

    enable_gripper_penalty: bool = True
    """Subtract ``gripper_penalty`` from the reward on each gripper state change."""

    gripper_penalty: float = 0.1
    """Reward penalty per effective gripper action."""

    save_video_path: Optional[str] = None
    """Path to save episode videos. ``None`` disables saving."""

    joint_reset_cycle: int = 20000
    """Number of episode resets between full joint resets."""

    success_hold_steps: int = 1
    """Number of consecutive steps in the target zone required for success."""


class GimArmEnv(gym.Env):
    """Six-DOF GimArm environment with absolute joint-position actions.

    Actions contain six joint positions in radians and one binary gripper
    command. Rewards compare the forward-kinematics TCP pose with the target.
    """

    # No registered teleoperation device currently produces this joint layout.
    TELEOP = ()
    TELEOP_DEFAULT = "none"
    # The legacy GimArm task was registered directly and had no wrapper stack.
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: GimArmRobotConfig,
        worker_info: Optional[WorkerInfo],
        robot_info: Optional[RobotInfo[GimArmConfig]],
        env_idx: int,
    ) -> None:
        self._logger = get_logger()
        self.config = config
        self.robot_info = robot_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._state = GimArmRobotState()
        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)
        self._success_hold_counter = 0
        self.robot: Robot | None = None

        if not self.config.is_dummy:
            self._setup_hardware()

        # Camera-free setups expose an empty frames dictionary.
        if self.config.camera_serials is None:
            self.config.camera_serials = []
        if not self.config.camera_serials:
            self._logger.info(
                "No camera serials configured. "
                "Observations will not contain camera frames."
            )

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the controller's first valid state.
        start_time = time.time()
        while not self._controller.is_robot_up():
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time:.0f}s for GimArm to be ready."
                )

        self._controller.reset_joint(self.config.reset_joint_qpos)
        time.sleep(1.0)
        self._state = self._controller.get_state()

        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    # Hardware setup.

    def _setup_hardware(self) -> None:
        assert self.env_idx >= 0, "env_idx must be set for GimArmEnv."
        assert isinstance(self.robot_info, RobotInfo) and isinstance(
            self.robot_info.config, GimArmConfig
        ), f"robot_info must contain a GimArmConfig, but got {type(self.robot_info)}."

        # Fill unset connection fields from enumerated hardware configuration.
        if self.config.can_interface is None:
            self.config.can_interface = self.robot_info.config.can_interface
        if self.config.arm_variant is None:
            self.config.arm_variant = self.robot_info.config.arm_variant
        if self.config.camera_serials is None:
            self.config.camera_serials = self.robot_info.config.camera_serials
        if self.config.camera_type is None:
            self.config.camera_type = getattr(
                self.robot_info.config, "camera_type", "realsense"
            )

        controller_node_rank = getattr(
            self.robot_info.config, "controller_node_rank", None
        )
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self.robot = GimArmRobot.build(
            can_interface=self.config.can_interface,
            arm_variant=self.config.arm_variant,
            enable_gripper=self.config.enable_gripper,
            gripper_type=self.config.gripper_type,
            control_mode=self.config.control_mode,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            cameras={info.name: info for info in self._camera_infos()},
        )
        self.robot.connect()
        # The arm part, for the operations the Arm contract names. Reading
        # and commanding go through the robot, not through this handle.
        self._controller = self.robot.child("arm")

    def _init_action_obs_spaces(self) -> None:
        """Initialize action and observation spaces."""
        self._joint_limit_low = np.array(self.config.joint_limit_low, dtype=np.float64)
        self._joint_limit_high = np.array(
            self.config.joint_limit_high, dtype=np.float64
        )

        # Six bounded joint positions followed by a binary gripper command.
        action_low = np.append(self._joint_limit_low, -1.0).astype(np.float32)
        action_high = np.append(self._joint_limit_high, 1.0).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "arm_joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.camera_serials or []))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # Gymnasium API.

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return the joint-position and gripper action parts."""
        return (
            ActionPart("arm", 6, ActionKind.JOINT_POSITION),
            ActionPart("end_effector", 1, ActionKind.GRIPPER),
        )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: ``(7,)`` float array.
                ``action[:6]`` are absolute joint positions in radians,
                bounded by the configured joint limits.
                ``action[6]`` is the gripper command (binary open/close).

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            q_target = np.clip(
                action[:6], self._joint_limit_low, self._joint_limit_high
            )
            self.robot.send_action({"arm": {"joint_position": q_target}})

            gripper_action = float(action[6])
            is_gripper_effective = self._gripper_action(gripper_action)
        else:
            is_gripper_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._state = self._controller.get_state()

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_effective)

        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps

        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def reset(
        self,
        joint_reset: bool = False,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to the rest pose."""
        # A run with no hardware samples this space instead of reading,
        # so seeding it is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._base_observation_space)
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0

        joint_reset_cycle = next(self._joint_reset_cycle)
        if joint_reset_cycle == 0:
            self._logger.info(
                f"Periodic joint reset triggered after "
                f"{self.config.joint_reset_cycle} resets."
            )
            joint_reset = True

        self.go_to_rest(joint_reset)
        self._num_steps = 0
        self._state = self._controller.get_state()
        return self._get_observation(), {}

    def go_to_rest(self, joint_reset: bool = False) -> None:
        """Move to the rest configuration.

        If ``joint_reset`` is ``True``, move to :attr:`joint_reset_qpos`
        first (full range of motion); otherwise move to
        :attr:`reset_joint_qpos`.
        """
        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos)
            time.sleep(0.5)

        self._controller.reset_joint(self.config.reset_joint_qpos)

    # Reward calculation.

    def _calc_step_reward(
        self,
        observation: dict,
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute sparse or dense reward from TCP distance to the target."""
        if not self.config.is_dummy:
            euler_angles = np.abs(
                R.from_quat(self._state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)

            is_in_target_zone = np.all(
                target_delta[:3] <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                self._success_hold_counter += 1
                reward = 1.0
            else:
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = float(np.exp(-500.0 * np.sum(np.square(target_delta[:3]))))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Not in target zone. delta={target_delta}, reward={reward}"
                )

            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty

            return reward
        return 0.0

    # Observation construction.

    def _get_observation(self) -> dict[str, Any]:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "tcp_pose": self._state.tcp_pose,
                "tcp_vel": self._state.tcp_vel,
                "arm_joint_position": self._state.arm_joint_position,
                "gripper_position": np.array([self._state.gripper_position]),
                "tcp_force": self._state.tcp_force,
                "tcp_torque": self._state.tcp_torque,
            }
            return copy.deepcopy({"state": state, "frames": frames})
        return self._base_observation_space.sample()

    # Camera handling.

    def _camera_infos(self) -> list[CameraInfo]:
        """Return declarations for the configured wrist cameras."""
        camera_type = self.config.camera_type or "realsense"
        return [
            CameraInfo(
                name=f"wrist_{index + 1}",
                serial_number=serial,
                camera_type=camera_type,
            )
            for index, serial in enumerate(self.config.camera_serials or [])
        ]

    def _open_cameras(self) -> None:
        """Take the cameras the robot composed and connected."""
        self._cameras: list[BaseCamera] = list(
            self.robot.parts_of_type(Camera).values()
        )

    def _close_cameras(self) -> None:
        """Drop the camera references; the robot closes what it opened."""
        self._cameras = []

    def close(self) -> None:
        """Release cameras and detach the composed robot runtime."""
        if hasattr(self, "_cameras"):
            self._close_cameras()
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if self.robot is not None:
            self.robot.disconnect()
        super().close()

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Centre-crop to square then resize."""
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
        resized = cv2.resize(cropped, reshape_size)
        return cropped, resized

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        display_frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self.observation_space["frames"][camera.name].shape[:2][
                    ::-1
                ]
                cropped, resized = self._crop_frame(frame, reshape_size)
                frames[camera.name] = resized[..., ::-1]  # BGR input.
                display_frames[camera.name] = resized
                display_frames[f"{camera.name}_full"] = cropped
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera.name} not producing frames. "
                    "Waiting 5s and retrying."
                )
                time.sleep(5)
                camera.reopen()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_frames)
        return frames

    # Gripper control.

    def _gripper_action(self, position: float) -> bool:
        """Execute a binary gripper open/close.

        Returns:
            ``True`` if a gripper state transition occurred (penalty applies).
        """
        if not self.config.enable_gripper:
            return False
        closing = position <= -self.config.binary_gripper_threshold
        opening = position >= self.config.binary_gripper_threshold
        # Only act on an edge, so a held command is not resent every step.
        if not (closing and self._state.gripper_open) and not (
            opening and not self._state.gripper_open
        ):
            return False

        # The gripper rides on the arm, and its view is binary: the sign of
        # the target opens or closes it.
        self.robot.send_action(
            {"arm": {"end_effector": {"target": np.array([1.0 if opening else -1.0])}}}
        )
        time.sleep(0.6)
        return True

    # Utilities.

    @property
    def target_ee_pose(self) -> np.ndarray:
        """Target EEF pose as ``[x, y, z, qx, qy, qz, qw]``."""
        return np.concatenate(
            [
                self.config.target_ee_pose[:3],
                R.from_euler("xyz", self.config.target_ee_pose[3:].copy()).as_quat(),
            ]
        ).copy()
