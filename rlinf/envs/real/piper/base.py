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

"""Piper environment: absolute joint targets and a continuous gripper.

The Piper reports its six joints and the tool pose its own controller solves
for. Both reach the observation, but the reward here is joint-space: a task
that scores Cartesian error can read ``tcp_pose`` from the same observation.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import (
    Camera,
    PiperConfig,
    PiperRobot,
    RobotInfo,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.piper import PiperArm
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

#: Arm joints, matching :pyattr:`PiperArm.DOF`.
_DOF = PiperArm.DOF


@dataclass
class PiperRobotConfig:
    """Environment-side configuration for a Piper.

    Connection fields left as ``None`` are filled from the enumerated
    :class:`~rlinf.robotics.RobotInfo`.
    """

    backend: Optional[str] = None
    """Arm backend. ``None`` takes it from hardware info, which in turn
    defaults to :attr:`PiperRobot.BACKEND`."""

    can_channel: Optional[str] = None
    """CAN channel the arm is on. ``None`` takes it from hardware info."""

    can_interface: Optional[str] = None
    """python-can backend. ``None`` takes it from hardware info."""

    model: Optional[str] = None
    """Arm variant. ``None`` takes it from hardware info."""

    firmware: Optional[str] = None
    """Firmware profile. ``None`` takes it from hardware info, which in turn
    reads the version off the arm."""

    speed_percent: Optional[int] = None
    """Percentage of maximum speed. ``None`` takes it from hardware info."""

    with_gripper: bool = True
    """Whether an AgxGripper is fitted."""

    camera_serials: Optional[list[str]] = None
    """Camera identifiers. ``None`` takes them from hardware info."""

    camera_type: Optional[str] = None
    """Camera backend. ``None`` takes it from hardware info."""

    enable_camera_player: bool = True
    """Whether to show captured frames in a viewer window."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    reset_joint_qpos: list[float] = field(default_factory=lambda: [0.0] * _DOF)
    """Rest configuration, in radians."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: PiperArm.JOINT_LIMITS_LOWER.copy()
    )
    """Lower joint bounds, in radians."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: PiperArm.JOINT_LIMITS_UPPER.copy()
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


class PiperEnv(gym.Env):
    """Piper environment with absolute joint-position actions.

    An action is ``(7,)``: six joint positions in radians followed by a gripper
    opening in ``0..1``. Reward compares the measured joints with
    :pyattr:`PiperRobotConfig.target_joint_qpos`.
    """

    # No registered teleoperation device produces this joint layout.
    TELEOP = ()
    TELEOP_DEFAULT = "none"
    # The gripper is continuous, so the one-axis binary wrapper does not fit.
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: PiperRobotConfig,
        worker_info: Optional[WorkerInfo],
        robot_info: "Optional[RobotInfo[PiperConfig]]",
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

        self._num_steps = 0
        self._success_hold_counter = 0
        self._last_gripper: Optional[float] = None
        self._joints = np.zeros(_DOF)
        self.robot: Optional[PiperRobot] = None

        if not self.config.is_dummy:
            self._setup_hardware()

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

        self._arm.reset_joint(self.config.reset_joint_qpos)
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    # Hardware setup.

    def _setup_hardware(self) -> None:
        """Compose and connect the robot, filling gaps from hardware info."""
        assert self.env_idx >= 0, "env_idx must be set for PiperEnv."

        controller_node_rank = None
        if self.robot_info is not None:
            hardware = self.robot_info.config
            for name in (
                "backend",
                "can_channel",
                "can_interface",
                "model",
                "firmware",
                "speed_percent",
                "camera_serials",
                "camera_type",
            ):
                if getattr(self.config, name, None) is None:
                    setattr(self.config, name, getattr(hardware, name, None))
            controller_node_rank = getattr(hardware, "controller_node_rank", None)

        if self.config.can_channel is None:
            raise ValueError(
                "A Piper needs a CAN channel. Set 'can_channel' on the env "
                "config, or let hardware discovery supply it."
            )
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        # Cameras are declared here, after discovery has filled the serials.
        self.robot = PiperRobot.build(
            can_channel=self.config.can_channel,
            backend=self.config.backend,
            can_interface=self.config.can_interface or "socketcan",
            model=self.config.model or "piper",
            firmware=self.config.firmware,
            speed_percent=(
                30 if self.config.speed_percent is None else self.config.speed_percent
            ),
            with_gripper=self.config.with_gripper,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            cameras={info.name: info for info in self._camera_infos()},
        )
        self.robot.connect()
        # The arm part, for the operations the Arm contract names. Reading and
        # commanding go through the robot, not through this handle.
        self._arm: PiperArm = self.robot.child("arm", PiperArm)

    def _init_action_obs_spaces(self) -> None:
        """Build the joint-space action and observation spaces."""
        self._joint_limit_low = np.asarray(
            self.config.joint_limit_low, dtype=np.float64
        )
        self._joint_limit_high = np.asarray(
            self.config.joint_limit_high, dtype=np.float64
        )

        # Six bounded joints, then a gripper opening in 0..1.
        action_low = np.append(self._joint_limit_low, 0.0).astype(np.float32)
        action_high = np.append(self._joint_limit_high, 1.0).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        state: dict[str, gym.Space] = {
            "arm_joint_position": gym.spaces.Box(-np.inf, np.inf, shape=(_DOF,)),
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
        }
        if self.config.with_gripper:
            state["gripper_position"] = gym.spaces.Box(0, 1, shape=(1,))
        spaces: dict[str, gym.Space] = {"state": gym.spaces.Dict(state)}

        frames = {
            f"wrist_{index + 1}": gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )
            for index in range(len(self.config.camera_serials or []))
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
        parts = [ActionPart("arm", _DOF, ActionKind.JOINT_POSITION)]
        if self.config.with_gripper:
            parts.append(ActionPart("end_effector", 1, ActionKind.GRIPPER))
        return tuple(parts)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Args:
            action: ``(7,)`` float array. ``action[:6]`` are absolute joint
                positions in radians, bounded by the configured limits.
                ``action[6]`` is the gripper opening, ``0`` shut to ``1`` open.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        gripper_moved = False
        if not self.config.is_dummy:
            arm_action: dict[str, Any] = {
                "joint_position": np.clip(
                    action[:_DOF], self._joint_limit_low, self._joint_limit_high
                )
            }
            if self.config.with_gripper:
                opening = float(np.clip(action[_DOF], 0.0, 1.0))
                gripper_moved = (
                    self._last_gripper is not None
                    and abs(opening - self._last_gripper) > self.config.reward_threshold
                )
                self._last_gripper = opening
                arm_action["end_effector"] = {"target": np.array([opening])}
            # The gripper rides beneath the arm, so one call dispatches both.
            self.robot.send_action({"arm": arm_action})

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, gripper_moved)

        terminated = (
            reward >= 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
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
        self._last_gripper = None
        if self.config.is_dummy:
            return self._get_observation(), {}

        self.go_to_rest()
        return self._get_observation(), {}

    def go_to_rest(self) -> None:
        """Move to :pyattr:`PiperRobotConfig.reset_joint_qpos`."""
        self._arm.reset_joint(self.config.reset_joint_qpos)

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
        """Return the arm state and any camera frames."""
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        # One read of the whole robot, gripper included.
        reading = self.robot.get_observation()["arm"]

        # The driver works in float64; an observation outside its own declared
        # space fails Gymnasium's env checker.
        self._joints = np.asarray(reading["arm_joint_position"], dtype=float)
        state: dict[str, Any] = {
            "arm_joint_position": np.asarray(
                reading["arm_joint_position"], dtype=np.float32
            ),
            "tcp_pose": np.asarray(reading["tcp_pose"], dtype=np.float32),
        }
        if self.config.with_gripper:
            state["gripper_position"] = np.asarray(
                reading["end_effector"]["state"], dtype=np.float32
            )
        observation: dict[str, Any] = {"state": state}
        # Kept in step with the space, which omits 'frames' when there are none.
        if "frames" in self.observation_space.spaces:
            observation["frames"] = self._get_camera_frames()
        return copy.deepcopy(observation)

    # Cameras.

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
                frames[name] = self._crop_frame(camera.get_frame(), size)
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
