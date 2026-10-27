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

"""LeKiwi environment: bounded body velocity and a front camera.

The LeKiwi reports the body velocity its wheels measure and accepts the
velocity to command, both ``[x, y, theta]`` in m/s, m/s, and deg/s. There is
no position or pose sensing, so the state is the measured velocity and the
reward is zero: the tasks this base serves -- manual control, teleoperation,
navigation -- decide success themselves.
"""

import copy
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.image import crop_and_resize
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import Camera, LeKiwiConfig, LeKiwiRobot, RobotInfo
from rlinf.robotics.parts.base import PartGroup
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger


@dataclass
class LeKiwiEnvConfig:
    """Task, control, and observation settings for a LeKiwi environment."""

    enable_camera_player: bool = True
    """Whether to show captured frames in a viewer window."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step commands velocity for one period."""

    camera_max_age: float = 0.5
    """How old, in seconds, a frame may be and still be reused by a step.

    A camera publishes on its own schedule, usually faster than the control
    loop, so a step takes the newest frame within this age instead of waiting
    for one captured after the step began. ``0`` makes every step wait for a
    new frame, which stalls the control loop when the camera is slower than
    ``step_frequency``.
    """

    camera_names: dict[str, str] = field(default_factory=dict)
    """Camera serial to frame name, such as ``{"<serial>": "front"}``.
    Cameras left unlisted report under ``camera_1`` and so on."""

    frame_size: tuple[int, int] = (128, 128)
    """Width and height of every frame in the observation, in pixels."""

    max_linear_velocity: float = 0.5
    """Absolute bound on ``x`` and ``y`` commands, in metres per second."""

    max_angular_velocity: float = 90.0
    """Absolute bound on ``theta`` commands, in degrees per second."""

    max_num_steps: int = 100
    """Steps before an episode is truncated."""

    def __post_init__(self) -> None:
        """Coerce the tuple-valued field, which may arrive as a list."""
        self.frame_size = (int(self.frame_size[0]), int(self.frame_size[1]))


class LeKiwiEnv(gym.Env):
    """LeKiwi mobile-base environment with velocity actions.

    An action is ``(3,)``: the body velocity ``[x, y, theta]`` in m/s, m/s,
    and deg/s, clipped to the configured limits. A step holds the command for
    one control period and then stops the wheels, so the base only moves
    while the policy keeps stepping -- a dead-man's grip on the hardware.
    """

    # No teleoperation device is bound to a velocity-controlled base yet.
    TELEOP: tuple[str, ...] = ()
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: LeKiwiEnvConfig,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[LeKiwiConfig]]" = None,
        env_idx: int = 0,
    ) -> None:
        self._logger = get_logger()
        self.config = config
        for name, value in (
            ("step_frequency", config.step_frequency),
            ("camera_max_age", config.camera_max_age),
            ("max_linear_velocity", config.max_linear_velocity),
            ("max_angular_velocity", config.max_angular_velocity),
            ("frame_size[0]", config.frame_size[0]),
            ("frame_size[1]", config.frame_size[1]),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.hardware = get_hardware_config(
            LeKiwiConfig, robot_info, is_dummy=config.is_dummy
        )
        self.robot_info = robot_info
        self.env_idx = env_idx
        # Parts ride scheduler workers when this env itself runs in one; a
        # standalone process owns its hardware in-process instead.
        self.node_rank = (
            worker_info.cluster_node_rank if worker_info is not None else None
        )
        self.env_worker_rank = worker_info.rank if worker_info is not None else 0

        self._num_steps = 0
        self.robot: Optional[LeKiwiRobot] = None

        if not self.config.is_dummy:
            self._setup_hardware()
            # This view borrows the base; self.robot owns its connection.
            self._state_parts = PartGroup(base=self._base)

        if not self.hardware.camera_serials:
            self._logger.info(
                "No camera serials configured. "
                "Observations will not contain camera frames."
            )

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    # Hardware setup.

    def _setup_hardware(self) -> None:
        """Compose and connect the configured hardware."""
        if self.hardware.port is None:
            raise ValueError(
                "Real hardware needs a serial port: set 'port' on the LeKiwi "
                "hardware config."
            )
        node_rank = self.hardware.node_rank
        if node_rank is None:
            node_rank = self.node_rank
        self.robot = LeKiwiRobot.build(
            port=self.hardware.port,
            wheel_radius=self.hardware.wheel_radius,
            base_radius=self.hardware.base_radius,
            max_raw=self.hardware.max_raw,
            env_idx=self.env_idx,
            node_rank=node_rank,
            worker_rank=self.env_worker_rank,
            cameras={info.name: info for info in self._camera_infos()},
        )
        self.robot.connect()
        # The base part, for the stop operation the MobileBase contract names.
        # Commanding goes through the robot, not through this handle.
        self._base = self.robot.child("base")

    def _init_action_obs_spaces(self) -> None:
        """Build the velocity action and speed-plus-frames observation spaces."""
        limit = np.array(
            [
                self.config.max_linear_velocity,
                self.config.max_linear_velocity,
                self.config.max_angular_velocity,
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-limit, limit)

        width, height = self.config.frame_size
        spaces: dict[str, gym.Space] = {
            "state": gym.spaces.Dict(
                {"velocity": gym.spaces.Box(-np.inf, np.inf, shape=(3,))}
            )
        }
        frames = {
            name: gym.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)
            for name in self._frame_names()
        }
        # Gymnasium's env checker rejects an empty Dict space, so a base with
        # no camera reports no 'frames' key at all rather than an empty one.
        if frames:
            spaces["frames"] = gym.spaces.Dict(frames)

        self.observation_space = gym.spaces.Dict(spaces)
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # Gymnasium API.

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one control period.

        Args:
            action: ``(3,)`` float array, the body velocity ``[x, y, theta]``
                in m/s, m/s, and deg/s, clipped to the configured limits.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
            The reward is always ``0.0``; a task decides success itself.
        """
        start_time = time.monotonic()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected an action of shape {self.action_space.shape}, "
                f"got {action.shape}."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            self.robot.send_action({"base": {"velocity": action}})
        self._num_steps += 1

        # Hold the command for the rest of the control period before reading
        # the wheels, so the observation measures the commanded motion.
        remaining = 1.0 / self.config.step_frequency - (time.monotonic() - start_time)
        time.sleep(max(0.0, remaining))
        observation = self._get_observation()

        # The step is over: stop the wheels before returning, so a policy
        # that stops stepping stops the base.
        if not self.config.is_dummy:
            self._base.reset()

        truncated = self._num_steps >= self.config.max_num_steps
        return observation, 0.0, False, truncated, {}

    @property
    def num_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._num_steps

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Stop the base and report the resting state."""
        # A run with no hardware samples this space instead of reading,
        # so seeding it is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._base_observation_space)
        self._num_steps = 0
        if not self.config.is_dummy:
            self._base.reset()
        return self._get_observation(), {}

    # Observation.

    def _get_observation(self) -> dict[str, Any]:
        """Return the measured body velocity and any camera frames."""
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        reading = self._state_parts.get_observation()["base"]

        observation: dict[str, Any] = {
            "state": {"velocity": np.asarray(reading["velocity"], dtype=np.float32)}
        }
        # Kept in step with the space, which omits 'frames' when there are none.
        if "frames" in self.observation_space.spaces:
            observation["frames"] = self._get_camera_frames()
        return copy.deepcopy(observation)

    # Cameras.

    def _frame_names(self) -> list[str]:
        """Return the observation frame name of every configured camera."""
        names = [
            self.config.camera_names.get(str(serial), f"camera_{index + 1}")
            for index, serial in enumerate(self.hardware.camera_serials or [])
        ]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Camera frame names must be unique; got {names}. Fix "
                "'camera_names' in the env config."
            )
        return names

    def _camera_infos(self) -> list[CameraInfo]:
        """Return declarations for the configured cameras."""
        camera_type = self.hardware.camera_type or "webcam"
        return [
            CameraInfo(
                name=name,
                serial_number=str(serial),
                camera_type=camera_type,
                resolution=self.config.frame_size,
            )
            for name, serial in zip(
                self._frame_names(), self.hardware.camera_serials or [], strict=True
            )
        ]

    def _open_cameras(self) -> None:
        """Take the cameras the robot composed and connected."""
        self._cameras: list[BaseCamera] = list(
            self.robot.parts_of_type(Camera).values()
        )

    def _close_cameras(self) -> None:
        """Drop the camera references; the robot closes what it opened."""
        self._cameras = []

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Read one frame per camera, reopening a camera that has stalled.

        A camera that delivers nothing is reopened once. A second timeout
        raises instead of retrying forever: a stalled camera cannot be driven
        around, and a step that never returns hides the fault.
        """
        declared = self.observation_space["frames"]
        frames: dict[str, np.ndarray] = {}
        for camera in getattr(self, "_cameras", []):
            camera: BaseCamera
            name = camera.name
            # Cameras deliver their native resolution; the space fixes one.
            size = declared[name].shape[:2][::-1]
            try:
                frame = camera.get_frame(max_age=self.config.camera_max_age)
            except queue.Empty:
                self._logger.warning(
                    "Camera %s is not producing frames; reopening it.", name
                )
                time.sleep(5)
                # Reopen this camera rather than rebuilding the declarations,
                # which would drop the placement the robot gave it.
                camera.reopen()
                try:
                    frame = camera.get_frame(max_age=self.config.camera_max_age)
                except queue.Empty as exc:
                    raise RuntimeError(
                        f"Camera {name} still delivers no frame after reopening."
                    ) from exc
            frames[name] = crop_and_resize(frame, size)

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
