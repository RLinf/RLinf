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

"""UR5e Cartesian-delta environment for collection and policy execution."""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.pose import clip_euler_to_target_window, quat_slerp
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import Camera, RobotInfo, UR5eConfig, UR5eRobot
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.base import BaseArm
from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
from rlinf.robotics.parts.end_effectors import EndEffector
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger


@dataclass
class URCollectionEnvConfig:
    """Control and observation settings for UR5e Cartesian-delta episodes."""

    is_dummy: bool = False
    step_frequency: float = 10.0
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.1, 1.0]))
    """Metres, radians, and gripper-signal scale per normalized action."""

    ee_pose_limit_min: Optional[np.ndarray] = None
    ee_pose_limit_max: Optional[np.ndarray] = None
    """Required base-frame xyz and Euler bounds, in metres and radians."""

    reset_ee_pose: Optional[np.ndarray] = None
    """Required reset xyz and Euler pose, in metres and radians."""

    reset_duration: float = 3.0
    """Minimum duration of the interpolated TCP reset trajectory, in seconds."""
    reset_timeout: float = 5.0
    """Seconds to wait for the reset target after sending the trajectory."""
    reset_position_tolerance: float = 0.005
    reset_orientation_tolerance: float = 0.02

    binary_gripper_threshold: float = 0.5
    camera_names: Optional[dict[str, str]] = None
    camera_crop_regions: Optional[dict[str, list[float]]] = None
    image_size: tuple[int, int] = (224, 224)
    """Recorded image height and width after cropping and resizing."""

    enable_camera_player: bool = False
    task_description: str = ""

    def __post_init__(self) -> None:
        """Validate command limits before opening any hardware."""
        if not np.isfinite(self.step_frequency) or self.step_frequency <= 0:
            raise ValueError("UR step_frequency must be finite and positive.")
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)
        if (
            self.action_scale.shape != (3,)
            or not np.isfinite(self.action_scale).all()
            or np.any(self.action_scale <= 0)
        ):
            raise ValueError(
                "UR action_scale must contain three finite positive values."
            )
        for name in ("ee_pose_limit_min", "ee_pose_limit_max", "reset_ee_pose"):
            value = getattr(self, name)
            if value is None:
                if not self.is_dummy:
                    raise ValueError(f"Set UR {name} before opening real hardware.")
                continue
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError(f"UR {name} must contain six finite xyz/Euler values.")
            setattr(self, name, value)
        if self.ee_pose_limit_min is not None and self.ee_pose_limit_max is not None:
            if np.any(self.ee_pose_limit_min >= self.ee_pose_limit_max):
                raise ValueError("UR ee_pose_limit bounds require min < max.")
            if self.reset_ee_pose is not None and (
                np.any(self.reset_ee_pose < self.ee_pose_limit_min)
                or np.any(self.reset_ee_pose > self.ee_pose_limit_max)
            ):
                raise ValueError(
                    "UR reset_ee_pose must be inside ee_pose_limit bounds."
                )
        for name in (
            "reset_duration",
            "reset_timeout",
            "reset_position_tolerance",
            "reset_orientation_tolerance",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"UR {name} must be finite and positive.")
        if (
            not np.isfinite(self.binary_gripper_threshold)
            or not 0 < self.binary_gripper_threshold <= 1
        ):
            raise ValueError("UR binary_gripper_threshold must be in (0, 1].")
        if len(self.image_size) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in self.image_size
        ):
            raise ValueError(
                "UR image_size must contain positive integer height/width."
            )
        self.image_size = tuple(self.image_size)
        if self.camera_names is not None:
            self.camera_names = {
                str(serial): str(name) for serial, name in self.camera_names.items()
            }
        if self.camera_crop_regions is not None:
            self.camera_crop_regions = {
                str(serial): list(region)
                for serial, region in self.camera_crop_regions.items()
            }


class URCollectionEnv(gym.Env):
    """Apply Cartesian deltas while leaving episode scoring to the operator.

    The arm action contains normalized base-frame translation and xyz Euler
    increments. An optional seventh value is the established binary gripper
    signal: negative closes, positive opens. Reset interpolates to the configured
    TCP pose and leaves the gripper unchanged. Opening a fitted Robotiq connection
    runs its activation procedure, which may move it.
    """

    TELEOP = ("spacemouse",)
    TELEOP_DEFAULT = "spacemouse"
    ACTION_WRAPPERS = ("GripperCloseEnv",)
    TRANSFORMS = ()

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Optional[RobotInfo[UR5eConfig]] = None,
        env_idx: int = 0,
    ) -> None:
        self.config = URCollectionEnvConfig(**override_cfg)
        self._logger = get_logger()
        self.hardware = get_hardware_config(
            UR5eConfig, robot_info, is_dummy=self.config.is_dummy
        )
        self.env_idx = env_idx
        if env_idx < 0:
            raise ValueError("UR env_idx must be nonnegative.")
        self.node_rank = worker_info.cluster_node_rank if worker_info else None
        self.env_worker_rank = worker_info.rank if worker_info else 0
        self._num_steps = 0
        self._arm_was_driving = False
        self.robot: Optional[UR5eRobot] = None
        self._cameras: dict[str, BaseCamera] = {}
        self._end_effector: Optional[EndEffector] = None
        self._with_gripper = bool(self.hardware.gripper_type)
        self._last_reading: dict[str, Any] = {}
        if self._with_gripper:
            part = EndEffector.backend(self.hardware.gripper_type)
            if not part.is_gripper or part.is_hand:
                raise ValueError("UR collection supports a one-axis gripper.")
            if part.action_dim != 1 or part.state_dim != 1:
                raise ValueError(
                    "UR collection requires scalar gripper actions and state."
                )
        self._camera_infos = self._build_camera_infos()
        self._init_spaces()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)
        if self.config.is_dummy:
            return
        try:
            self._setup_hardware()
        except BaseException:
            self.close()
            raise

    @property
    def task_description(self) -> str:
        """Return the operator's description stored with each episode."""
        return self.config.task_description

    @property
    def num_steps(self) -> int:
        """Return the number of control steps in the current episode."""
        return self._num_steps

    def get_tcp_pose(self) -> np.ndarray:
        """Read the current base-frame pose without waiting for camera frames."""
        if self.config.is_dummy:
            return self.observation_space["state"]["tcp_pose"].sample()
        return np.asarray(self._arm.get_state().tcp_pose, dtype=np.float32).copy()

    def get_action_scale(self) -> np.ndarray:
        """Return the translation, rotation, and gripper action scales."""
        return self.config.action_scale.copy()

    def get_gripper_open(self) -> bool:
        """Return the current gripper latch state for SpaceMouse reset."""
        return self._end_effector is None or self._end_effector.is_open

    def _setup_hardware(self) -> None:
        controller_node = self.hardware.controller_node_rank
        self.robot = UR5eRobot.build(
            robot_ip=self.hardware.robot_ip,
            node_rank=self.node_rank,
            controller_node_rank=controller_node,
            worker_rank=self.env_worker_rank,
            env_idx=self.env_idx,
            backend=self.hardware.backend,
            frequency=self.hardware.frequency,
            lookahead_time=self.hardware.lookahead_time,
            gain=self.hardware.gain,
            gripper_type=self.hardware.gripper_type,
            gripper_connection=self.hardware.gripper_connection,
            end_effector_config=self.hardware.end_effector_config,
            cameras={info.name: info for info in self._camera_infos},
            camera_node_rank=self.hardware.camera_node_rank,
        )
        self.robot.connect()
        self._arm = self.robot.child("arm", BaseArm)
        if self._with_gripper:
            self._end_effector = self.robot.child("end_effector", EndEffector)
        self._cameras = {
            path.rsplit(".", 1)[-1]: camera
            for path, camera in self.robot.parts_of_type(Camera).items()
        }
        self._last_reading = self.robot.get_observation()

    def _init_spaces(self) -> None:
        width = 7 if self._with_gripper else 6
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(width,), dtype=np.float32)
        state: dict[str, gym.Space] = {
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        }
        if self._with_gripper:
            state["gripper_position"] = gym.spaces.Box(
                0.0, np.inf, shape=(1,), dtype=np.float32
            )
        spaces: dict[str, gym.Space] = {"state": gym.spaces.Dict(state)}
        if self._camera_infos:
            height, width = self.config.image_size
            spaces["frames"] = gym.spaces.Dict(
                {
                    info.name: gym.spaces.Box(
                        0, 255, shape=(height, width, 3), dtype=np.uint8
                    )
                    for info in self._camera_infos
                }
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Describe Cartesian arm increments and the optional gripper signal."""
        parts = (ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),)
        if self._with_gripper:
            parts += (ActionPart("end_effector", 1, ActionKind.GRIPPER),)
        return parts

    def get_hold_action(
        self, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return a zero delta that holds the measured pose after motion."""
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def _clip_position_to_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Apply the configured Cartesian and target-centered Euler limits."""
        pose = pose.copy()
        pose[:3] = np.clip(
            pose[:3],
            self.config.ee_pose_limit_min[:3],
            self.config.ee_pose_limit_max[:3],
        )
        euler = clip_euler_to_target_window(
            R.from_quat(pose[3:]).as_euler("xyz"),
            self.config.reset_ee_pose[3:],
            self.config.ee_pose_limit_min[3:],
            self.config.ee_pose_limit_max[3:],
        )
        pose[3:] = R.from_euler("xyz", euler).as_quat()
        return pose

    def _check_current_pose(self, pose: np.ndarray) -> None:
        """Require a measured pose inside the configured limits before moving."""
        clipped = self._clip_position_to_safety_box(pose)
        orientation_error = (
            R.from_quat(clipped[3:]) * R.from_quat(pose[3:]).inv()
        ).magnitude()
        if (
            not np.allclose(clipped[:3], pose[:3], atol=1e-7, rtol=0)
            or orientation_error > 1e-7
        ):
            raise ValueError(
                "Measured UR TCP is outside the configured pose limits; "
                "reposition it before sending motion commands."
            )

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply one finite normalized delta without automatic task termination."""
        started = time.monotonic()
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(
                f"UR action must have shape {self.action_space.shape} and finite values."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if not self.config.is_dummy:
            if self._with_gripper:
                self._gripper_action(float(action[6]) * self.config.action_scale[2])
            moving = bool(np.any(action[:6]))
            if moving or self._arm_was_driving:
                pose = self.get_tcp_pose().astype(np.float64)
                if moving:
                    self._check_current_pose(pose)
                    pose[:3] += action[:3] * self.config.action_scale[0]
                    pose[3:] = (
                        R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                        * R.from_quat(pose[3:])
                    ).as_quat()
                    pose = self._clip_position_to_safety_box(pose)
                self.robot.send_action({"arm": {"tcp_pose": pose.astype(np.float32)}})
            self._arm_was_driving = moving
        self._num_steps += 1
        time.sleep(
            max(0.0, 1.0 / self.config.step_frequency - (time.monotonic() - started))
        )
        observation = self._get_observation()
        return observation, 0.0, False, False, {}

    def _gripper_action(self, signal: float) -> None:
        if (
            signal <= -self.config.binary_gripper_threshold
            and self._end_effector.is_open
        ):
            self._end_effector.close()
        elif (
            signal >= self.config.binary_gripper_threshold
            and not self._end_effector.is_open
        ):
            self._end_effector.open()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return to the configured TCP pose and begin episode bookkeeping."""
        seed_sampled_spaces(seed, self.observation_space)
        self._num_steps = 0
        if not self.config.is_dummy:
            self.go_to_rest()
        return self._get_observation(), {}

    def go_to_rest(self) -> None:
        """Interpolate to the reset pose, then wait for measured arrival."""

        def reached() -> bool:
            measured = self.get_tcp_pose()
            angle = (
                R.from_quat(target[3:]) * R.from_quat(measured[3:]).inv()
            ).magnitude()
            return bool(
                np.linalg.norm(target[:3] - measured[:3])
                <= self.config.reset_position_tolerance
                and angle <= self.config.reset_orientation_tolerance
            )

        try:
            self._get_observation()
            start = self.get_tcp_pose().astype(np.float64)
            self._check_current_pose(start)
            target = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:]).as_quat(),
                ]
            )
            if not reached() or self._arm_was_driving:
                distance = np.max(np.abs(target[:3] - start[:3]))
                angle = (
                    R.from_quat(target[3:]) * R.from_quat(start[3:]).inv()
                ).magnitude()
                steps = max(
                    1,
                    int(
                        np.ceil(self.config.reset_duration * self.config.step_frequency)
                    ),
                    int(np.ceil(distance / self.config.action_scale[0])),
                    int(np.ceil(angle / self.config.action_scale[1])),
                )
                positions = np.linspace(start[:3], target[:3], steps + 1)
                orientations = quat_slerp(start[3:], target[3:], steps + 1)
                for position, orientation in zip(positions[1:], orientations[1:]):
                    pose = self._clip_position_to_safety_box(
                        np.concatenate([position, orientation])
                    )
                    self.robot.send_action(
                        {"arm": {"tcp_pose": pose.astype(np.float32)}}
                    )
                    self._arm_was_driving = True
                    time.sleep(1.0 / self.config.step_frequency)
            deadline = time.monotonic() + self.config.reset_timeout
            while not reached():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "UR reset did not reach the configured TCP pose."
                    )
                self._get_observation()
                if not reached():
                    time.sleep(1.0 / self.config.step_frequency)
            self._arm_was_driving = False
        except BaseException:
            try:
                pose = self._arm.get_state().tcp_pose.copy()
                self.robot.send_action({"arm": {"tcp_pose": pose}})
                self._arm_was_driving = False
            except Exception:
                self._logger.exception(
                    "Could not hold the UR TCP after a failed reset."
                )
            raise

    def _build_camera_infos(self) -> list[CameraInfo]:
        names = self.config.camera_names or {}
        crops = self.config.camera_crop_regions or {}
        infos = []
        for index, serial in enumerate(self.hardware.camera_serials or [], start=1):
            serial = str(serial)
            name = names.get(serial, f"wrist_{index}")
            if name in {"arm", "end_effector"} or any(
                info.name == name for info in infos
            ):
                raise ValueError(
                    f"UR camera name {name!r} must be unique and name a camera."
                )
            crop = crops.get(serial)
            if crop is not None:
                values = np.asarray(crop, dtype=np.float64)
                if (
                    values.shape != (4,)
                    or not np.isfinite(values).all()
                    or not 0 <= values[0] < values[2] <= 1
                    or not 0 <= values[1] < values[3] <= 1
                ):
                    raise ValueError(
                        "UR camera crop must be [top, left, bottom, right] in [0, 1]."
                    )
                crop = tuple(float(value) for value in values)
            infos.append(
                CameraInfo(
                    name=name,
                    serial_number=serial,
                    camera_type=self.hardware.camera_type,
                    resolution=self.hardware.camera_resolution,
                    fps=self.hardware.camera_fps,
                    enable_depth=False,
                    crop_region=crop,
                )
            )
        return infos

    def _get_observation(self) -> dict[str, Any]:
        if self.config.is_dummy:
            return self.observation_space.sample()
        reading = self.robot.get_observation()
        self._last_reading = reading
        observation = {
            "state": {
                "tcp_pose": np.asarray(reading["arm"]["tcp_pose"], dtype=np.float32)
            }
        }
        if self._with_gripper:
            observation["state"]["gripper_position"] = np.asarray(
                reading["end_effector"]["state"], dtype=np.float32
            )
        if self._cameras:
            frames = {}
            display = {}
            for name, camera in self._cameras.items():
                frame = np.asarray(reading[name]["frame"])
                height, width = frame.shape[:2]
                crop = camera.camera_info.crop_region
                if crop is not None:
                    top, left, bottom, right = crop
                    frame = frame[
                        int(top * height) : int(bottom * height),
                        int(left * width) : int(right * width),
                    ]
                else:
                    side = min(height, width)
                    top, left = (height - side) // 2, (width - side) // 2
                    frame = frame[top : top + side, left : left + side]
                if frame.size == 0:
                    raise ValueError(f"UR camera {name!r} crop contains no pixels.")
                resized = cv2.resize(frame, self.config.image_size[::-1])
                display[name] = resized
                frames[name] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.camera_player.put_frame(display)
            observation["frames"] = frames
        return copy.deepcopy(observation)

    def close(self) -> None:
        """Release the preview and every connection owned by the robot."""
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        self._cameras = {}
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
