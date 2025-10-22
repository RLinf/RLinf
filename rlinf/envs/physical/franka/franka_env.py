# Copyright 2025 The RLinf Authors.
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
from typing import Dict, List, Tuple

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.physical.common.camera import Camera, CameraInfo
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import euler_2_quat, quat_2_euler

from .franka_controller import FrankaController, FrankaRobotState


@dataclass
class FrankaRobotConfig:
    robot_ip: str
    cameras: List[CameraInfo] = field(
        default_factory=lambda: [
            CameraInfo(name="wrist_1", serial_number="serial_1"),
            CameraInfo(name="wrist_2", serial_number="serial_2"),
        ]
    )
    step_frequency: float = 10.0  # Max number of steps per second
    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    reset_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Algorithm parameters
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = False
    random_xy_range: float = 0.0
    random_rz_range: float = 0.0
    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    position_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    position_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(6))
    compliance_param: Dict[str, float] = field(default_factory=dict)
    precision_param: Dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1


class FrankaEnv(gym.Env):
    """Franka robot arm environment. This is adapted from SERL's FrankaEnv."""

    def __init__(self, config: FrankaRobotConfig):
        """Initialize the Franka robot arm environment.

        Args:
            robot_ip (str): The IP address of the robot arm.
            gripper_ip (str): The IP address of the gripper.
        """
        self._logger = get_logger()
        self._config = config
        self._franka_state = FrankaRobotState()
        self._franka_state.arm_position = np.concatenate(
            [
                self._config.reset_position[:3],
                euler_2_quat(self._config.reset_position[3:]),
            ]
        )
        self._num_steps = 0

        # Launch Franka controller on the same node as the env
        # TODO: Support launching on different nodes
        cluster = Cluster()
        self._node_id = Worker.current_worker._node_id
        placement = NodePlacementStrategy(node_ids=[self._node_id])
        self._controller = FrankaController.create_group(self._config.robot_ip).launch(
            cluster=cluster, placement_strategy=placement
        )
        start_time = time.time()
        # Wait for the robot to be ready
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
                )

        # Init action and observation spaces
        self._init_action_obs_spaces()

        # Init cameras
        if self._config.cameras is not None:
            assert len(self._config.cameras) == 2, (
                "Currently FrankaEnv only support 2 cameras from wrist_1 and wrist_2."
            )
            self._cameras: List[Camera] = []
            self._open_cameras(self._config.cameras)

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        action (np.ndarray): The action to take, which is a 7D vector representing the desired end-effector position and orientation,
        as well as the gripper action. The first 3 elements correspond to the delta in x, y, z position, the next 3 elements correspond to the delta in rx, ry, rz orientation (in euler angles), and the last element corresponds to the gripper action.
        [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper_action]
        """
        start_time = time.time()
        action = np.clip(action, self._action_space.low, self._action_space.high)
        xyz_delta = action[:3]

        self.next_position = self._franka_state.arm_position.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self._config.action_scale[0]
        )

        self.next_position[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self._config.action_scale[1])
            * Rotation.from_quat(self._franka_state.arm_position[3:])
        ).as_quat()

        gripper_action = action[6] * self._config.action_scale[2]

        is_gripper_action_effective = self._gripper_action(gripper_action)
        self._move_action(self._clip_position_to_safety_box(self.next_position))

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self._config.step_frequency) - step_time))

        self._franka_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_action_effective)
        done = self._num_steps >= self._config.max_num_steps or reward == 1
        return observation, reward, done, False, {}

    def _calc_step_reward(
        self,
        observation: Dict[str, np.ndarray | FrankaRobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute the reward for the current observation, namely the robot state and camera frames.

        Args:
            observation (Dict[str, np.ndarray]): The current observation from the environment.
            is_gripper_action_effective (bool): Whether the gripper action was effective (i.e., the gripper state changed).
        """
        # Convert orientation to euler angles
        franka_state: FrankaRobotState = observation["state"]
        euler_angles = np.abs(quat_2_euler(franka_state.arm_position[3:]))
        position = np.hstack([franka_state.arm_position[:3], euler_angles])
        target_delta = np.abs(position - self._config.target_position)
        if np.all(target_delta <= self._config.reward_threshold):
            reward = 1
        else:
            self._logger.debug(
                f"Does not meet reward criteria. Target delta: {target_delta}, Reward threshold: {self._config.reward_threshold}"
            )
            reward = 0

        if self._config.enable_gripper_penalty and is_gripper_action_effective:
            reward -= self._config.gripper_penalty

        return reward

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space = gym.spaces.Box(
            low=self._config.position_limit_min[:3],
            high=self._config.position_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self._config.position_limit_min[3:],
            high=self._config.position_limit_max[3:],
            dtype=np.float64,
        )
        self._action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self._obs_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "arm_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "arm_velocity": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                        "arm_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "arm_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "wrist_1": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        "wrist_2": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )

    def _open_cameras(self, camera_info: List[CameraInfo]):
        for info in camera_info:
            camera = Camera(info)
            camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: Tuple[int, int]
    ) -> np.ndarray:
        """Crop the frame to the desired resolution."""
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return resized_frame

    def _get_camera_frames(self) -> Dict[str, np.ndarray]:
        """Get frames from all cameras."""
        if not self._config.cameras:
            # Return empty frames if no cameras are configured
            return {
                "wrist_1": np.zeros((128, 128, 3), dtype=np.uint8),
                "wrist_2": np.zeros((128, 128, 3), dtype=np.uint8),
            }
        frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self._obs_space["frames"][
                    camera._camera_info.name
                ].shape[:2][::-1]
                cropped_frame = self._crop_frame(frame, reshape_size)
                frames[camera._camera_info.name] = cropped_frame[
                    ..., ::-1
                ]  # Convert RGB to BGR
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera._camera_info.name} is not producing frames. Wait 5 seconds and try again."
                )
                time.sleep(5)
                camera.close()
                self._open_cameras(self._config.cameras)
                return self._get_camera_frames()
        return frames

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = Rotation.from_quat(position[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self._rpy_safe_space.low[0],
                self._rpy_safe_space.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self._rpy_safe_space.low[1:], self._rpy_safe_space.high[1:]
        )
        position[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        if is_binary:
            print(
                f"Gripper action: {position}, {self._config.binary_gripper_threshold}, {self._franka_state.gripper_open}"
            )
            if (
                position <= -self._config.binary_gripper_threshold
                and self._franka_state.gripper_open
            ):
                # Close gripper
                self._controller.close_gripper().wait()
                return True
            elif (
                position >= self._config.binary_gripper_threshold
                and not self._franka_state.gripper_open
            ):
                # Open gripper
                self._controller.open_gripper().wait()
                time.sleep(0.6)
                return True
            else:  # No change
                return False
        else:
            raise NotImplementedError("Non-binary gripper action not implemented.")

    def _move_action(self, position: np.ndarray):
        self._clear_error()
        self._controller.move_arm(position.astype(np.float32)).wait()

    def _get_observation(self) -> Dict:
        frames = self._get_camera_frames()
        observation = {
            "state": self._franka_state,
            "frames": frames,
        }
        return copy.deepcopy(observation)
