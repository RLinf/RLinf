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
from itertools import cycle
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import Camera, CameraInfo
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import (
    FrankaHWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger

from .franka_robot_state import FrankaRobotState
from .utils import construct_adjoint_matrix, construct_homogeneous_matrix, quat_slerp


@dataclass
class FrankaRobotConfig:
    robot_ip: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    enable_camera_player: bool = True

    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0  # Max number of steps per second

    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0, 0, 0, -1.9, -0, 2, 0]
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = False

    random_xy_range: float = 0.0
    random_rz_range: float = 0.0  # np.pi / 6

    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(6))
    compliance_param: dict[str, float] = field(default_factory=dict)
    precision_param: dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    joint_reset_cycle: int = 20000  # Number of resets before resetting joints


class FrankaEnv(gym.Env):
    """Franka robot arm environment."""

    def __init__(
        self,
        config: FrankaRobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[FrankaHWInfo],
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

        self._franka_state = FrankaRobotState()
        if not self.config.is_dummy:
            self._reset_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:].copy()).as_quat(),
                ]
            ).copy()
        else:
            self._reset_pose = np.zeros(7)
        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)  # Initialize the cycle

        if not self.config.is_dummy:
            self._setup_hardware()

        # Init action and observation spaces
        assert (
            self.config.camera_serials is not None
            and len(self.config.camera_serials) > 0
        ), "At least one camera serial must be provided for FrankaEnv."
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the robot to be ready
        start_time = time.time()
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
                )

        self._interpolate_move(self._reset_pose)
        time.sleep(1.0)
        self._franka_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._open_cameras()
        # Video player for displaying camera frames
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _setup_hardware(self):
        from .franka_controller import FrankaController

        assert self.env_idx >= 0, "env_idx must be set for FrankaEnv."

        # Setup Franka IP and camera serials
        assert isinstance(self.hardware_info, FrankaHWInfo), (
            f"hardware_info must be FrankaHWInfo, but got {type(self.hardware_info)}."
        )
        # Only set robot_ip and camera_serials if they are not provided in config
        if self.config.robot_ip is None:
            self.config.robot_ip = self.hardware_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials

        # Launch Franka controller
        self._controller = FrankaController.launch_controller(
            robot_ip=self.config.robot_ip,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )

    def transform_action_ee_to_base(self, action):
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        action (np.ndarray): The action to take, which is a 7D vector representing the desired end-effector position and orientation,
        as well as the gripper action. The first 3 elements correspond to the delta in x, y, z position, the next 3 elements correspond to the delta in rx, ry, rz orientation (in euler angles), and the last element corresponds to the gripper action.
        [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper_action]
        """
        start_time = time.time()

        # if self.use_rel_frame:
        #     action = self.transform_action_ee_to_base(action)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.next_position = self._franka_state.tcp_pose.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self.config.action_scale[0]
        )

        if not self.config.is_dummy:
            self.next_position[3:] = (
                R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                * R.from_quat(self._franka_state.tcp_pose[3:].copy())
            ).as_quat()

            gripper_action = action[6] * self.config.action_scale[2]

            is_gripper_action_effective = self._gripper_action(gripper_action)
            self._move_action(self._clip_position_to_safety_box(self.next_position))
        else:
            is_gripper_action_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        else:
            self._franka_state = self._franka_state
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_action_effective)
        terminated = reward == 1
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray | FrankaRobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute the reward for the current observation, namely the robot state and camera frames.

        Args:
            observation (Dict[str, np.ndarray]): The current observation from the environment.
            is_gripper_action_effective (bool): Whether the gripper action was effective (i.e., the gripper state changed).
        """
        if not self.config.is_dummy:
            # Convert orientation to euler angles
            euler_angles = np.abs(
                R.from_quat(self._franka_state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._franka_state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)
            is_success = np.all(target_delta[:3] <= self.config.reward_threshold[:3])
            if is_success:
                reward = 1.0
            else:
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria. Target delta: {target_delta}, "
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty

            return reward
        else:
            return 0.0

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        # Reset joint
        joint_reset_cycle = next(self._joint_reset_cycle)
        joint_reset = False
        if joint_reset_cycle == 0:
            self._logger.info(
                f"Number of resets reached {self.config.joint_reset_cycle}, resetting joints to initial position."
            )
            joint_reset = True

        self.go_to_rest(joint_reset)

        self._clear_error()
        self._num_steps = 0
        self._franka_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()

        return observation, {}

    def go_to_rest(self, joint_reset=False):
        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos).wait()
            time.sleep(0.5)

        # Reset arm
        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose[:2] += np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            euler_random = self.config.target_ee_pose[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range
            )
            reset_pose[3:] = R.from_euler("xyz", euler_random).as_quat()
        else:
            reset_pose = self._reset_pose.copy()

        self._franka_state = self._controller.get_state().wait()[0]
        cnt = 0
        while not np.allclose(self._franka_state.tcp_pose[:3], reset_pose[:3], 0.02):
            cnt += 1
            self._interpolate_move(reset_pose)
            self._franka_state = self._controller.get_state().wait()[0]
            if cnt > 2:
                break

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        obs_tcp_pose_dim = 7
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(obs_tcp_pose_dim,)
                        ),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
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
                        for k in range(len(self.config.camera_serials))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _open_cameras(self):
        self._cameras: list[Camera] = []
        if self.config.camera_serials is None:
            return
        camera_infos = [
            CameraInfo(name=f"wrist_{i + 1}", serial_number=n)
            for i, n in enumerate(self.config.camera_serials)
        ]
        for info in camera_infos:
            camera = Camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
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
        return cropped_frame, resized_frame

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get frames from all cameras."""
        frames = {}
        display_frames = {}
        for camera in self._cameras:
            try:
                frame = camera.get_frame()
                reshape_size = self.observation_space["frames"][
                    camera._camera_info.name
                ].shape[:2][::-1]
                cropped_frame, resized_frame = self._crop_frame(frame, reshape_size)
                frames[camera._camera_info.name] = resized_frame[
                    ..., ::-1
                ]  # Convert RGB to BGR
                display_frames[camera._camera_info.name] = (
                    resized_frame  # Original RGB for display
                )
                display_frames[f"{camera._camera_info.name}_full"] = (
                    cropped_frame  # Non-resized version
                )
            except queue.Empty:
                self._logger.warning(
                    f"Camera {camera._camera_info.name} is not producing frames. Wait 5 seconds and try again."
                )
                time.sleep(5)
                camera.close()
                self._open_cameras()
                return self._get_camera_frames()

        self.camera_player.put_frame(display_frames)
        return frames

    # Robot actions

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")

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
        position[3:] = R.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        if is_binary:
            if (
                position <= -self.config.binary_gripper_threshold
                and self._franka_state.gripper_open
            ):
                # Close gripper
                self._controller.close_gripper().wait()
                time.sleep(0.6)
                return True
            elif (
                position >= self.config.binary_gripper_threshold
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

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5):
        num_steps = int(timeout * self.config.step_frequency)
        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]
        pos_path = np.linspace(
            self._franka_state.tcp_pose[:3], pose[:3], int(num_steps) + 1
        )
        quat_path = quat_slerp(
            self._franka_state.tcp_pose[3:], pose[3:], int(num_steps) + 1
        )

        for pos, quat in zip(pos_path[1:], quat_path[1:]):
            pose = np.concatenate([pos, quat])
            self._move_action(pose.astype(np.float32))
            time.sleep(1.0 / self.config.step_frequency)

        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]

    def _move_action(self, position: np.ndarray):
        if not self.config.is_dummy:
            self._clear_error()
            self._controller.move_arm(position.astype(np.float32)).wait()
        else:
            print(f"Executing dummy action towards {position=}.")

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "tcp_pose": self._franka_state.tcp_pose,
                "tcp_vel": self._franka_state.tcp_vel,
                "gripper_position": np.array(
                    [
                        self._franka_state.gripper_position,
                    ]
                ),
                "tcp_force": self._franka_state.tcp_force,
                "tcp_torque": self._franka_state.tcp_torque,
            }
            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            obs = self._base_observation_space.sample()
            return obs

    def transform_obs_base_to_ee(self, state):
        self.adjoint_matrix = construct_adjoint_matrix(self._franka_state.tcp_pose)
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)

        state["tcp_vel"] = adjoint_inv @ state["tcp_vel"]

        T_b_o = construct_homogeneous_matrix(self._franka_state.tcp_pose)
        T_r_o = self.T_b_r_inv @ T_b_o

        p_r_o = T_r_o[:3, 3]
        quat_r_o = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
        state["tcp_pose"] = np.concatenate([p_r_o, quat_r_o], axis=0)

        return state

    @property
    def target_ee_pose(self):
        tgt = np.concatenate(
            [
                self.config.target_ee_pose[:3],
                R.from_euler("xyz", self.config.target_ee_pose[3:].copy()).as_quat(),
            ]
        ).copy()
        return tgt
