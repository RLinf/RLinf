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
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.xsquare.turtle2_robot_state import Turtle2RobotState
from rlinf.scheduler import (
    Turtle2HWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger


@dataclass
class Turtle2RobotConfig:
    use_camera_ids: list[int] = field(default_factory=lambda: [2])  # [0, 1, 2]
    use_arm_ids: list[int] = field(default_factory=lambda: [1])  # [0, 1]

    is_dummy: bool = True
    use_dense_reward: bool = False
    step_frequency: float = 10.0  # Max number of steps per second
    smooth_frequency: int = 50  # Frequency for smooth controller

    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.15, 0.0, 1, 0.0]]
        )
    )
    reset_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.3, 0, 0.0, 0.2, 0, 0], [0.1, 0, 0.1, 0, 0.8, 0.0]]
        )
    )

    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = False

    random_xy_range: float = 0.05
    random_rz_range: float = np.pi / 10

    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), -np.inf)
    )
    ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), np.inf)
    )
    gripper_width_limit_min: float = 0.0
    gripper_width_limit_max: float = 5.0
    enforce_gripper_close: bool = True
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None


class Turtle2Env(gym.Env):
    """Turtle2 robot arm environment."""

    def __init__(
        self,
        config: Turtle2RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[Turtle2HWInfo],
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

        assert len(self.config.use_arm_ids) > 0 and len(self.config.use_arm_ids) <= 2, (
            "please choose arm IDs from [0, 1]."
        )
        assert (
            len(self.config.use_camera_ids) > 0 and len(self.config.use_camera_ids) <= 3
        ), "please choose camera IDs from [0, 1, 2]."
        self._turtle2_state = Turtle2RobotState()
        self._num_steps = 0

        if not self.config.is_dummy:
            self._setup_hardware()

        # Init action and observation spaces
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the first frame
        self._reset_arms()
        self._turtle2_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._check_cameras()
        # Video player for displaying camera frames

    def _setup_hardware(self):
        from .turtle2_smooth_controller import Turtle2SmoothController

        assert self.env_idx >= 0, "env_idx must be set for Turtle2Env."

        # Launch Turtle controller
        self._controller = Turtle2SmoothController.launch_controller(
            freq=self.config.smooth_frequency,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space1 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[0, :3].flatten(),
            high=self.config.ee_pose_limit_max[0, :3].flatten(),
            dtype=np.float64,
        )
        self._rpy_safe_space1 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[0, 3:].flatten(),
            high=self.config.ee_pose_limit_max[0, 3:].flatten(),
            dtype=np.float64,
        )
        self._xyz_safe_space2 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[1, :3].flatten(),
            high=self.config.ee_pose_limit_max[1, :3].flatten(),
            dtype=np.float64,
        )
        self._rpy_safe_space2 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[1, 3:].flatten(),
            high=self.config.ee_pose_limit_max[1, 3:].flatten(),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            np.ones((len(self.config.use_arm_ids) * 7), dtype=np.float32) * -1,
            np.ones((len(self.config.use_arm_ids) * 7), dtype=np.float32),
        )

        obs_dim_per_arm = 7  # xyz(3) + quat(4)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(len(self.config.use_arm_ids) * obs_dim_per_arm,),
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.use_camera_ids))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _reset_arms(self):
        if self.config.is_dummy:
            return

        self._logger.info("pre-reset")
        self._controller.move_arm(
            [0.2, 0, 0.1, 0, 0, 0, 0], [0.2, 0, 0.1, 0, 0, 0, 0]
        ).wait()
        time.sleep(2.0)

        if self.config.enable_random_reset:
            random_xy1 = np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            random_xy2 = np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            random_euler1 = np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range, (3,)
            )
            random_euler2 = np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range, (3,)
            )
        else:
            random_xy1 = np.zeros(2)
            random_xy2 = np.zeros(2)
            random_euler1 = np.zeros(3)
            random_euler2 = np.zeros(3)

        if 0 in self.config.use_arm_ids:
            left_arm_reset_pose = self.config.reset_ee_pose[0].copy()
            left_arm_reset_pose[:2] += random_xy1
            left_arm_reset_pose[3:6] += random_euler1
            left_arm_reset_pose = left_arm_reset_pose.tolist()
            left_arm_reset_pose.append(0.0)
        else:
            left_arm_reset_pose = [0, 0, 0, 0, 0, 0, 0]
        if 1 in self.config.use_arm_ids:
            right_arm_reset_pose = self.config.reset_ee_pose[1].copy()
            right_arm_reset_pose[:2] += random_xy2
            right_arm_reset_pose[3:6] += random_euler2
            right_arm_reset_pose = right_arm_reset_pose.tolist()
            right_arm_reset_pose.append(0.0)
        else:
            right_arm_reset_pose = [0, 0, 0, 0, 0, 0, 0]

        self._logger.info("going to reset:", left_arm_reset_pose, right_arm_reset_pose)

        self._controller.move_arm(left_arm_reset_pose, right_arm_reset_pose).wait()

        reach = False
        start_time = time.time()
        while not reach:
            state = self._controller.get_state().wait()[0]
            left_pos = state.follow1_pos
            right_pos = state.follow2_pos
            left_reach = (
                np.linalg.norm(left_pos[:6] - np.array(left_arm_reset_pose)[:6]) < 0.04
                if 0 in self.config.use_arm_ids
                else True
            )
            right_reach = (
                np.linalg.norm(right_pos[:6] - np.array(right_arm_reset_pose)[:6])
                < 0.04
                if 1 in self.config.use_arm_ids
                else True
            )
            reach = left_reach and right_reach
            if time.time() - start_time > 10.0:
                raise ValueError(
                    "Reset arms timeout, lefterr:",
                    np.linalg.norm(left_pos[:6] - np.array(left_arm_reset_pose)[:6]),
                    " ; righterr:",
                    np.linalg.norm(right_pos[:6] - np.array(right_arm_reset_pose)[:6]),
                )

            time.sleep(0.1)
        time.sleep(0.5)
        return

    def _check_cameras(self):
        if self.config.is_dummy:
            return

        cam1_ok, cam2_ok, cam3_ok = self._controller.check_cams().wait()[0]
        if 0 in self.config.use_camera_ids and not cam1_ok:
            raise ValueError("Camera 1 not available.")
        if 1 in self.config.use_camera_ids and not cam2_ok:
            raise ValueError("Camera 2 not available.")
        if 2 in self.config.use_camera_ids and not cam3_ok:
            raise ValueError("Camera 3 not available.")

    def reset(self, *, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        # Reset
        self._reset_arms()
        self._num_steps = 0
        self._turtle2_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        # save if debug
        # for key in observation["frames"].keys():
        #     img = Image.fromarray(observation["frames"][key])
        #     img.save(f'{key}.jpg')

        return observation, {}

    def transform_action_ee_to_base(self, action):
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        action (np.ndarray): The action to take, in shape (7, ) or (14, ).
        """
        assert action.shape == (len(self.config.use_arm_ids) * 7,), (
            f"Action shape must be {(len(self.config.use_arm_ids) * 7,)}, but got {action.shape}."
        )

        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # deal with dual arms (xyz)
        action = action.reshape(-1, 7)
        xyz_delta = action[:, :3]

        # self._turtle2_state = self._controller.get_state().wait()[0]
        next_position1 = self._turtle2_state.follow1_pos.copy()
        next_position2 = self._turtle2_state.follow2_pos.copy()

        if 0 in self.config.use_arm_ids:
            next_position1[:3] = (
                next_position1[:3] + xyz_delta[0] * self.config.action_scale[0]
            )
        if 1 in self.config.use_arm_ids:
            next_position2[:3] = (
                next_position2[:3] + xyz_delta[-1] * self.config.action_scale[0]
            )

        # deal with dual arms (rpy)
        if 0 in self.config.use_arm_ids:
            next_position1[3:6] = (
                next_position1[3:6] + action[0, 3:6] * self.config.action_scale[1]
            )
        if 1 in self.config.use_arm_ids:
            next_position2[3:6] = (
                next_position2[3:6] + action[-1, 3:6] * self.config.action_scale[1]
            )

        if self.config.enforce_gripper_close:
            next_position1[6] = self.config.gripper_width_limit_min
            next_position2[6] = self.config.gripper_width_limit_min
        else:
            if 0 in self.config.use_arm_ids:
                next_position1[6] = action[0, 6]
            if 1 in self.config.use_arm_ids:
                next_position2[6] = action[-1, 6]

        # clip to safety box
        next_position = self._clip_position_to_safety_box(
            np.stack([next_position1, next_position2])
        )
        next_position1 = next_position[0]
        next_position2 = next_position[1]

        if not self.config.is_dummy:
            self._controller.move_arm(
                next_position1.tolist(), next_position2.tolist()
            ).wait()
        else:
            pass

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._turtle2_state = self._controller.get_state().wait()[0]
        else:
            self._turtle2_state = self._turtle2_state
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = reward == 1
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray],
    ) -> float:
        """Compute the reward for the current observation, namely the robot state and camera frames.

        Args:
            observation (Dict[str, np.ndarray]): The current observation from the environment.
        """
        if not self.config.is_dummy:
            # Convert orientation to euler angles
            position1 = self._turtle2_state.follow1_pos[0:6]
            position2 = self._turtle2_state.follow2_pos[0:6]
            delta1 = np.abs(position1 - self.config.target_ee_pose[0, 0:6])
            delta2 = np.abs(position2 - self.config.target_ee_pose[1, 0:6])

            success1 = (
                np.all(delta1 <= self.config.reward_threshold)
                if 0 in self.config.use_arm_ids
                else True
            )
            success2 = (
                np.all(delta2 <= self.config.reward_threshold)
                if 1 in self.config.use_arm_ids
                else True
            )
            is_success = success1 and success2

            if is_success:
                reward = 1.0
            else:
                if self.config.use_dense_reward:
                    delta1_sq = (
                        np.sum(np.square(delta1[0:6]))
                        if 0 in self.config.use_arm_ids
                        else 0.0
                    )
                    delta2_sq = (
                        np.sum(np.square(delta2[0:6]))
                        if 1 in self.config.use_arm_ids
                        else 0.0
                    )
                    reward = np.exp(-200 * (delta1_sq + delta2_sq))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria."
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            return reward
        else:
            return 0.0

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
        return resized_frame

    # Robot actions

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[0, 0:3] = np.clip(
            position[0, 0:3], self._xyz_safe_space1.low, self._xyz_safe_space1.high
        )
        position[0, 3:6] = np.clip(
            position[0, 3:6], self._rpy_safe_space1.low, self._rpy_safe_space1.high
        )
        position[0, 6] = np.clip(
            position[0, 6],
            self.config.gripper_width_limit_min,
            self.config.gripper_width_limit_max,
        )
        position[1, 0:3] = np.clip(
            position[1, 0:3], self._xyz_safe_space2.low, self._xyz_safe_space2.high
        )
        position[1, 3:6] = np.clip(
            position[1, 3:6], self._rpy_safe_space2.low, self._rpy_safe_space2.high
        )
        position[1, 6] = np.clip(
            position[1, 6],
            self.config.gripper_width_limit_min,
            self.config.gripper_width_limit_max,
        )

        position = position.reshape(2, -1)
        return position

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._controller.get_cams(self.config.use_camera_ids).wait()[0]
            assert len(frames) == len(self.config.use_camera_ids), "get frames failed."
            for i in range(len(frames)):
                frames[i] = self._crop_frame(frames[i], (128, 128))
            tcp_pose = []
            if 0 in self.config.use_arm_ids:
                tmp = np.zeros(7)
                tmp[0:3] = self._turtle2_state.follow1_pos[0:3]
                r1 = R.from_euler("xyz", self._turtle2_state.follow1_pos[3:6])
                tmp[3:7] = r1.as_quat()
                tcp_pose.append(tmp.copy())
            if 1 in self.config.use_arm_ids:
                tmp = np.zeros(7)
                tmp[0:3] = self._turtle2_state.follow2_pos[0:3]
                r2 = R.from_euler("xyz", self._turtle2_state.follow2_pos[3:6])
                tmp[3:7] = r2.as_quat()
                tcp_pose.append(tmp.copy())
            tcp_pose = np.concatenate(tcp_pose, axis=0)
            state = {
                "tcp_pose": tcp_pose,
            }
            frames_dict = {}
            for k in range(len(self.config.use_camera_ids)):
                frames_dict[f"wrist_{k + 1}"] = frames[k]

            observation = {
                "state": state,
                "frames": frames_dict,
            }
            return copy.deepcopy(observation)
        else:
            obs = self._base_observation_space.sample()
            return obs
