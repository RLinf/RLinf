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

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics import (
    Camera,
    Robot,
    RobotInfo,
    Turtle2Config,
    Turtle2Robot,
)
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger


@dataclass
class Turtle2RobotConfig:
    use_camera_ids: list[int] = field(default_factory=lambda: [2])
    use_arm_ids: list[int] = field(default_factory=lambda: [1])

    is_dummy: bool = True
    use_dense_reward: bool = False
    step_frequency: float = 10.0  # Maximum environment steps per second.
    smooth_frequency: int = 50  # Smoothing-controller frequency.

    # Poses use xyz position and xyz Euler orientation; observations use quaternions.
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
    )  # Translation, orientation, and gripper scales.
    enable_random_reset: bool = False

    random_xy_range: float = 0.05
    random_rz_range: float = np.pi / 10

    # Cartesian limits use xyz position followed by xyz Euler orientation.
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
    """Gymnasium environment wrapping the Turtle2 dual-arm robot.

    Supports single- and dual-arm control with optional camera observations,
    dense/sparse rewards, safety-box clipping, and a dummy mode for offline use.
    """

    TELEOP = ("spacemouse", "gello", "pico")
    TELEOP_DEFAULT = "spacemouse"
    ACTION_WRAPPERS = ("GripperCloseEnv",)
    TRANSFORMS = ("RelativeFrame", "Quat2EulerWrapper")

    def __init__(
        self,
        config: Turtle2RobotConfig,
        worker_info: Optional[WorkerInfo],
        robot_info: Optional[RobotInfo[Turtle2Config]],
        env_idx: int,
    ) -> None:
        """Initialize a Turtle2 environment.

        Args:
            config: Robot and environment configuration.
            worker_info: Scheduler worker info used to resolve node/worker rank.
            robot_info: Hardware descriptor for the Turtle2 platform.
            env_idx: Index of this environment instance within its worker.
        """
        self._logger = get_logger()
        self.config = config
        self.robot_info = robot_info
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
        # A dummy env reads no hardware, so start from a zero pose the
        # same shape the robot would report.
        self._reading: dict[str, Any] = {
            side: {"arm": {"tcp_pose": np.zeros(7)}} for side in ("left", "right")
        }
        self._num_steps = 0
        self.robot: Robot | None = None

        if not self.config.is_dummy:
            self._setup_hardware()

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Reset the arms before reading the initial state.
        self._reset_arms()
        self._reading = self.robot.get_observation()

        self._check_cameras()

    def _setup_hardware(self) -> None:
        assert self.env_idx >= 0, "env_idx must be set for Turtle2Env."

        self.robot = Turtle2Robot.build(
            frequency=self.config.smooth_frequency,
            camera_ids=self.config.use_camera_ids,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )
        self.robot.connect()

    def close(self) -> None:
        """Disconnect the composed Turtle2 runtime."""
        if self.robot is not None:
            self.robot.disconnect()
        super().close()

    def _init_action_obs_spaces(self) -> None:
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

        obs_dim_per_arm = 7  # xyz position and quaternion orientation.
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

    def _reset_arms(self) -> None:
        """Move both arms to their reset poses, blocking until they arrive.

        Does nothing in dummy mode.
        """
        if self.config.is_dummy:
            return

        self._logger.info("pre-reset")
        self._command_arms(
            np.array([0.2, 0, 0.1, 0, 0, 0, 0]), np.array([0.2, 0, 0.1, 0, 0, 0, 0])
        )
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

        self._logger.info(
            "Going to reset: left=%s, right=%s",
            repr(left_arm_reset_pose),
            repr(right_arm_reset_pose),
        )

        self._command_arms(
            np.asarray(left_arm_reset_pose), np.asarray(right_arm_reset_pose)
        )

        reach = False
        start_time = time.time()
        while not reach:
            reading = self.robot.get_observation()
            left_pos = reading["left"]["arm"]["tcp_pose"]
            right_pos = reading["right"]["arm"]["tcp_pose"]
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
                left_err = np.linalg.norm(
                    left_pos[:6] - np.array(left_arm_reset_pose)[:6]
                )
                right_err = np.linalg.norm(
                    right_pos[:6] - np.array(right_arm_reset_pose)[:6]
                )
                raise ValueError(
                    f"Reset arms timeout: left_err={left_err:.6f}, right_err={right_err:.6f}"
                )

            time.sleep(0.1)
        time.sleep(0.5)
        return

    def _camera_parts(self) -> list[Camera]:
        """The robot's cameras, in the order their ids were configured."""
        cameras = self.robot.parts_of_type(Camera)
        return [
            cameras[f"wrist_{index + 1}"]
            for index in range(len(self.config.use_camera_ids))
            if f"wrist_{index + 1}" in cameras
        ]

    def _command_arms(self, left: np.ndarray, right: np.ndarray) -> None:
        """Send both arm poses and gripper widths as one action.

        Each pose is ``[x, y, z, rx, ry, rz, gripper_width]``: the first six
        drive the arm and the last its gripper, which rides beneath it.
        """
        self.robot.send_action(
            {
                side: {
                    "arm": {"tcp_pose": np.asarray(pose[:6], dtype=float)},
                    "gripper": {"target": np.asarray([pose[6]], dtype=float)},
                }
                for side, pose in (("left", left), ("right", right))
            }
        )

    def _check_cameras(self) -> None:
        if self.config.is_dummy:
            return

        ready = [camera.is_ready() for camera in self._camera_parts()]
        cam1_ok, cam2_ok, cam3_ok = (ready + [False] * 3)[:3]
        if 0 in self.config.use_camera_ids and not cam1_ok:
            raise ValueError("Camera 1 not available.")
        if 1 in self.config.use_camera_ids and not cam2_ok:
            raise ValueError("Camera 2 not available.")
        if 2 in self.config.use_camera_ids and not cam3_ok:
            raise ValueError("Camera 3 not available.")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        # A run with no hardware samples this space instead of reading,
        # so seeding it is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._base_observation_space)
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        self._reset_arms()
        self._num_steps = 0
        self._reading = self.robot.get_observation()
        observation = self._get_observation()
        return observation, {}

    def transform_action_ee_to_base(self, action: np.ndarray) -> np.ndarray:
        """Transform action from end-effector frame to base frame.

        Args:
            action: Action array in end-effector coordinates.

        Returns:
            Action array in base frame coordinates.
        """
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return Cartesian-delta and gripper parts for each active arm."""
        from rlinf.envs.real.wrappers.teleop.layout import mirrored

        per_arm = (
            ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),
            ActionPart("end_effector", 1, ActionKind.GRIPPER),
        )
        if len(self.config.use_arm_ids) == 1:
            return per_arm
        return mirrored(per_arm, ("left", "right"))

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Delta end-effector action of shape ``(7,)`` for single arm
                or ``(14,)`` for dual arm (xyz, rpy, gripper per arm).

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        assert action.shape == (len(self.config.use_arm_ids) * 7,), (
            f"Action shape must be {(len(self.config.use_arm_ids) * 7,)}, but got {action.shape}."
        )

        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply translation deltas to each active arm.
        action = action.reshape(-1, 7)
        xyz_delta = action[:, :3]

        next_position1 = self._reading["left"]["arm"]["tcp_pose"].copy()
        next_position2 = self._reading["right"]["arm"]["tcp_pose"].copy()

        if 0 in self.config.use_arm_ids:
            next_position1[:3] = (
                next_position1[:3] + xyz_delta[0] * self.config.action_scale[0]
            )
        if 1 in self.config.use_arm_ids:
            next_position2[:3] = (
                next_position2[:3] + xyz_delta[-1] * self.config.action_scale[0]
            )

        # Apply Euler-angle deltas to each active arm.
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

        # Enforce Cartesian and gripper limits before dispatch.
        next_position = self._clip_position_to_safety_box(
            np.stack([next_position1, next_position2])
        )
        next_position1 = next_position[0]
        next_position2 = next_position[1]

        if not self.config.is_dummy:
            self._command_arms(next_position1, next_position2)
        else:
            pass

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._reading = self.robot.get_observation()
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = reward == 1
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray],
    ) -> float:
        """Compute the per-step reward from the current robot state.

        Args:
            observation: Current observation dict (unused directly; reward is
                derived from internal robot state).

        Returns:
            ``1.0`` on success, a dense exponential reward when
            ``use_dense_reward`` is set, or ``0.0`` otherwise.
        """
        if not self.config.is_dummy:
            position1 = self._reading["left"]["arm"]["tcp_pose"][0:6]
            position2 = self._reading["right"]["arm"]["tcp_pose"][0:6]
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

    # Robot action helpers.

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

    def _get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Build an observation from robot state and camera frames.

        Returns:
            Observation with ``state`` and ``frames`` dictionaries.
        """
        if not self.config.is_dummy:
            frames = [
                self._reading[f"wrist_{index + 1}"]["frame"]
                for index in range(len(self.config.use_camera_ids))
            ]
            assert len(frames) == len(self.config.use_camera_ids), "get frames failed."
            for i in range(len(frames)):
                frames[i] = self._crop_frame(frames[i], (128, 128))
            tcp_pose = []
            if 0 in self.config.use_arm_ids:
                tmp = np.zeros(7)
                tmp[0:3] = self._reading["left"]["arm"]["tcp_pose"][0:3]
                r1 = R.from_euler("xyz", self._reading["left"]["arm"]["tcp_pose"][3:6])
                tmp[3:7] = r1.as_quat()
                tcp_pose.append(tmp.copy())
            if 1 in self.config.use_arm_ids:
                tmp = np.zeros(7)
                tmp[0:3] = self._reading["right"]["arm"]["tcp_pose"][0:3]
                r2 = R.from_euler("xyz", self._reading["right"]["arm"]["tcp_pose"][3:6])
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
