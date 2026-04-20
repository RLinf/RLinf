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

"""Dual-arm Franka environment."""

from __future__ import annotations

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

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import DualFrankaHWInfo, WorkerInfo
from rlinf.utils.logging import get_logger

from .franka_robot_state import FrankaRobotState
from .utils import clip_euler_to_target_window, quat_slerp

NUM_ARMS = 2
ACTION_DIM_PER_ARM = 7  # xyz_delta(3) + rpy_delta(3) + gripper(1)
TCP_POSE_DIM = 7  # xyz(3) + quat(4)
TCP_VEL_DIM = 6
# Avoids Ray actor name collision when both arms land on the same node.
_RIGHT_ARM_ENV_IDX_OFFSET = 1000
_MAX_CAMERA_RETRIES = 3


@dataclass
class DualFrankaRobotConfig:
    """Configuration for the dual-arm Franka environment."""

    left_robot_ip: Optional[str] = None
    right_robot_ip: Optional[str] = None

    left_camera_serials: Optional[list[str]] = None
    right_camera_serials: Optional[list[str]] = None
    base_camera_serials: Optional[list[str]] = None
    camera_type: Optional[str] = None

    left_gripper_type: Optional[str] = None
    right_gripper_type: Optional[str] = None
    left_gripper_connection: Optional[str] = None
    right_gripper_connection: Optional[str] = None

    enable_camera_player: bool = True
    is_dummy: bool = False
    use_dense_reward: bool = False
    step_frequency: float = 10.0

    # (2, 6) arrays: row 0 = left arm, row 1 = right arm
    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    joint_reset_qpos: list[list[float]] = field(
        default_factory=lambda: [[0, 0, 0, -1.9, 0, 2, 0]] * 2
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    action_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    enable_random_reset: bool = False
    random_xy_range: float = 0.0
    random_rz_range: float = 0.0

    # (2, 6) arrays for per-arm safety box
    ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), -np.inf)
    )
    ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), np.inf)
    )

    compliance_param: dict[str, float] = field(default_factory=dict)
    precision_param: dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    joint_reset_cycle: int = 20000
    task_description: str = ""
    success_hold_steps: int = 1

    def __post_init__(self):
        self.target_ee_pose = np.array(self.target_ee_pose).reshape(2, 6)
        self.reset_ee_pose = np.array(self.reset_ee_pose).reshape(2, 6)
        self.reward_threshold = np.array(self.reward_threshold).reshape(2, 6)
        self.action_scale = np.array(self.action_scale)
        self.ee_pose_limit_min = np.array(self.ee_pose_limit_min).reshape(2, 6)
        self.ee_pose_limit_max = np.array(self.ee_pose_limit_max).reshape(2, 6)


class DualFrankaEnv(gym.Env):
    """Dual-arm Franka env; each arm's controller may live on a different Ray node."""

    CONFIG_CLS: type[DualFrankaRobotConfig] = DualFrankaRobotConfig

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[DualFrankaHWInfo],
        env_idx: int,
    ):
        config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self.config = config
        self._task_description = config.task_description
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._left_state = FrankaRobotState()
        self._right_state = FrankaRobotState()

        if not self.config.is_dummy:
            self._reset_poses = np.zeros((2, 7))
            for arm_idx in range(NUM_ARMS):
                euler = self.config.reset_ee_pose[arm_idx]
                self._reset_poses[arm_idx] = np.concatenate(
                    [
                        euler[:3],
                        R.from_euler("xyz", euler[3:].copy()).as_quat(),
                    ]
                )
        else:
            self._reset_poses = np.zeros((2, 7))

        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)
        self._success_hold_counter = 0

        if not self.config.is_dummy:
            self._setup_hardware()

        all_serials = self._all_camera_serials()
        assert len(all_serials) > 0, (
            "At least one camera serial must be provided for DualFrankaEnv."
        )
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for both arms to be ready
        for label, ctrl in [("left", self._left_ctrl), ("right", self._right_ctrl)]:
            t0 = time.time()
            while not ctrl.is_robot_up().wait()[0]:
                time.sleep(0.5)
                if time.time() - t0 > 30:
                    self._logger.warning(
                        "Waited %.0fs for %s Franka to be ready.",
                        time.time() - t0,
                        label,
                    )

        self._interpolate_move_both(self._reset_poses)
        time.sleep(1.0)
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    @property
    def task_description(self):
        return self._task_description

    def close(self):
        if hasattr(self, "_cameras"):
            self._close_cameras()
        if hasattr(self, "camera_player"):
            self.camera_player.stop()

    def _all_camera_specs(self) -> list[tuple[str, str]]:
        """Camera specs as ``[(name, serial), ...]`` with pi0-aligned names."""
        specs: list[tuple[str, str]] = []
        if self.config.base_camera_serials:
            for j, serial in enumerate(self.config.base_camera_serials):
                specs.append((f"base_{j}_rgb", serial))
        for arm, serials in (
            ("left", self.config.left_camera_serials),
            ("right", self.config.right_camera_serials),
        ):
            if not serials:
                continue
            for j, serial in enumerate(serials):
                specs.append((f"{arm}_wrist_{j}_rgb", serial))
        return specs

    def _all_camera_serials(self) -> list[str]:
        return [serial for _, serial in self._all_camera_specs()]

    def _setup_hardware(self):
        from .franka_controller import FrankaController

        assert self.env_idx >= 0, "env_idx must be set for DualFrankaEnv."

        if self.hardware_info is not None:
            assert isinstance(self.hardware_info, DualFrankaHWInfo), (
                f"hardware_info must be DualFrankaHWInfo, got {type(self.hardware_info)}."
            )
            hw = self.hardware_info.config
            if self.config.left_robot_ip is None:
                self.config.left_robot_ip = hw.left_robot_ip
            if self.config.right_robot_ip is None:
                self.config.right_robot_ip = hw.right_robot_ip
            if self.config.left_camera_serials is None:
                self.config.left_camera_serials = hw.left_camera_serials
            if self.config.right_camera_serials is None:
                self.config.right_camera_serials = hw.right_camera_serials
            if self.config.base_camera_serials is None:
                self.config.base_camera_serials = getattr(
                    hw, "base_camera_serials", None
                )
            if self.config.camera_type is None:
                self.config.camera_type = getattr(hw, "camera_type", "zed")
            if self.config.left_gripper_type is None:
                self.config.left_gripper_type = getattr(
                    hw, "left_gripper_type", "franka"
                )
            if self.config.right_gripper_type is None:
                self.config.right_gripper_type = getattr(
                    hw, "right_gripper_type", "franka"
                )
            if self.config.left_gripper_connection is None:
                self.config.left_gripper_connection = getattr(
                    hw, "left_gripper_connection", None
                )
            if self.config.right_gripper_connection is None:
                self.config.right_gripper_connection = getattr(
                    hw, "right_gripper_connection", None
                )

        left_node = self.node_rank
        right_node = self.node_rank
        if self.hardware_info is not None:
            hw = self.hardware_info.config
            if hw.left_controller_node_rank is not None:
                left_node = hw.left_controller_node_rank
            if hw.right_controller_node_rank is not None:
                right_node = hw.right_controller_node_rank

        self._left_ctrl = FrankaController.launch_controller(
            robot_ip=self.config.left_robot_ip,
            env_idx=self.env_idx,
            node_rank=left_node,
            worker_rank=self.env_worker_rank,
            gripper_type=self.config.left_gripper_type or "franka",
            gripper_connection=self.config.left_gripper_connection,
        )
        self._right_ctrl = FrankaController.launch_controller(
            robot_ip=self.config.right_robot_ip,
            env_idx=self.env_idx + _RIGHT_ARM_ENV_IDX_OFFSET,
            node_rank=right_node,
            worker_rank=self.env_worker_rank,
            gripper_type=self.config.right_gripper_type or "franka",
            gripper_connection=self.config.right_gripper_connection,
        )

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        camera_type = self.config.camera_type or "zed"
        camera_infos = [
            CameraInfo(name=name, serial_number=serial, camera_type=camera_type)
            for name, serial in self._all_camera_specs()
        ]
        for info in camera_infos:
            camera = create_camera(info)
            camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
        resized = cv2.resize(cropped, reshape_size)
        return cropped, resized

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        for attempt in range(_MAX_CAMERA_RETRIES):
            frames: dict[str, np.ndarray] = {}
            display_frames: dict[str, np.ndarray] = {}
            failed = False
            for camera in self._cameras:
                try:
                    frame = camera.get_frame()
                    reshape_size = self.observation_space["frames"][
                        camera._camera_info.name
                    ].shape[:2][::-1]
                    cropped, resized = self._crop_frame(frame, reshape_size)
                    frames[camera._camera_info.name] = resized[..., ::-1]
                    display_frames[camera._camera_info.name] = resized
                    display_frames[f"{camera._camera_info.name}_full"] = cropped
                except queue.Empty:
                    self._logger.warning(
                        "Camera %s not producing frames (attempt %d/%d). Retrying in 5s.",
                        camera._camera_info.name,
                        attempt + 1,
                        _MAX_CAMERA_RETRIES,
                    )
                    time.sleep(5)
                    self._close_cameras()
                    self._open_cameras()
                    failed = True
                    break
            if not failed:
                self.camera_player.put_frame(display_frames)
                return frames
        raise RuntimeError(
            f"Cameras failed to produce frames after {_MAX_CAMERA_RETRIES} attempts."
        )

    def _init_action_obs_spaces(self):
        # Per-arm safety boxes
        self._xyz_safe_spaces = []
        self._rpy_safe_spaces = []
        for arm in range(NUM_ARMS):
            self._xyz_safe_spaces.append(
                gym.spaces.Box(
                    low=self.config.ee_pose_limit_min[arm, :3],
                    high=self.config.ee_pose_limit_max[arm, :3],
                    dtype=np.float64,
                )
            )
            self._rpy_safe_spaces.append(
                gym.spaces.Box(
                    low=self.config.ee_pose_limit_min[arm, 3:],
                    high=self.config.ee_pose_limit_max[arm, 3:],
                    dtype=np.float64,
                )
            )

        total_action_dim = NUM_ARMS * ACTION_DIM_PER_ARM
        self.action_space = gym.spaces.Box(
            np.ones(total_action_dim, dtype=np.float32) * -1,
            np.ones(total_action_dim, dtype=np.float32),
        )

        camera_specs = self._all_camera_specs()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(NUM_ARMS * TCP_POSE_DIM,),
                        ),
                        "tcp_vel": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(NUM_ARMS * TCP_VEL_DIM,),
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(NUM_ARMS,)),
                        "tcp_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * 3,)
                        ),
                        "tcp_torque": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * 3,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0,
                            255,
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                        )
                        for name, _ in camera_specs
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = action.reshape(NUM_ARMS, ACTION_DIM_PER_ARM)

        is_gripper_effective = [True, True]

        if not self.config.is_dummy:
            states = [self._left_state, self._right_state]
            ctrls = [self._left_ctrl, self._right_ctrl]
            # Compute target positions for both arms
            target_positions = []
            for arm in range(NUM_ARMS):
                arm_action = actions[arm]
                next_pos = states[arm].tcp_pose.copy()
                next_pos[:3] += arm_action[:3] * self.config.action_scale[0]
                next_pos[3:] = (
                    R.from_euler("xyz", arm_action[3:6] * self.config.action_scale[1])
                    * R.from_quat(states[arm].tcp_pose[3:].copy())
                ).as_quat()
                next_pos = self._clip_position_to_safety_box(next_pos, arm)
                target_positions.append(next_pos)

            # Handle grippers
            for arm in range(NUM_ARMS):
                gripper_val = actions[arm, 6] * self.config.action_scale[2]
                is_gripper_effective[arm] = self._gripper_action(
                    ctrls[arm],
                    states[arm],
                    gripper_val,
                )

            # Send move commands in parallel (fire both, then wait both)
            left_future = ctrls[0].move_arm(target_positions[0].astype(np.float32))
            right_future = ctrls[1].move_arm(target_positions[1].astype(np.float32))
            left_future.wait()
            right_future.wait()

        self._num_steps += 1
        if not self.config.is_dummy:
            step_time = time.time() - start_time
            time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            # Read states in parallel
            left_st_f = ctrls[0].get_state()
            right_st_f = ctrls[1].get_state()
            self._left_state = left_st_f.wait()[0]
            self._right_state = right_st_f.wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(is_gripper_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        self._num_steps = 0
        self._success_hold_counter = 0

        if self.config.is_dummy:
            return self._get_observation(), {}

        for ctrl in (self._left_ctrl, self._right_ctrl):
            ctrl.reconfigure_compliance_params(self.config.compliance_param).wait()

        joint_cycle = next(self._joint_reset_cycle)
        joint_reset = joint_cycle == 0
        if joint_reset:
            self._logger.info(
                "Number of resets reached %d, resetting joints.",
                self.config.joint_reset_cycle,
            )

        self._go_to_rest(joint_reset)
        self._clear_errors()

        left_st_f = self._left_ctrl.get_state()
        right_st_f = self._right_ctrl.get_state()
        self._left_state = left_st_f.wait()[0]
        self._right_state = right_st_f.wait()[0]
        return self._get_observation(), {}

    def get_tcp_pose(self) -> np.ndarray:
        """Return concatenated TCP poses ``(14,)`` for both arms."""
        left_st_f = self._left_ctrl.get_state()
        right_st_f = self._right_ctrl.get_state()
        self._left_state = left_st_f.wait()[0]
        self._right_state = right_st_f.wait()[0]
        return np.concatenate([self._left_state.tcp_pose, self._right_state.tcp_pose])

    def get_action_scale(self) -> np.ndarray:
        """Return the action scaling factors used by teleop wrappers."""
        return self.config.action_scale

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def target_ee_pose(self):
        """Return concatenated target poses ``(14,)`` in quaternion form."""
        poses = []
        for arm in range(NUM_ARMS):
            euler = self.config.target_ee_pose[arm]
            poses.append(
                np.concatenate(
                    [
                        euler[:3],
                        R.from_euler("xyz", euler[3:].copy()).as_quat(),
                    ]
                )
            )
        return np.concatenate(poses)

    def _clip_position_to_safety_box(
        self, position: np.ndarray, arm_idx: int
    ) -> np.ndarray:
        position[:3] = np.clip(
            position[:3],
            self._xyz_safe_spaces[arm_idx].low,
            self._xyz_safe_spaces[arm_idx].high,
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")
        euler = clip_euler_to_target_window(
            euler=euler,
            target_euler=self.config.target_ee_pose[arm_idx, 3:],
            lower_euler=self._rpy_safe_spaces[arm_idx].low,
            upper_euler=self._rpy_safe_spaces[arm_idx].high,
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()
        return position

    def _gripper_action(self, ctrl, state: FrankaRobotState, position: float) -> bool:
        threshold = self.config.binary_gripper_threshold
        if position <= -threshold and state.gripper_open:
            ctrl.close_gripper().wait()
            time.sleep(0.6)
            return True
        elif position >= threshold and not state.gripper_open:
            ctrl.open_gripper().wait()
            time.sleep(0.6)
            return True
        return False

    def _clear_errors(self):
        l = self._left_ctrl.clear_errors()
        r = self._right_ctrl.clear_errors()
        l.wait()
        r.wait()

    def _interpolate_move_both(self, target_poses: np.ndarray, timeout: float = 1.5):
        """Interpolate both arms towards *target_poses* ``(2, 7)``."""
        num_steps = int(timeout * self.config.step_frequency)
        left_st = self._left_ctrl.get_state().wait()[0]
        right_st = self._right_ctrl.get_state().wait()[0]
        states = [left_st, right_st]
        ctrls = [self._left_ctrl, self._right_ctrl]

        paths = []
        for arm in range(NUM_ARMS):
            pos_path = np.linspace(
                states[arm].tcp_pose[:3], target_poses[arm, :3], num_steps + 1
            )
            quat_path = quat_slerp(
                states[arm].tcp_pose[3:], target_poses[arm, 3:], num_steps + 1
            )
            paths.append((pos_path, quat_path))

        for step_i in range(1, num_steps + 1):
            for arm in range(NUM_ARMS):
                pose = np.concatenate([paths[arm][0][step_i], paths[arm][1][step_i]])
                ctrls[arm].move_arm(pose.astype(np.float32)).wait()
            time.sleep(1.0 / self.config.step_frequency)

    def _go_to_rest(self, joint_reset: bool = False):
        ctrls = [self._left_ctrl, self._right_ctrl]
        if joint_reset:
            for arm, ctrl in enumerate(ctrls):
                ctrl.reset_joint(self.config.joint_reset_qpos[arm]).wait()
            time.sleep(0.5)

        reset_poses = self._reset_poses.copy()
        if self.config.enable_random_reset:
            for arm in range(NUM_ARMS):
                reset_poses[arm, :2] += np.random.uniform(
                    -self.config.random_xy_range,
                    self.config.random_xy_range,
                    (2,),
                )
                euler_random = self.config.target_ee_pose[arm, 3:].copy()
                euler_random[-1] += np.random.uniform(
                    -self.config.random_rz_range,
                    self.config.random_rz_range,
                )
                reset_poses[arm, 3:] = R.from_euler("xyz", euler_random).as_quat()

        left_st = ctrls[0].get_state().wait()[0]
        right_st = ctrls[1].get_state().wait()[0]
        states_arr = [left_st, right_st]

        for cnt in range(3):
            converged = True
            for arm in range(NUM_ARMS):
                if not np.allclose(
                    states_arr[arm].tcp_pose[:3], reset_poses[arm, :3], atol=0.02
                ):
                    converged = False
            if converged:
                break
            self._interpolate_move_both(reset_poses)
            left_st = ctrls[0].get_state().wait()[0]
            right_st = ctrls[1].get_state().wait()[0]
            states_arr = [left_st, right_st]

    def _calc_step_reward(self, is_gripper_effective: list[bool]) -> float:
        if self.config.is_dummy:
            return 0.0

        all_in_zone = True
        dense_sq_sum = 0.0
        for arm, state in enumerate([self._left_state, self._right_state]):
            euler = np.abs(R.from_quat(state.tcp_pose[3:].copy()).as_euler("xyz"))
            position = np.hstack([state.tcp_pose[:3], euler])
            delta = np.abs(position - self.config.target_ee_pose[arm])
            if not np.all(delta[:3] <= self.config.reward_threshold[arm, :3]):
                all_in_zone = False
                dense_sq_sum += np.sum(np.square(delta[:3]))

        if all_in_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(np.exp(-500 * dense_sq_sum))
            else:
                reward = 0.0

        if self.config.enable_gripper_penalty:
            for eff in is_gripper_effective:
                if eff:
                    reward -= self.config.gripper_penalty
        return reward

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "tcp_pose": np.concatenate(
                    [
                        self._left_state.tcp_pose,
                        self._right_state.tcp_pose,
                    ]
                ),
                "tcp_vel": np.concatenate(
                    [
                        self._left_state.tcp_vel,
                        self._right_state.tcp_vel,
                    ]
                ),
                "gripper_position": np.array(
                    [
                        self._left_state.gripper_position,
                        self._right_state.gripper_position,
                    ],
                    dtype=np.float32,
                ),
                "tcp_force": np.concatenate(
                    [
                        self._left_state.tcp_force,
                        self._right_state.tcp_force,
                    ]
                ),
                "tcp_torque": np.concatenate(
                    [
                        self._left_state.tcp_torque,
                        self._right_state.tcp_torque,
                    ]
                ),
            }
            return copy.deepcopy({"state": state, "frames": frames})
        else:
            return self._base_observation_space.sample()
