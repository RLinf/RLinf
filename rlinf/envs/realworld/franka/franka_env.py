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
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import (
    FrankaHWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger

from .end_effectors.base import EndEffectorType, normalize_end_effector_type
from .franka_robot_state import FrankaRobotState
from .utils import (
    clip_euler_to_target_window,
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
    quat_slerp,
)


@dataclass
class FrankaRobotConfig:
    robot_ip: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    camera_names: Optional[dict[str, str]] = None
    camera_type: Optional[str] = None
    gripper_type: Optional[str] = None
    gripper_connection: Optional[str] = None
    enable_camera_player: bool = True
    # Per-camera crop regions keyed by serial number.
    # Each value is [top%, left%, bottom%, right%] in 0..1 range.
    # Example: {"230322271990": [0.0, 0.15, 1.0, 0.85]}
    camera_crop_regions: Optional[dict[str, list[float]]] = None

    is_dummy: bool = False
    use_dense_reward: bool = False
    reward_scale: float = 1.0  # Scale dense reward to make training stable
    step_frequency: float = 10.0  # Max number of steps per second

    use_reward_model: bool = False
    reward_worker_cfg: Optional[dict] = None
    reward_worker_hardware_rank: Optional[int] = None
    reward_worker_node_rank: Optional[int] = None
    reward_worker_node_group: Optional[str] = None
    reward_image_key: Optional[str] = None

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
    task_description: str = ""
    success_hold_steps: int = (
        1  # Default to 1 to maintain backward compatibility (immediate success)
    )

    # -- End-effector selection -------------------------------------------
    # One of "franka_gripper", "robotiq_gripper", or "ruiyan_hand".
    end_effector_type: str = "franka_gripper"
    # Extra kwargs forwarded to the end-effector constructor.
    end_effector_config: dict = field(default_factory=dict)
    # Target hand pose used for dense-reward success criteria (6-D).
    hand_target_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Default hand pose after ``reset()`` (6-D).
    hand_reset_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # Hand action scale (for continuous hand control).
    hand_action_scale: float = 1.0
    # Max per-step change for hand joints (set to inf to disable).
    hand_max_delta_per_step: float = float("inf")

    def __post_init__(self):
        """Convert list fields from YAML/Hydra to numpy arrays."""
        if self.camera_names is not None:
            self.camera_names = {
                str(serial): str(camera_name)
                for serial, camera_name in self.camera_names.items()
            }
        if self.camera_crop_regions is not None:
            self.camera_crop_regions = {
                str(serial): crop_region
                for serial, crop_region in self.camera_crop_regions.items()
            }
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = np.array(self.reset_ee_pose)
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array(self.action_scale)
        self.ee_pose_limit_min = np.array(self.ee_pose_limit_min)
        self.ee_pose_limit_max = np.array(self.ee_pose_limit_max)
        self.hand_target_state = np.array(self.hand_target_state)
        self.hand_reset_state = np.array(self.hand_reset_state)


class FrankaEnv(gym.Env):
    """Franka robot arm environment."""

    CONFIG_CLS: type[FrankaRobotConfig] = FrankaRobotConfig

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[FrankaHWInfo],
        env_idx: int,
    ):
        config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self.config = config
        self.config.end_effector_type = normalize_end_effector_type(
            self.config.end_effector_type,
            self.config.gripper_type,
        ).value
        self._task_description = config.task_description
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

        self._success_hold_counter = 0  # Initialize the success hold counter
        self._last_hand_command: np.ndarray | None = None
        self._reward_worker = None

        if not self.config.is_dummy:
            self._setup_hardware()
            self._setup_reward_worker()

        self._camera_infos = self._build_camera_infos()

        # Init action and observation spaces
        assert self._camera_infos, (
            "At least one camera serial must be provided for FrankaEnv."
        )
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

    @property
    def task_description(self):
        return self._task_description

    def _setup_hardware(self):
        from .franka_controller import FrankaController

        assert self.env_idx >= 0, "env_idx must be set for FrankaEnv."

        # Setup Franka IP and camera serials
        assert isinstance(self.hardware_info, FrankaHWInfo), (
            f"hardware_info must be FrankaHWInfo, but got {type(self.hardware_info)}."
        )
        if self.config.robot_ip is None:
            self.config.robot_ip = self.hardware_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials
        if self.config.camera_type is None:
            self.config.camera_type = getattr(
                self.hardware_info.config, "camera_type", "realsense"
            )
        if self.config.gripper_type is None:
            self.config.gripper_type = getattr(
                self.hardware_info.config, "gripper_type", "franka"
            )
        if self.config.gripper_connection is None:
            self.config.gripper_connection = getattr(
                self.hardware_info.config, "gripper_connection", None
            )
        self.config.end_effector_type = normalize_end_effector_type(
            self.config.end_effector_type,
            self.config.gripper_type,
        ).value

        # Place the controller on controller_node_rank if the arm lives on a
        # different machine (e.g. cameras on GPU server, arm on NUC).
        # Falls back to the env worker's own node when not specified.
        controller_node_rank = getattr(
            self.hardware_info.config, "controller_node_rank", None
        )
        if controller_node_rank is None:
            controller_node_rank = self.node_rank
        self._controller = FrankaController.launch_controller(
            robot_ip=self.config.robot_ip,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            end_effector_type=self.config.end_effector_type,
            end_effector_config=self.config.end_effector_config,
            gripper_connection=self.config.gripper_connection,
        )

    def _setup_reward_worker(self):
        if not self.config.use_reward_model:
            return
        if self.config.reward_worker_cfg is None:
            raise ValueError(
                "use_reward_model=True but reward_worker_cfg is not provided in env override_cfg."
            )

        from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

        reward_node_rank = self.config.reward_worker_node_rank
        if reward_node_rank is None:
            reward_node_rank = self.node_rank

        self._reward_worker = EmbodiedRewardWorker.launch_for_realworld(
            reward_cfg=self.config.reward_worker_cfg,
            node_rank=reward_node_rank,
            node_group_label=self.config.reward_worker_node_group,
            hardware_rank=self.config.reward_worker_hardware_rank,
            env_idx=self.env_idx,
            worker_rank=self.env_worker_rank,
        )
        self._reward_worker.init_worker().wait()

    def transform_action_ee_to_base(self, action):
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        For gripper end-effectors (7-D action)::

            [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper_action]

        For dexterous hand (12-D action)::

            [
                x_delta,
                y_delta,
                z_delta,
                rx_delta,
                ry_delta,
                rz_delta,
                h1,
                h2,
                h3,
                h4,
                h5,
                h6,
            ]
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.next_position = self._franka_state.tcp_pose.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self.config.action_scale[0]
        )

        is_ee_action_effective = True
        if not self.config.is_dummy:
            self.next_position[3:] = (
                R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                * R.from_quat(self._franka_state.tcp_pose[3:].copy())
            ).as_quat()

            # --- End-effector action ---
            ee_action = action[6:]
            is_ee_action_effective = self._end_effector_action(ee_action)

            self._move_action(self._clip_position_to_safety_box(self.next_position))

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        else:
            self._franka_state = self._franka_state
        observation = self._get_observation()

        # Calculate reward and update the internal hold counter
        reward = self._calc_step_reward(observation, is_ee_action_effective)

        # Logic to determine termination
        # The episode is done only if the robot has reached the target (reward == 1.0)
        # AND has held the position for the required number of steps.
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )

        truncated = self._num_steps >= self.config.max_num_steps
        reward *= self.config.reward_scale
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def get_tcp_pose(self) -> np.ndarray:
        """Return the current TCP pose ``[x, y, z, qx, qy, qz, qw]``."""
        self._franka_state = self._controller.get_state().wait()[0]
        return self._franka_state.tcp_pose

    def get_action_scale(self) -> np.ndarray:
        """Return the action scale ``[pos_scale, ori_scale, gripper_scale]``."""
        return self.config.action_scale

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
        if self.config.use_reward_model:
            reward = self._compute_reward_model(observation)
            if reward >= 1.0:
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty
            return reward

        if not self.config.is_dummy:
            # Convert orientation to euler angles
            euler_angles = np.abs(
                R.from_quat(self._franka_state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._franka_state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)

            # Check if current state meets the success threshold
            is_in_target_zone = np.all(
                target_delta[:3] <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                # Increment hold counter if in target zone
                self._success_hold_counter += 1
                reward = 1.0
            else:
                # Reset counter if robot leaves the target zone
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria. Target delta: {target_delta}, "
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            if (
                self.config.enable_gripper_penalty
                and not self._is_hand
                and is_gripper_action_effective
            ):
                reward -= self.config.gripper_penalty

            return reward
        else:
            return 0.0

    def _compute_reward_model(
        self, observation: dict[str, np.ndarray | FrankaRobotState]
    ) -> float:
        if self._reward_worker is None:
            raise RuntimeError("Reward worker is not initialized.")

        frames = observation.get("frames", {})
        if not frames:
            raise ValueError("No frames available for reward model inference.")

        image_key = self.config.reward_image_key
        if image_key is None:
            image_key = sorted(frames.keys())[0]
        if image_key not in frames:
            raise KeyError(
                f"reward_image_key '{image_key}' not found in frames. "
                f"Available keys: {list(frames.keys())}"
            )

        image_batch = np.expand_dims(frames[image_key], axis=0)
        reward_output = self._reward_worker.compute_image_rewards(image_batch).wait()[0]
        if hasattr(reward_output, "detach"):
            reward_output = reward_output.detach().cpu().numpy()
        reward_array = np.asarray(reward_output).reshape(-1)
        return float(reward_array[0])

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        self._success_hold_counter = 0  # Reset hold counter at the start of the episode

        self._controller.reconfigure_compliance_params(
            self.config.compliance_param
        ).wait()

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

        # Reset dexterous hands here. Gripper state is task-specific, matching
        # the upstream Franka reset path where the base env does not open/close it.
        if self._is_hand:
            self._controller.reset_end_effector(self.config.hand_reset_state).wait()
            self._last_hand_command = (
                np.array(self.config.hand_reset_state, dtype=np.float64)
                * self.config.hand_action_scale
            )

    @property
    def _ee_type(self) -> EndEffectorType:
        """Cached end-effector type enum."""
        return EndEffectorType(self.config.end_effector_type)

    @property
    def _is_hand(self) -> bool:
        """Whether the active end-effector is a dexterous hand."""
        return self._ee_type.is_hand

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box.

        The action dimension adapts to the active end-effector:
        - Gripper: 7-D (6 arm + 1 gripper)
        - Dexterous hand: 12-D (6 arm + 6 hand DOFs)
        """
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

        # Arm DOF (xyz + rpy) = 6; end-effector DOF depends on type
        ee_action_dim = 6 if self._is_hand else 1
        total_action_dim = 6 + ee_action_dim
        self.action_space = gym.spaces.Box(
            np.ones((total_action_dim,), dtype=np.float32) * -1,
            np.ones((total_action_dim,), dtype=np.float32),
        )

        obs_tcp_pose_dim = 7
        # End-effector state key and dimension
        if self._is_hand:
            ee_state_key = "hand_position"
            ee_state_dim = 6
            ee_low, ee_high = 0.0, 1.0
        else:
            ee_state_key = "gripper_position"
            ee_state_dim = 1
            ee_low, ee_high = -1.0, 1.0

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(obs_tcp_pose_dim,)
                        ),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        ee_state_key: gym.spaces.Box(
                            ee_low, ee_high, shape=(ee_state_dim,)
                        ),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        camera_info.name: gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for camera_info in self._camera_infos
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    @staticmethod
    def _normalize_crop_region(
        crop_region: Any,
        *,
        camera_name: str,
        serial: str,
    ) -> tuple[float, float, float, float]:
        """Validate and normalize a crop region from the config."""
        if not isinstance(crop_region, (list, tuple)) or len(crop_region) != 4:
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): expected "
                "[top, left, bottom, right]."
            )

        try:
            top_pct, left_pct, bottom_pct, right_pct = tuple(
                float(value) for value in crop_region
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): expected numeric values, "
                f"got {crop_region!r}."
            ) from exc

        normalized_crop_region = (top_pct, left_pct, bottom_pct, right_pct)
        if not all(0.0 <= value <= 1.0 for value in normalized_crop_region):
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): values must be within "
                f"[0, 1], got {crop_region!r}."
            )
        if bottom_pct <= top_pct or right_pct <= left_pct:
            raise ValueError(
                "Invalid crop_region for camera "
                f"'{camera_name}' ({serial}): expected "
                "bottom > top and right > left, "
                f"got {crop_region!r}."
            )

        return normalized_crop_region

    def _build_camera_infos(self) -> list[CameraInfo]:
        if self.config.camera_serials is None:
            return []

        ordered_serials = [str(serial) for serial in self.config.camera_serials]

        default_camera_type = self.config.camera_type or "realsense"
        camera_names = self.config.camera_names or {}
        camera_crop_regions = self.config.camera_crop_regions or {}
        camera_infos: list[CameraInfo] = []
        for camera_index, serial in enumerate(ordered_serials, start=1):
            default_name = f"wrist_{camera_index}"
            name = camera_names.get(serial, default_name)

            crop_region = camera_crop_regions.get(serial)
            if crop_region is not None:
                crop_region = self._normalize_crop_region(
                    crop_region,
                    camera_name=name,
                    serial=serial,
                )

            camera_infos.append(
                CameraInfo(
                    name=name,
                    serial_number=serial,
                    camera_type=default_camera_type,
                    crop_region=crop_region,
                )
            )

        return camera_infos

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        if not self._camera_infos:
            return
        for info in self._camera_infos:
            camera = create_camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def close(self):
        """Release all hardware resources including cameras and video player."""
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if not self.config.is_dummy and hasattr(self, "_cameras"):
            self._close_cameras()
        super().close()

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self,
        frame: np.ndarray,
        reshape_size: tuple[int, int],
        crop_region: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop the frame and resize.

        Args:
            frame: Raw camera frame ``(H, W, C)``.
            reshape_size: Target ``(width, height)`` after resize.
            crop_region: Optional relative crop ``(top, left, bottom, right)``
                where each value is in ``[0, 1]``.  ``None`` falls back to the
                default centre-square crop.

        Returns:
            A tuple of ``(cropped_frame, resized_frame)``.
        """
        h, w, _ = frame.shape
        if crop_region is not None:
            top_pct, left_pct, bottom_pct, right_pct = crop_region
            y1 = int(h * top_pct)
            x1 = int(w * left_pct)
            y2 = int(h * bottom_pct)
            x2 = int(w * right_pct)
            cropped_frame = frame[y1:y2, x1:x2]
        else:
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
                cropped_frame, resized_frame = self._crop_frame(
                    frame,
                    reshape_size,
                    crop_region=camera._camera_info.crop_region,
                )
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
        euler = clip_euler_to_target_window(
            euler=euler,
            target_euler=self.config.target_ee_pose[3:],
            lower_euler=self._rpy_safe_space.low,
            upper_euler=self._rpy_safe_space.high,
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _binary_gripper_action(self, position: float) -> bool:
        """Execute a scaled binary gripper command."""
        if (
            position <= -self.config.binary_gripper_threshold
            and self._franka_state.gripper_open
        ):
            self._controller.close_gripper().wait()
            time.sleep(0.6)
            return True
        if (
            position >= self.config.binary_gripper_threshold
            and not self._franka_state.gripper_open
        ):
            self._controller.open_gripper().wait()
            time.sleep(0.6)
            return True
        return False

    def _end_effector_action(self, ee_action: np.ndarray) -> bool:
        """Dispatch an action to the active end-effector.

        For gripper end-effectors the action is a scalar binary signal;
        for dexterous hands it is a 6-D continuous target.

        Args:
            ee_action: End-effector portion of the action vector (after the
                first 6 arm DOFs).

        Returns:
            ``True`` if the action caused a meaningful state change.
        """
        if self._ee_type.is_gripper:
            # Binary gripper logic (backward compatible)
            position = float(ee_action[0]) * self.config.action_scale[2]
            return self._binary_gripper_action(position)
        else:
            scaled = (
                np.asarray(ee_action, dtype=np.float64) * self.config.hand_action_scale
            )
            if self._last_hand_command is not None:
                delta = scaled - self._last_hand_command
                max_d = self.config.hand_max_delta_per_step
                scaled = self._last_hand_command + np.clip(delta, -max_d, max_d)
            self._last_hand_command = scaled.copy()
            self._controller.command_end_effector(scaled).wait()
            return True

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
            state: dict = {
                "tcp_pose": self._franka_state.tcp_pose,
                "tcp_vel": self._franka_state.tcp_vel,
                "tcp_force": self._franka_state.tcp_force,
                "tcp_torque": self._franka_state.tcp_torque,
            }
            # End-effector state (key matches observation_space)
            if self._is_hand:
                hand_pos = self._franka_state.hand_position
                if hand_pos is None:
                    hand_pos = np.zeros(6)
                state["hand_position"] = hand_pos
            else:
                state["gripper_position"] = np.array(
                    [self._franka_state.gripper_position]
                )
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
