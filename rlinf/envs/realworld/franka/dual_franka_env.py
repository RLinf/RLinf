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

"""Dual-arm Franka env driven through ``FrankyController`` (libfranka).

Abstract base; concrete subclasses (Joint, Rot6d) set ``PER_ARM_ACTION_DIM``
/ ``GRIPPER_IDX_IN_ARM`` and implement ``_init_action_obs_spaces`` +
``_get_observation`` + ``_dispatch_arm_motion``.
"""

from __future__ import annotations

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
from .franky_controller import FrankyController


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
    base_camera_type: Optional[str] = None
    left_camera_type: Optional[str] = None
    right_camera_type: Optional[str] = None

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
    """Dual-arm Franka env driven through ``FrankyController`` (libfranka).

    Abstract base. Subclasses set ``PER_ARM_ACTION_DIM`` / ``GRIPPER_IDX_IN_ARM``
    and implement ``_init_action_obs_spaces`` + ``_get_observation`` +
    ``_dispatch_arm_motion``.
    """

    CONFIG_CLS: type[DualFrankaRobotConfig] = DualFrankaRobotConfig
    PER_ARM_ACTION_DIM: int = 0
    GRIPPER_IDX_IN_ARM: int = 0
    _DEFAULT_GRIPPER_TYPE: str = "robotiq"

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

        # Initial state read
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

    # ---------------------------------------------------------------- cameras

    def _all_camera_specs(self) -> list[tuple[str, str, str]]:
        """Camera specs as ``[(name, serial, camera_type), ...]`` with pi0-aligned names.

        Per-slot ``*_camera_type`` falls back to the global ``camera_type``.
        """
        default_ct = self.config.camera_type or "realsense"
        specs: list[tuple[str, str, str]] = []
        if self.config.base_camera_serials:
            ct = self.config.base_camera_type or default_ct
            for j, serial in enumerate(self.config.base_camera_serials):
                specs.append((f"base_{j}_rgb", serial, ct))
        for arm, serials, slot_ct in (
            ("left", self.config.left_camera_serials, self.config.left_camera_type),
            ("right", self.config.right_camera_serials, self.config.right_camera_type),
        ):
            if not serials:
                continue
            ct = slot_ct or default_ct
            for j, serial in enumerate(serials):
                specs.append((f"{arm}_wrist_{j}_rgb", serial, ct))
        return specs

    def _all_camera_serials(self) -> list[str]:
        return [serial for _, serial, _ in self._all_camera_specs()]

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        camera_infos = [
            CameraInfo(name=name, serial_number=serial, camera_type=ct)
            for name, serial, ct in self._all_camera_specs()
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

    # ---------------------------------------------------------------- hardware

    def _resolve_hw_overrides(self) -> None:
        if self.hardware_info is None:
            return
        assert isinstance(self.hardware_info, DualFrankaHWInfo), (
            f"hardware_info must be DualFrankaHWInfo, got {type(self.hardware_info)}."
        )
        hw = self.hardware_info.config
        # (field_name, default if hw lacks the attr)
        hw_fallback_fields: tuple[tuple[str, object], ...] = (
            ("left_robot_ip", None),
            ("right_robot_ip", None),
            ("left_camera_serials", None),
            ("right_camera_serials", None),
            ("base_camera_serials", None),
            ("camera_type", "realsense"),
            ("base_camera_type", None),
            ("left_camera_type", None),
            ("right_camera_type", None),
            ("left_gripper_connection", None),
            ("right_gripper_connection", None),
        )
        for field_name, default in hw_fallback_fields:
            if getattr(self.config, field_name, None) is None:
                setattr(self.config, field_name, getattr(hw, field_name, default))
        for side in ("left_gripper_type", "right_gripper_type"):
            if getattr(self.config, side, None) is None:
                setattr(
                    self.config,
                    side,
                    getattr(hw, side, self._DEFAULT_GRIPPER_TYPE),
                )

    def _resolve_controller_node_ranks(self) -> tuple[int, int]:
        """Return per-arm node ranks, honoring the hw config overrides."""
        left_node = self.node_rank
        right_node = self.node_rank
        if self.hardware_info is not None:
            hw = self.hardware_info.config
            if hw.left_controller_node_rank is not None:
                left_node = hw.left_controller_node_rank
            if hw.right_controller_node_rank is not None:
                right_node = hw.right_controller_node_rank
        return left_node, right_node

    def _setup_hardware(self):
        assert self.env_idx >= 0, f"env_idx must be set for {type(self).__name__}."

        self._resolve_hw_overrides()
        left_node, right_node = self._resolve_controller_node_ranks()

        self._left_ctrl = FrankyController.launch_controller(
            robot_ip=self.config.left_robot_ip,
            env_idx=self.env_idx,
            node_rank=left_node,
            worker_rank=self.env_worker_rank,
            gripper_type=self.config.left_gripper_type or self._DEFAULT_GRIPPER_TYPE,
            gripper_connection=self.config.left_gripper_connection,
        )
        self._right_ctrl = FrankyController.launch_controller(
            robot_ip=self.config.right_robot_ip,
            env_idx=self.env_idx + _RIGHT_ARM_ENV_IDX_OFFSET,
            node_rank=right_node,
            worker_rank=self.env_worker_rank,
            gripper_type=self.config.right_gripper_type or self._DEFAULT_GRIPPER_TYPE,
            gripper_connection=self.config.right_gripper_connection,
        )

    # ---------------------------------------------------------------- reset/step

    def _go_to_rest(self, joint_reset: bool = False):
        del joint_reset
        try:
            self._left_ctrl.open_gripper()
            self._right_ctrl.open_gripper()
        except Exception as exc:
            self._logger.warning("open_gripper during reset failed: %s", exc)

        left_f = self._left_ctrl.reset_joint(self.config.joint_reset_qpos[0])
        right_f = self._right_ctrl.reset_joint(self.config.joint_reset_qpos[1])
        left_f.wait()
        right_f.wait()
        time.sleep(0.5)
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

    def reset(self, *, seed=None, options=None):
        """``options["skip_reset_to_home"]`` lets teleop wrappers keep tracking
        from the episode-end pose instead of bouncing through home."""
        del seed
        skip_reset_to_home = bool((options or {}).get("skip_reset_to_home", False))
        self._num_steps = 0
        self._success_hold_counter = 0

        if self.config.is_dummy:
            return self._get_observation(), {}

        joint_cycle = next(self._joint_reset_cycle)
        joint_reset = joint_cycle == 0
        if joint_reset:
            self._logger.info(
                "Number of resets reached %d, resetting joints.",
                self.config.joint_reset_cycle,
            )

        if skip_reset_to_home:
            self._logger.info(
                "skip_reset_to_home=True: holding arms at episode-end pose "
                "(teleop wrapper will realign to device)."
            )
        else:
            self._go_to_rest(joint_reset)
        self._clear_errors()

        left_st_f = self._left_ctrl.get_state()
        right_st_f = self._right_ctrl.get_state()
        self._left_state = left_st_f.wait()[0]
        self._right_state = right_st_f.wait()[0]
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = action.reshape(2, self.PER_ARM_ACTION_DIM)

        is_gripper_effective = [True, True]

        if not self.config.is_dummy:
            states = [self._left_state, self._right_state]
            ctrls = [self._left_ctrl, self._right_ctrl]
            dt = 1.0 / self.config.step_frequency

            # Grippers first so they don't contend with a fresh motion command.
            for arm in range(2):
                gripper_val = (
                    actions[arm, self.GRIPPER_IDX_IN_ARM] * self.config.action_scale[2]
                )
                is_gripper_effective[arm] = self._gripper_action(
                    ctrls[arm], states[arm], gripper_val
                )

            self._dispatch_arm_motion(actions, states, ctrls, dt)

        self._num_steps += 1
        if not self.config.is_dummy:
            if self._pace_between_action_and_state_read():
                step_time = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
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

    def _clear_errors(self):
        l = self._left_ctrl.clear_errors()
        r = self._right_ctrl.clear_errors()
        l.wait()
        r.wait()

    # ---------------------------------------------------------------- gripper / utils

    def _gripper_action(self, ctrl, state, position: float) -> bool:
        # Fire-and-forget: collection streams gripper RPCs at 10 Hz and a
        # blocking .wait() + 0.6 s sleep stretches eval steps to ~700 ms
        # and rings out j7.
        threshold = self.config.binary_gripper_threshold
        if position <= -threshold and state.gripper_open:
            ctrl.close_gripper()
            return True
        elif position >= threshold and not state.gripper_open:
            ctrl.open_gripper()
            return True
        return False

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

    def get_joint_positions(self) -> np.ndarray:
        """Stacked ``(2, 7)`` joint positions from cached state (no RPC)."""
        return np.stack(
            [
                self._left_state.arm_joint_position.copy(),
                self._right_state.arm_joint_position.copy(),
            ]
        )

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def target_ee_pose(self):
        """Return concatenated target poses ``(14,)`` in quaternion form."""
        poses = []
        for arm in range(2):
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

    def _cartesian_safety_boxes(self) -> None:
        self._xyz_safe_spaces = []
        self._rpy_safe_spaces = []
        for arm in range(2):
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

    def _build_observation_space(self, joint_position_dim: int) -> gym.spaces.Dict:
        camera_specs = self._all_camera_specs()
        return gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2 * 7,)
                        ),
                        "tcp_vel": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2 * 6,)
                        ),
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(joint_position_dim,)
                        ),
                        "joint_velocity": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2 * 7,)
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(2,)),
                        "tcp_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2 * 3,)
                        ),
                        "tcp_torque": gym.spaces.Box(
                            -np.inf, np.inf, shape=(2 * 3,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(224, 224, 3), dtype=np.uint8
                        )
                        for name, _, _ in camera_specs
                    }
                ),
            }
        )

    # ---------------------------------------------------------------- reward

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

    # --------------------------------------------------------- subclass hooks

    def _init_action_obs_spaces(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_action_obs_spaces"
        )

    def _get_observation(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_observation"
        )

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        """Override in subclass to issue move_joints / move_tcp_pose."""
        del actions, states, ctrls, dt

    def _pace_between_action_and_state_read(self) -> bool:
        return True
