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

"""Robbyant R2 real-world environment: joint-position control with grippers/hands and optional base."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import Camera, R2Config, R2Robot, RobotInfo
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.r2 import (
    R2_HAND_FINGER_ORDER,
    R2_LEFT_HOME_QPOS,
    R2_RIGHT_HOME_QPOS,
    R2IKConfig,
    R2KinematicsAdapter,
)
from rlinf.robotics.parts.base import PartGroup
from rlinf.robotics.parts.cameras import BaseCamera
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

ARM_DOF = 7
_DEFAULT_SINGLE_LIMIT_LOW = np.full(ARM_DOF, -2.8, dtype=np.float64)
_DEFAULT_SINGLE_LIMIT_HIGH = np.full(ARM_DOF, 2.8, dtype=np.float64)


@dataclass
class R2EnvConfig:
    """Task, control, and observation settings for an R2 environment."""

    enable_camera_player: bool = True
    """Whether to show captured frames in a viewer window."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 20.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    camera_max_age: float = 0.5
    """Maximum age in seconds of a camera frame reused between control steps."""

    active_arms: tuple[str, ...] = ("left", "right")
    """Active R2 arms controlled by the environment: subset of ``('left', 'right')``."""

    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [
            float(x) for x in np.concatenate([R2_LEFT_HOME_QPOS, R2_RIGHT_HOME_QPOS])
        ]
    )
    """Rest joint configuration in radians across active arms."""

    reset_duration: float = 2.0
    """Minimum reset trajectory duration in seconds."""

    reset_joint_speed: float = float(np.deg2rad(30.0))
    """Maximum commanded joint speed during reset, in radians per second."""

    reset_on_init: bool = True
    """Whether construction moves the active arms to their reset pose."""

    joint_limit_low: np.ndarray = field(
        default_factory=lambda: np.tile(_DEFAULT_SINGLE_LIMIT_LOW, 2)
    )
    """Lower joint bounds in radians across active arms."""

    joint_limit_high: np.ndarray = field(
        default_factory=lambda: np.tile(_DEFAULT_SINGLE_LIMIT_HIGH, 2)
    )
    """Upper joint bounds in radians across active arms."""

    max_num_steps: int = 100
    """Steps before an episode is truncated."""

    target_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0] * (ARM_DOF * 2)
    )
    """Goal joint configuration the reward measures against, in radians."""

    reward_threshold: float = 0.05
    """Per-joint tolerance in radians. Every active joint must be within it."""

    use_dense_reward: bool = False
    """Report the negative Euclidean joint distance instead of a sparse hit."""

    success_hold_steps: int = 1
    """Consecutive successful steps before the episode terminates."""

    gripper_penalty: float = 0.1
    """Reward subtracted when an end-effector command changes its state."""

    enable_gripper_penalty: bool = False
    """Whether to charge :pyattr:`gripper_penalty` for end-effector motion."""

    truncate_on_fault_recovery: bool = True
    """Whether to mark the current episode as ``truncated=True`` after an automatic hardware fault recovery."""

    joint_step_limits: tuple[float, ...] = (
        0.08,
        0.08,
        0.08,
        0.08,
        0.12,
        0.12,
        0.12,
    )
    """Maximum per-tick joint step in radians for each of the 7 arm joints during teleoperation."""

    teleop_max_joint_lead: float = 0.3
    """Largest gap in radians a teleoperation command may run ahead of the measured joints.

    PICO teleoperation integrates its own joint command, so the per-tick step
    limit bounds its speed however far the streamed arm lags; this bound stops
    the command running away from an arm that is blocked.
    """

    ik: R2IKConfig = field(default_factory=R2IKConfig)
    """Inverse-kinematics solver configuration used by PICO VR teleoperation."""

    def __post_init__(self) -> None:
        """Coerce and validate active arms and joint bound arrays."""
        self.active_arms = tuple(self.active_arms)
        if not self.active_arms or any(
            side not in ("left", "right") for side in self.active_arms
        ):
            raise ValueError(
                "active_arms must be a non-empty subset of ('left', 'right'), "
                f"got {self.active_arms!r}."
            )
        num_joints = ARM_DOF * len(self.active_arms)
        low = np.asarray(self.joint_limit_low, dtype=np.float64).reshape(-1)
        high = np.asarray(self.joint_limit_high, dtype=np.float64).reshape(-1)
        if low.size == ARM_DOF and len(self.active_arms) == 2:
            low = np.tile(low, 2)
        elif low.size == ARM_DOF * 2 and len(self.active_arms) == 1:
            low = low[:ARM_DOF]
        if high.size == ARM_DOF and len(self.active_arms) == 2:
            high = np.tile(high, 2)
        elif high.size == ARM_DOF * 2 and len(self.active_arms) == 1:
            high = high[:ARM_DOF]
        if low.size != num_joints or high.size != num_joints:
            raise ValueError(
                f"Expected joint limits of length {num_joints}, "
                f"got low={low.size} and high={high.size}."
            )
        self.joint_limit_low = low
        self.joint_limit_high = high
        if len(self.reset_joint_qpos) == ARM_DOF * 2 and len(self.active_arms) == 1:
            self.reset_joint_qpos = list(
                self.reset_joint_qpos[:ARM_DOF]
                if self.active_arms[0] == "left"
                else self.reset_joint_qpos[ARM_DOF:]
            )
        if len(self.target_joint_qpos) == ARM_DOF * 2 and len(self.active_arms) == 1:
            self.target_joint_qpos = list(self.target_joint_qpos[:ARM_DOF])
        step_limits = tuple(float(x) for x in self.joint_step_limits)
        if len(step_limits) != ARM_DOF or any(
            not np.isfinite(x) or x <= 0.0 for x in step_limits
        ):
            raise ValueError(
                f"joint_step_limits must have {ARM_DOF} positive finite values, "
                f"got {self.joint_step_limits!r}."
            )
        self.joint_step_limits = step_limits
        if not isinstance(self.ik, R2IKConfig):
            self.ik = R2IKConfig(**dict(self.ik))


class R2Env(gym.Env):
    """Robbyant R2 environment with absolute joint-position actions."""

    TELEOP: tuple[str, ...] = ("pico",)
    TELEOP_DEFAULT = "none"
    TELEOP_MARK_FLAG = True
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        config: R2EnvConfig,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Optional[RobotInfo[R2Config]] = None,
        env_idx: int = 0,
    ) -> None:
        self._logger = get_logger()
        self.config = config
        if not np.isfinite(config.step_frequency) or config.step_frequency <= 0:
            raise ValueError("step_frequency must be finite and positive.")
        if not np.isfinite(config.camera_max_age) or config.camera_max_age < 0:
            raise ValueError("camera_max_age must be finite and nonnegative.")
        for name in ("reset_duration", "reset_joint_speed"):
            value = getattr(config, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        self.hardware = get_hardware_config(
            R2Config, robot_info, is_dummy=config.is_dummy
        )
        if self.hardware.body_dof in (None, "auto") or (
            self.hardware.end_effector_type in (None, "auto")
        ):
            if not self.config.is_dummy:
                from rlinf.robotics.parts.arms.r2 import R2Connection

                probed_dof, probed_ee = R2Connection.probe_hardware()
                if self.hardware.body_dof in (None, "auto"):
                    self.hardware.body_dof = probed_dof
                if self.hardware.end_effector_type in (None, "auto"):
                    self.hardware.end_effector_type = probed_ee
            else:
                if self.hardware.body_dof in (None, "auto"):
                    self.hardware.body_dof = 21
                if self.hardware.end_effector_type in (None, "auto"):
                    self.hardware.end_effector_type = "gripper"

        self.robot_info = robot_info
        self.env_idx = env_idx
        self.node_rank: Optional[int] = None
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._active_arms: tuple[str, ...] = tuple(self.config.active_arms)
        self._ee_type: str = str(self.hardware.end_effector_type)
        if self._ee_type == "gripper":
            self._ee_dim = 1
            self._ee_key: Optional[str] = "gripper"
        elif self._ee_type == "hand":
            self._ee_dim = len(R2_HAND_FINGER_ORDER)
            self._ee_key = "hand"
        else:
            self._ee_dim = 0
            self._ee_key = None

        self._num_steps = 0
        self._success_hold_counter = 0
        self._last_ee: Optional[np.ndarray] = None
        self._last_step_end: Optional[float] = None
        self._has_measured_state = False
        self._joints = np.zeros(ARM_DOF * len(self._active_arms))
        self._waist_qpos: Optional[np.ndarray] = None
        self.robot: Optional[R2Robot] = None
        self._state_robot: Optional[PartGroup] = None

        if not self.config.is_dummy:
            self._setup_hardware()

        if not self.hardware.cameras:
            self._logger.info(
                "No R2 cameras configured. Observations will not contain camera frames."
            )

        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        if self.config.reset_on_init:
            self.go_to_rest()
        self._open_cameras()
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _setup_hardware(self) -> None:
        """Compose and connect the configured R2 robot."""
        assert self.env_idx >= 0, "env_idx must be nonnegative."
        hardware = self.hardware
        controller_node_rank = hardware.controller_node_rank
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        self.robot = R2Robot.build(
            body_dof=hardware.body_dof,
            end_effector_type=hardware.end_effector_type,
            enable_base=hardware.enable_base,
            cameras=hardware.cameras,
            enable_depth=hardware.enable_depth,
            use_compressed_camera=hardware.use_compressed_camera,
            controller_mode=hardware.controller_mode,
            enable_fault_recovery=hardware.enable_fault_recovery,
            max_recovery_attempts=hardware.max_recovery_attempts,
            recovery_cooldown_window=hardware.recovery_cooldown_window,
            motion_profile=hardware.motion_profile,
            gripper_min_closure=hardware.gripper_min_closure,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
        )
        self.robot.connect()
        self._state_robot = PartGroup(
            **{
                name: part
                for name, part in self.robot._children.items()
                if not isinstance(part, Camera)
            }
        )

    def _init_action_obs_spaces(self) -> None:
        """Build the joint-space action and observation spaces."""
        self._joint_limit_low = np.asarray(
            self.config.joint_limit_low, dtype=np.float64
        )
        self._joint_limit_high = np.asarray(
            self.config.joint_limit_high, dtype=np.float64
        )

        low_parts: list[np.ndarray] = []
        high_parts: list[np.ndarray] = []
        for idx, _side in enumerate(self._active_arms):
            q_slice = slice(idx * ARM_DOF, (idx + 1) * ARM_DOF)
            low_parts.append(self._joint_limit_low[q_slice])
            high_parts.append(self._joint_limit_high[q_slice])
            if self._ee_dim > 0:
                low_parts.append(np.zeros(self._ee_dim, dtype=np.float64))
                high_parts.append(np.ones(self._ee_dim, dtype=np.float64))

        if self.hardware.enable_base:
            low_parts.append(np.full(3, -1.0, dtype=np.float64))
            high_parts.append(np.full(3, 1.0, dtype=np.float64))

        action_low = np.concatenate(low_parts).astype(np.float32)
        action_high = np.concatenate(high_parts).astype(np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high)

        num_joints = ARM_DOF * len(self._active_arms)
        num_tcp = 6 * len(self._active_arms)
        state_spaces: dict[str, gym.Space] = {
            "arm_joint_position": gym.spaces.Box(
                -np.inf, np.inf, shape=(num_joints,), dtype=np.float32
            ),
            "tcp_pose": gym.spaces.Box(
                -np.inf, np.inf, shape=(num_tcp,), dtype=np.float32
            ),
        }
        if self._ee_dim > 0:
            state_spaces["gripper_position"] = gym.spaces.Box(
                0.0,
                1.0,
                shape=(self._ee_dim * len(self._active_arms),),
                dtype=np.float32,
            )
        if self.hardware.enable_base:
            state_spaces["base_pose"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )
            state_spaces["base_velocity"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            )

        spaces: dict[str, gym.Space] = {"state": gym.spaces.Dict(state_spaces)}
        frames = {
            cam_name: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            for cam_name in (self.hardware.cameras or [])
        }
        if frames:
            spaces["frames"] = gym.spaces.Dict(frames)

        self.observation_space = gym.spaces.Dict(spaces)
        self._base_observation_space = copy.deepcopy(self.observation_space)
        self._action_state = self._default_action_state()

    def _default_action_state(self) -> np.ndarray:
        """Build a within-limits initial action state from ``reset_joint_qpos``."""
        rest_q = np.asarray(self.config.reset_joint_qpos, dtype=np.float32)
        chunks: list[np.ndarray] = []
        for idx, _side in enumerate(self._active_arms):
            q_slice = slice(idx * ARM_DOF, (idx + 1) * ARM_DOF)
            chunks.append(
                np.clip(
                    rest_q[q_slice],
                    self._joint_limit_low[q_slice],
                    self._joint_limit_high[q_slice],
                ).astype(np.float32)
            )
            if self._ee_dim > 0:
                chunks.append(np.zeros(self._ee_dim, dtype=np.float32))
        if self.hardware.enable_base:
            chunks.append(np.zeros(3, dtype=np.float32))
        return np.concatenate(chunks).astype(np.float32)

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return the ordered action parts matching ``self.action_space``."""
        parts: list[ActionPart] = []
        single = len(self._active_arms) == 1
        for side in self._active_arms:
            prefix = "" if single else f"{side}."
            parts.append(ActionPart(f"{prefix}arm", ARM_DOF, ActionKind.JOINT_POSITION))
            if self._ee_dim > 0:
                if self._ee_type == "gripper":
                    parts.append(
                        ActionPart(
                            f"{prefix}end_effector",
                            self._ee_dim,
                            ActionKind.GRIPPER,
                        )
                    )
                else:
                    parts.append(
                        ActionPart(f"{prefix}hand", self._ee_dim, ActionKind.HAND)
                    )
        if self.hardware.enable_base:
            parts.append(ActionPart("base", 3, ActionKind.BASE_VELOCITY))
        return tuple(parts)

    def arm_kinematics(self, side: Optional[str] = None) -> R2KinematicsAdapter:
        """Build the 7-DoF FK/IK model of one R2 arm for PICO teleoperation."""
        resolved_side = (
            self._active_arms[0]
            if (side is None and len(self._active_arms) == 1)
            else side
        )
        if resolved_side not in self._active_arms:
            raise ValueError(
                f"Expected active R2 arm in {self._active_arms}, got {side!r}."
            )
        idx = self._active_arms.index(resolved_side)
        q_slice = slice(idx * ARM_DOF, (idx + 1) * ARM_DOF)
        return R2KinematicsAdapter(
            resolved_side,
            config=self.config.ik,
            joint_lower=self._joint_limit_low[q_slice],
            joint_upper=self._joint_limit_high[q_slice],
            waist_qpos=self._measured_waist_qpos,
        )

    def _measured_waist_qpos(self) -> Optional[np.ndarray]:
        """Return the measured waist for IK, read once per observation."""
        if self._waist_qpos is None and self.robot is not None:
            arm = self.robot.child(self._active_arms[0]).child("arm")
            self._waist_qpos = np.asarray(
                arm.owner.read_state()["waist_positions"], dtype=np.float64
            ).reshape(-1)
        return self._waist_qpos

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one step on the R2 robot."""
        start_time = time.monotonic()
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1),
            self.action_space.low,
            self.action_space.high,
        )

        ee_moved = False
        if not self.config.is_dummy and self.robot is not None:
            robot_action: dict[str, Any] = {}
            offset = 0
            ee_targets: list[np.ndarray] = []
            for idx, side in enumerate(self._active_arms):
                q_slice = slice(idx * ARM_DOF, (idx + 1) * ARM_DOF)
                q_target = np.clip(
                    action[offset : offset + ARM_DOF],
                    self._joint_limit_low[q_slice],
                    self._joint_limit_high[q_slice],
                ).astype(np.float32)
                offset += ARM_DOF

                side_cmd: dict[str, Any] = {
                    "arm": {"joint_position": q_target},
                }
                if self._ee_key is not None and self._ee_dim > 0:
                    ee_target = np.clip(
                        action[offset : offset + self._ee_dim], 0.0, 1.0
                    ).astype(np.float32)
                    offset += self._ee_dim
                    side_cmd[self._ee_key] = {"target": ee_target}
                    ee_targets.append(ee_target)
                robot_action[side] = side_cmd

            if self.hardware.enable_base:
                base_vel = action[offset : offset + 3].astype(np.float32)
                robot_action["base"] = {"base_velocity": base_vel}

            if ee_targets:
                combined_ee = np.concatenate(ee_targets)
                ee_moved = self._last_ee is not None and bool(
                    np.any(
                        np.abs(combined_ee - self._last_ee)
                        > self.config.reward_threshold
                    )
                )
                self._last_ee = combined_ee

            self.robot.send_action(robot_action)
        else:
            self._action_state = action.copy()

        self._num_steps += 1
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, ee_moved)

        terminated = (
            reward >= 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        info: dict[str, Any] = {}
        if not self.config.is_dummy and self.robot is not None:
            recovery_event = self.robot.consume_recovery_event()
            if recovery_event is not None:
                info["robot_fault"] = recovery_event
                if self.config.truncate_on_fault_recovery:
                    truncated = True
                    info["fault_truncated"] = True
        period = 1.0 / self.config.step_frequency
        cycle_start = (
            self._last_step_end
            if self._last_step_end is not None
            and (start_time - self._last_step_end) < 2.0 * period
            else start_time
        )
        step_time = time.monotonic() - cycle_start
        time.sleep(max(0.0, period - step_time))
        self._last_step_end = time.monotonic()
        return observation, reward, terminated, truncated, info

    def get_joint_positions(self) -> np.ndarray:
        """Return measured joints and end-effectors laid out in ``action_parts()`` order."""
        if (
            not hasattr(self, "_action_state")
            or self._action_state.size != self.action_space.shape[0]
            or (
                not getattr(self, "_has_measured_state", False)
                and not self.config.is_dummy
            )
        ):
            self._get_observation()
        return getattr(
            self,
            "_action_state",
            np.zeros(self.action_space.shape, dtype=np.float32),
        ).copy()

    @property
    def num_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._num_steps

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return active R2 arms to their rest configuration."""
        seed_sampled_spaces(seed, self._base_observation_space)
        self._num_steps = 0
        self._success_hold_counter = 0
        self._last_ee = None
        self._last_step_end = None
        if self.config.is_dummy:
            self._action_state = self._default_action_state()
            return self._get_observation(), {}

        self.go_to_rest()
        if self.robot is not None:
            self.robot.consume_recovery_event()
        return self._get_observation(), {}

    def go_to_rest(self) -> None:
        """Move active R2 arms to :pyattr:`R2EnvConfig.reset_joint_qpos`.

        Raises:
            RuntimeError: When the move does not complete, so no episode
                starts from wherever the arms stopped.
        """
        if self.robot is None:
            return
        rest = np.asarray(self.config.reset_joint_qpos, dtype=np.float32)
        with self.robot.batch_commands() as batch:
            for idx, side in enumerate(self._active_arms):
                q_slice = slice(idx * ARM_DOF, (idx + 1) * ARM_DOF)
                arm = self.robot.child(side).child("arm")
                arm.reset_joint(
                    rest[q_slice],
                    duration=self.config.reset_duration,
                    max_velocity=self.config.reset_joint_speed,
                )
        if not batch.ok:
            raise RuntimeError(
                f"R2 did not reach its rest pose {np.round(rest, 3).tolist()}: "
                f"{self.robot.diagnose().error_message or 'the move did not complete'}."
            )

    def _calc_step_reward(
        self, observation: dict[str, Any], ee_moved: bool = False
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

        if ee_moved and self.config.enable_gripper_penalty:
            reward -= self.config.gripper_penalty
        return reward

    def _get_observation(self) -> dict[str, Any]:
        """Return the robot state and any camera frames."""
        if self.config.is_dummy or self.robot is None:
            return self._base_observation_space.sample()

        state_source = (
            self._state_robot if self._state_robot is not None else self.robot
        )
        raw = state_source.get_observation()
        self._waist_qpos = None
        joint_chunks: list[np.ndarray] = []
        tcp_chunks: list[np.ndarray] = []
        ee_chunks: list[np.ndarray] = []
        for side in self._active_arms:
            side_obs = raw[side]
            arm_obs = side_obs["arm"]
            joint_chunks.append(
                np.asarray(arm_obs["joint_position"], dtype=np.float32).reshape(ARM_DOF)
            )
            tcp_chunks.append(
                np.asarray(arm_obs["tcp_pose"], dtype=np.float32).reshape(6)
            )
            if self._ee_key is not None and self._ee_key in side_obs:
                ee_chunks.append(
                    np.asarray(
                        side_obs[self._ee_key]["state"], dtype=np.float32
                    ).reshape(self._ee_dim)
                )

        joints = np.concatenate(joint_chunks)
        self._joints = joints.astype(float)
        action_chunks: list[np.ndarray] = []
        for idx in range(len(self._active_arms)):
            action_chunks.append(joint_chunks[idx])
            if self._ee_dim > 0:
                if idx < len(ee_chunks):
                    action_chunks.append(np.clip(ee_chunks[idx], 0.0, 1.0))
                else:
                    action_chunks.append(np.zeros(self._ee_dim, dtype=np.float32))
        if self.hardware.enable_base:
            action_chunks.append(np.zeros(3, dtype=np.float32))
        self._action_state = np.concatenate(action_chunks).astype(np.float32)
        self._has_measured_state = True
        state_obs: dict[str, Any] = {
            "arm_joint_position": joints,
            "tcp_pose": np.concatenate(tcp_chunks),
        }
        if ee_chunks:
            state_obs["gripper_position"] = np.clip(
                np.concatenate(ee_chunks), 0.0, 1.0
            ).astype(np.float32)
        if self.hardware.enable_base and "base" in raw:
            state_obs["base_pose"] = np.asarray(
                raw["base"]["base_pose"], dtype=np.float32
            ).reshape(3)
            state_obs["base_velocity"] = np.asarray(
                raw["base"]["base_velocity"], dtype=np.float32
            ).reshape(3)

        observation: dict[str, Any] = {"state": state_obs}
        if "frames" in self.observation_space.spaces:
            observation["frames"] = self._get_camera_frames()
        return observation

    def _open_cameras(self) -> None:
        """Collect the camera views composed into ``self.robot``."""
        if self.robot is None:
            self._cameras: list[BaseCamera] = []
            return
        self._cameras = list(self.robot.parts_of_type(Camera).values())

    def _close_cameras(self) -> None:
        """Drop camera references; ``self.robot`` owns their lifecycle."""
        self._cameras = []

    def _crop_frame(self, frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Centre-crop to a square, then resize to the declared frame size."""
        height, width, _ = frame.shape
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return cv2.resize(frame[top : top + side, left : left + side], size)

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Read one frame per camera."""
        declared = self.observation_space["frames"]
        frames: dict[str, np.ndarray] = {}
        for camera in getattr(self, "_cameras", []):
            name = camera.name
            if name not in declared.spaces:
                continue
            size = declared[name].shape[:2][::-1]
            frames[name] = self._crop_frame(
                camera.get_frame(max_age=self.config.camera_max_age), size
            )

        if hasattr(self, "camera_player"):
            self.camera_player.put_frame(frames)
        return frames

    def close(self) -> None:
        """Release cameras and disconnect the R2 robot."""
        if hasattr(self, "_cameras"):
            self._close_cameras()
        if hasattr(self, "camera_player"):
            self.camera_player.stop()
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
