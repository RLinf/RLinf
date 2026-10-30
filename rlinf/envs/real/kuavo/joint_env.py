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

"""Dual-arm Kuavo joint control with ROS camera observations."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.robotics import KuavoRobot, KuavoRobotConfig, RobotInfo
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.adapters import (
    LegacyObservationAdapter,
    VectorActionAdapter,
    VectorActionBinding,
)
from rlinf.robotics.parts.arms.kuavo import KuavoConnection
from rlinf.robotics.parts.base import PartGroup
from rlinf.robotics.parts.cameras import Camera, ROSCameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

_SIDES = ("left", "right")
_CAMERA_NAMES = ("head_cam_h", "wrist_cam_l", "wrist_cam_r")
_STATE_FIELDS = {
    "left_arm": ("left", "arm", "arm_joint_position"),
    "left_gripper": ("left", "gripper", "state"),
    "right_arm": ("right", "arm", "arm_joint_position"),
    "right_gripper": ("right", "gripper", "state"),
}
_JOINT_INDICES = np.asarray([*range(7), *range(8, 15)])


@dataclass
class KuavoJointEnvConfig:
    """Task settings; device limits and ROS topics come from hardware config."""

    is_dummy: bool = False
    task_description: str = "dual-arm Kuavo manipulation task"
    control_mode: str = "joint"
    step_frequency: float | None = None
    max_num_steps: int = 1000
    max_joint_delta: float = 0.05
    manual_episode_control_only: bool = False
    camera_frame_timeout_s: float = 1.0
    camera_stale_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid timing and command-bound settings before connection."""
        for name in ("is_dummy", "manual_episode_control_only"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Kuavo {name} must be a bool.")
        if self.control_mode != "joint":
            raise ValueError("KuavoJointEnv requires control_mode='joint'.")
        if (
            not isinstance(self.max_num_steps, int)
            or isinstance(self.max_num_steps, bool)
            or self.max_num_steps <= 0
        ):
            raise ValueError("Kuavo max_num_steps must be a positive integer.")
        for name in (
            "step_frequency",
            "max_joint_delta",
            "camera_frame_timeout_s",
            "camera_stale_timeout_s",
        ):
            value = getattr(self, name)
            if name == "step_frequency" and value is None:
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Kuavo {name} must be finite and positive.")


class KuavoJointEnv(gym.Env):
    """Expose absolute joints, normalized grippers and three RGB cameras.

    Action and state order is ``[left_q0..q6, left_gripper, right_q0..q6,
    right_gripper]``. Reset only reads the current pose; it does not home the
    arms. Rewards remain zero until an operator scores an episode through the
    shared keyboard wrapper. Joint delta clipping is a per-step command bound,
    not an additional controller or a high-frequency control loop.
    """

    metadata = {"render_modes": []}
    TELEOP: tuple[str, ...] = ()
    TELEOP_DEFAULT = "none"

    def __init__(
        self,
        override_cfg: dict[str, Any] | None,
        worker_info: WorkerInfo | None = None,
        robot_info: RobotInfo[KuavoRobotConfig] | None = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = KuavoJointEnvConfig(**dict(override_cfg or {}))
        self.hardware = get_hardware_config(
            KuavoRobotConfig, robot_info, is_dummy=self.config.is_dummy
        )
        if self.hardware.which_arm != "both":
            raise ValueError("KuavoJointEnv requires hardware.which_arm='both'.")
        if self.hardware.arm_state_keys != ["joint_q", "gripper"]:
            raise ValueError(
                "KuavoJointEnv requires hardware.arm_state_keys=['joint_q', 'gripper']."
            )
        routes = self.hardware.active_obs_key_map()
        gripper_message = (
            "lejuClawState"
            if self.hardware.end_effector_type == "leju_claw"
            else "JointState"
        )
        for name, expected in (
            ("joint_q", "sensorsData"),
            ("gripper", gripper_message),
        ):
            if routes[name]["msg_type"] != expected:
                raise ValueError(
                    f"Kuavo {name} requires msg_type={expected!r}, "
                    f"got {routes[name]['msg_type']!r}."
                )
        self.env_idx = env_idx
        self.node_rank = None if worker_info is None else self.hardware.node_rank
        self.controller_node_rank = (
            None if worker_info is None else self.hardware.controller_node_rank
        )
        self.env_worker_rank = 0 if worker_info is None else worker_info.rank
        self.step_frequency = (
            self.hardware.ros_rate
            if self.config.step_frequency is None
            else self.config.step_frequency
        )
        self._logger = get_logger()
        self.robot: KuavoRobot | None = None
        self._arms: PartGroup | None = None
        self._started = False
        self._start_failed = False
        self._faulted = False
        self._closed = False
        self._num_steps = 0
        self._last_tick_s: float | None = None
        self._state_adapter = LegacyObservationAdapter(_STATE_FIELDS, {})
        self._observation_adapter = LegacyObservationAdapter(
            _STATE_FIELDS,
            {name: ("cameras", name, "frame") for name in _CAMERA_NAMES},
        )
        self._action_adapter = VectorActionAdapter(
            16,
            [
                VectorActionBinding(("left", "arm", "arm_joint_position"), 0, 7),
                VectorActionBinding(("left", "gripper", "target"), 7, 8),
                VectorActionBinding(("right", "arm", "arm_joint_position"), 8, 15),
                VectorActionBinding(("right", "gripper", "target"), 15, 16),
            ],
        )
        self._action_low = self._pack_limits(
            self.hardware.joint_limits_min, self.hardware.end_effector_limits_min
        )
        self._action_high = self._pack_limits(
            self.hardware.joint_limits_max, self.hardware.end_effector_limits_max
        )
        self.action_space = gym.spaces.Box(
            self._action_low.astype(np.float32),
            self._action_high.astype(np.float32),
            dtype=np.float32,
        )
        width, height = self.hardware.image_size
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(16,), dtype=np.float32
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(height, width, 3), dtype=np.uint8
                        )
                        for name in _CAMERA_NAMES
                    }
                ),
            }
        )
        self._dummy_state = np.clip(np.zeros(16), self._action_low, self._action_high)

    @property
    def task_description(self) -> str:
        """Return the task prompt consumed by ``RealWorldEnv``."""
        return self.config.task_description

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Describe the shared wrapper layout of the 16-D action vector."""
        from rlinf.envs.real.wrappers.teleop.layout import mirrored

        return mirrored(
            (
                ActionPart("arm", 7, ActionKind.JOINT_POSITION),
                ActionPart("end_effector", 1, ActionKind.GRIPPER),
            ),
            _SIDES,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Open the configured station and read its pose without moving arms."""
        super().reset(seed=seed)
        self._ensure_started()
        self._num_steps = 0
        self._last_tick_s = None
        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Validate all 16 targets, bound them, and dispatch one hardware step."""
        target = np.asarray(action, dtype=np.float64)
        if target.shape != (16,) or not np.all(np.isfinite(target)):
            raise ValueError("Kuavo action must contain exactly 16 finite values.")
        self._ensure_started()
        self._pace()
        try:
            # Verify camera freshness and measured joints before dispatching motion.
            measured = self._get_observation()["state"]["joint_position"].astype(
                np.float64
            )
            joints = measured[_JOINT_INDICES]
            if np.any(joints < self.action_space.low[_JOINT_INDICES]) or np.any(
                joints > self.action_space.high[_JOINT_INDICES]
            ):
                raise ValueError(
                    "Measured Kuavo joints are outside the configured hardware limits; "
                    "refusing to move them automatically."
                )
            accepted = np.clip(target, self._action_low, self._action_high)
            accepted[_JOINT_INDICES] = np.clip(
                accepted[_JOINT_INDICES],
                joints - self.config.max_joint_delta,
                joints + self.config.max_joint_delta,
            )
            if self.config.is_dummy:
                self._dummy_state = accepted.copy()
            else:
                self._connection().begin_step()
                self._arms.send_action(self._action_adapter.adapt(accepted))
            observation = self._get_observation()
        except Exception:
            self._faulted = True
            if not self.config.is_dummy:
                try:
                    self._connection().hold_arms()
                except Exception as hold_error:
                    self._logger.error(
                        "Could not hold Kuavo arms after a fault: %s", hold_error
                    )
            raise
        self._num_steps += 1
        truncated = bool(
            not self.config.manual_episode_control_only
            and self._num_steps >= self.config.max_num_steps
        )
        return (
            observation,
            0.0,
            False,
            truncated,
            {
                "accepted_action": accepted.astype(np.float32),
                "action_clipped": bool(np.any(accepted != target)),
            },
        )

    def get_hold_action(self, fallback_action: Any = None) -> np.ndarray:
        """Return measured joints and grippers, never a zero-action fallback."""
        del fallback_action
        self._ensure_started()
        if self.config.is_dummy:
            return self._dummy_state.astype(np.float32)
        state = self._state_adapter.adapt(self._arms.get_observation())["state"]
        return self._pack_state(state)

    def observe(self) -> dict[str, Any]:
        """Read current state and cameras without sending an action."""
        self._ensure_started()
        return self._get_observation()

    def close(self) -> None:
        """Release connections without commanding a park or home pose."""
        if self._closed:
            return
        try:
            self._disconnect()
        finally:
            self._started = False
            self._closed = True

    @staticmethod
    def _pack_limits(joints: list[float], grippers: list[float]) -> np.ndarray:
        return np.asarray(
            [*joints[:7], grippers[0], *joints[7:], grippers[1]], dtype=np.float64
        )

    @staticmethod
    def _pack_state(state: Mapping[str, Any]) -> np.ndarray:
        parts = []
        for name, width in zip(_STATE_FIELDS, (7, 1, 7, 1), strict=True):
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != (width,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Kuavo {name} must contain {width} finite values.")
            parts.append(value)
        return np.concatenate(parts)

    def _camera_infos(self) -> dict[str, ROSCameraInfo]:
        routes = self.hardware.active_obs_key_map()
        return {
            name: ROSCameraInfo(
                name=name,
                serial_number=routes[name]["topic"],
                message_type=routes[name]["msg_type"],
                resolution=tuple(self.hardware.image_size),
                fps=routes[name]["frequency"],
                frame_timeout_s=self.config.camera_frame_timeout_s,
                stale_timeout_s=self.config.camera_stale_timeout_s,
            )
            for name in _CAMERA_NAMES
        }

    def _connect(self) -> None:
        hardware = self.hardware
        routes = hardware.active_obs_key_map()
        self.robot = KuavoRobot.build(
            which_arm=hardware.which_arm,
            end_effector_type=hardware.end_effector_type,
            head_position=hardware.head_position,
            joint_limits_min=hardware.joint_limits_min,
            joint_limits_max=hardware.joint_limits_max,
            end_effector_limits_min=hardware.end_effector_limits_min,
            end_effector_limits_max=hardware.end_effector_limits_max,
            feedback_topics={
                name: routes[name]["topic"] for name in ("joint_q", "gripper")
            },
            joint_state_start=hardware.joint_q_slices[0][0],
            feedback_timeout_s=hardware.feedback_timeout_s,
            feedback_stale_timeout_s=hardware.feedback_stale_timeout_s,
            cameras=self._camera_infos(),
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            controller_node_rank=self.controller_node_rank,
            worker_rank=self.env_worker_rank,
        )
        for camera in self.robot.parts_of_type(Camera).values():
            camera.connect()
            camera.get_observation()
        self.robot.connect()
        self._arms = PartGroup({side: self.robot.child(side) for side in _SIDES})
        self._get_observation()

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot use a closed KuavoJointEnv.")
        if self._start_failed:
            raise RuntimeError("KuavoJointEnv startup failed; close and rebuild it.")
        if self._faulted:
            raise RuntimeError(
                "KuavoJointEnv has a hardware fault; close and rebuild it."
            )
        if self._started:
            return
        try:
            if not self.config.is_dummy:
                self._connect()
            self._started = True
        except Exception as start_error:
            self._start_failed = True
            try:
                self._disconnect()
            except Exception as cleanup_error:
                self._logger.error(
                    "Kuavo startup failed (%s), and cleanup failed: %s",
                    start_error,
                    cleanup_error,
                )
            raise

    def _connection(self) -> KuavoConnection:
        if self.robot is None:
            raise RuntimeError("Kuavo robot is not connected.")
        owner = self.robot.child("left").child("arm").owner
        if not isinstance(owner, KuavoConnection):
            raise TypeError("Kuavo arm must belong to a KuavoConnection.")
        return owner

    def _disconnect(self) -> None:
        robot, self.robot = self.robot, None
        self._arms = None
        if robot is not None:
            robot.disconnect()

    def _get_observation(self) -> dict[str, Any]:
        if self.config.is_dummy:
            width, height = self.hardware.image_size
            return {
                "state": {"joint_position": self._dummy_state.astype(np.float32)},
                "frames": {
                    name: np.zeros((height, width, 3), dtype=np.uint8)
                    for name in _CAMERA_NAMES
                },
            }
        observation = self._observation_adapter.adapt(self.robot.get_observation())
        return {
            "state": {"joint_position": self._pack_state(observation["state"])},
            "frames": {
                name: np.ascontiguousarray(frame[..., ::-1])
                for name, frame in observation["frames"].items()
            },
        }

    def _pace(self) -> None:
        if self._last_tick_s is not None:
            remaining = 1.0 / self.step_frequency - (
                time.monotonic() - self._last_tick_s
            )
            if remaining > 0:
                time.sleep(remaining)
        self._last_tick_s = time.monotonic()
