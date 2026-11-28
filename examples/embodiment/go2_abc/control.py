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

"""Observation and DC-motor contracts for the 235-input Go2 policies."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    """Validate an external sensor or policy vector."""
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {name} as {size} finite values, got {result}")
    return result


@dataclass(frozen=True)
class JointState:
    """SDK motor order, IMU body frame, and local monotonic reception time."""

    q: list[float]
    dq: list[float]
    quaternion_wxyz: list[float]
    gyroscope_b: list[float]
    timestamp: float


@dataclass(frozen=True)
class Perception:
    """Measured body velocity and terrain samples in the training scan order.

    reference_z_w and ground_z_w share a world frame. The reference is the
    robot base height, excluding the ray origin's 20-metre casting offset.
    timestamp uses the same monotonic clock as JointState.timestamp.
    """

    base_lin_vel_b: list[float]
    reference_z_w: float
    ground_z_w: list[float]
    timestamp: float


class Go2Control:
    """Run deterministic policy inference at 50 Hz and motor control at 200 Hz.

    Args:
        checkpoint: RLinf actor weights or an RSL-RL Go2 checkpoint.
        task: A, B, or C, selecting the policy action bounds.
    """

    def __init__(self, checkpoint: Path, task: str) -> None:
        import torch

        from rlinf.models.embodiment.go2_policy import Go2Policy

        if task not in "ABC" or len(task) != 1:
            raise ValueError(f"Expected task A, B, or C, got {task!r}")
        self.spec = json.loads(Path(__file__).with_name("control.json").read_text())
        self.sdk_indices = np.array(
            [
                self.spec["sdk_joint_names"].index(name)
                for name in self.spec["policy_joint_names"]
            ]
        )
        self.default_q = vector(
            self.spec["default_joint_positions"], 12, "default joint positions"
        )
        self.model = Go2Policy(
            str(checkpoint), distribution_safety_patch=task == "B"
        ).eval()
        self.task = task
        self.last_action = np.zeros(12, dtype=np.float32)
        self.target_q = self.default_q.copy()
        self.steps = 0
        self.last_time = None
        self.torch = torch

    @staticmethod
    def _fresh(timestamp: float, now: float, limit: float, name: str) -> None:
        age = now - timestamp
        if not math.isfinite(age) or not 0 <= age <= limit:
            raise TimeoutError(
                f"Expected fresh {name} within {limit}s, found age {age}s"
            )

    def observation(
        self,
        state: JointState,
        perception: Perception,
        command: Sequence[float] | np.ndarray,
        now: float,
    ) -> np.ndarray:
        """Compose the eight unscaled observation terms without training noise."""
        self._fresh(state.timestamp, now, self.spec["state_timeout"], "joint state")
        self._fresh(
            perception.timestamp,
            now,
            self.spec["perception_timeout"],
            "terrain and velocity",
        )
        q = vector(state.q, 12, "SDK joint positions")[self.sdk_indices]
        dq = vector(state.dq, 12, "SDK joint velocities")[self.sdk_indices]
        quaternion = vector(state.quaternion_wxyz, 4, "wxyz quaternion")
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
            raise ValueError("Expected a unit base-to-world quaternion")
        w, x, y, z = quaternion
        gravity = np.array(
            [
                2 * (w * y - x * z),
                -2 * (y * z + w * x),
                -(w * w - x * x - y * y + z * z),
            ],
            dtype=np.float32,
        )
        if not math.isfinite(perception.reference_z_w):
            raise ValueError("Expected a finite terrain height reference")
        height = np.clip(
            perception.reference_z_w
            - vector(perception.ground_z_w, 187, "terrain samples")
            - self.spec["height_offset"],
            *self.spec["height_clip"],
        )
        return np.concatenate(
            (
                vector(perception.base_lin_vel_b, 3, "body velocity"),
                vector(state.gyroscope_b, 3, "body angular velocity"),
                gravity,
                vector(command, 3, "velocity command"),
                q - self.default_q,
                dq,
                self.last_action,
                height,
            )
        )

    def step(
        self,
        state: JointState,
        perception: Perception,
        command: Sequence[float] | np.ndarray,
        now: float,
    ) -> dict[str, np.ndarray]:
        """Return SDK-ordered torques using the DC-motor torque-speed envelope."""
        if (
            self.last_time is not None
            and not 0 < now - self.last_time <= 2 * self.spec["physics_dt"]
        ):
            raise TimeoutError(
                "Expected advancing control samples within two physics steps"
            )
        obs = self.observation(state, perception, command, now)
        if self.steps % self.spec["policy_decimation"] == 0:
            with self.torch.inference_mode():
                action = self.model.actor(self.torch.from_numpy(obs)[None]).numpy()[0]
            action = vector(action, 12, "policy action")
            self.last_action = (
                np.clip(action, -100, 100) if self.task == "B" else action.copy()
            )
            self.target_q = (
                self.default_q + self.spec["action_scale"] * self.last_action
            )
        q = vector(state.q, 12, "SDK joint positions")[self.sdk_indices]
        dq = vector(state.dq, 12, "SDK joint velocities")[self.sdk_indices]
        effort = (
            self.spec["stiffness"] * (self.target_q - q) - self.spec["damping"] * dq
        )
        speed = self.spec["velocity_limit"]
        stall, limit = self.spec["saturation_effort"], self.spec["effort_limit"]
        velocity = np.clip(
            dq, -speed * (1 + limit / stall), speed * (1 + limit / stall)
        )
        torque = np.clip(
            effort,
            np.maximum(stall * (-1 - velocity / speed), -limit),
            np.minimum(stall * (1 - velocity / speed), limit),
        )
        sdk_torque = np.empty(12, dtype=np.float32)
        sdk_torque[self.sdk_indices] = torque
        sdk_target = np.empty(12, dtype=np.float32)
        sdk_target[self.sdk_indices] = self.target_q
        self.last_time = now
        self.steps += 1
        return {
            "observation": obs,
            "action": self.last_action.copy(),
            "target_q_sdk": sdk_target,
            "torque_sdk": sdk_torque,
        }
