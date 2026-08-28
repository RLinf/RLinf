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

"""In-process SIMPLE System-0 adapters for high-level Psi0 actions."""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import torch

_STATE_SLICES = (
    (29, 32),
    (34, 36),
    (32, 34),
    (36, 43),
    (15, 22),
    (22, 29),
)


def extract_psi0_state(joint_qpos: np.ndarray, base_height: float) -> np.ndarray:
    """Build the official 32-D SIMPLE Teleop proprioceptive state."""
    joint_qpos = np.asarray(joint_qpos)
    if joint_qpos.ndim != 1 or joint_qpos.shape[0] < 43:
        raise ValueError(
            "SIMPLE joint_qpos must be one-dimensional with at least 43 joints."
        )
    waist_rpy = joint_qpos[[13, 14, 12]]
    state = np.concatenate(
        [joint_qpos[start:end] for start, end in _STATE_SLICES]
        + [waist_rpy, np.asarray([base_height])]
    )
    if state.shape != (32,):
        raise RuntimeError(f"Unexpected Psi0 SIMPLE state shape: {state.shape}.")
    return state.astype(np.float32, copy=False)


def psi0_upper_joint_values(action: np.ndarray) -> np.ndarray:
    """Reorder Psi0's first 28 values into SIMPLE's upper-joint order."""
    action = np.asarray(action)
    if action.shape != (36,):
        raise ValueError(
            f"Psi0 SIMPLE action must have shape (36,), got {action.shape}."
        )
    return np.concatenate(
        (
            action[14:28],
            action[0:3],
            action[5:7],
            action[3:5],
            action[7:14],
        )
    ).astype(np.float32, copy=False)


class SimpleController(Protocol):
    """Controller boundary owned by a SIMPLE EnvWorker process."""

    last_base_height: float
    control_dt: float

    def reset(self) -> None: ...

    def stabilize(self, observation: dict[str, Any]) -> Any: ...

    def is_stabilized(self) -> bool: ...

    def finish_stabilization(self) -> None: ...

    def action(
        self, observation: dict[str, Any], high_level_action: np.ndarray
    ) -> Any: ...


class SimpleTeleopController:
    """Drive the fixed SIMPLE decoupled-WBC runtime without an HTTP agent."""

    def __init__(self, robot: Any, sonic_config: dict[str, Any]) -> None:
        from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

        self.robot = robot
        self._agent = SonicDecoupledWbcAgent(robot, sonic_config=sonic_config)
        indices = self._agent._dwbc_robot_model.get_joint_group_indices("upper_body")
        self._upper_joint_names = [
            name
            for name, index in self._agent._dwbc_robot_model.joint_to_dof_index.items()
            if index in indices
        ]
        self.last_base_height = 0.74
        self.control_dt = 4 * float(self._agent.sim_dt)

    def reset(self) -> None:
        self._agent.reset()
        self._agent._wbc_policy.lower_body_policy.use_policy_action = True
        self.last_base_height = 0.74

    def stabilize(self, observation: dict[str, Any]) -> Any:
        return self._agent.get_stabilize_action(observation)

    def is_stabilized(self) -> bool:
        return bool(self.robot.stabilized)

    def finish_stabilization(self) -> None:
        """Match SIMPLE's policy-start gait phase."""
        self._agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(
            (1,), dtype=torch.float32
        )

    def action(self, observation: dict[str, Any], high_level_action: np.ndarray) -> Any:
        """Convert one 36-D command to one frozen decoupled-WBC action."""
        from simple.core.action import ActionCmd

        command = np.asarray(high_level_action, dtype=np.float32)
        upper_values = psi0_upper_joint_values(command)
        target_upper_body_pose = dict(zip(self.robot.joint_names[15:], upper_values))
        target_upper_body_pose.update(
            {
                "waist_yaw_joint": command[30],
                "waist_roll_joint": command[28],
                "waist_pitch_joint": command[29],
            }
        )

        proprio = self.robot.prepare_obs()
        wbc_obs = self._agent._build_wbc_observation(proprio)
        self._agent._wbc_policy.set_observation(wbc_obs)
        now = time.monotonic()
        control_frequency = self._agent._control_frequency
        goal = {
            "target_upper_body_pose": np.asarray(
                [target_upper_body_pose[name] for name in self._upper_joint_names],
                dtype=np.float32,
            ),
            "navigate_cmd": command[32:36],
            "base_height_command": command[31:32],
            "target_time": now + 1 / control_frequency,
            "interpolation_garbage_collection_time": now - 2 / control_frequency,
            "timestamp": now,
        }
        self._agent._wbc_policy.set_goal(goal)
        wbc_action = self._agent._wbc_policy.get_action(time=now)
        robot_model = self._agent._dwbc_robot_model
        self.last_base_height = float(command[31])
        return ActionCmd(
            "decoupled_wbc",
            target_q=robot_model.get_body_actuated_joints(wbc_action["q"]),
            left_hand_q=robot_model.get_hand_actuated_joints(
                wbc_action["q"], side="left"
            ),
            right_hand_q=robot_model.get_hand_actuated_joints(
                wbc_action["q"], side="right"
            ),
        )
