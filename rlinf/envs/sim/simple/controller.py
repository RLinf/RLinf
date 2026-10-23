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

"""In-process SIMPLE System-0 controller for high-level Psi0 actions."""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

import numpy as np
import torch

from rlinf.envs.sim.simple.utils import psi0_upper_joint_values


class _SimulationClock:
    """Stand-in for the ``time`` module whose monotonic clock follows simulation.

    The decoupled WBC interpolates upper-body and navigation goals by
    ``target_time`` and times out teleop goals older than one second. Both
    assume one 50 Hz control step per 20 ms of wall time, as in real-time
    teleoperation. Advancing by simulated control steps keeps that contract
    regardless of how fast this process can render.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _install_simulation_clock() -> _SimulationClock:
    """Route the WBC agent and policy clocks through simulated time.

    SimpleProcess owns exactly one controller, so the process-wide patch only
    affects that controller's upstream modules.
    """
    import decoupled_wbc.control.policy.g1_decoupled_whole_body_policy as policy
    import simple.agents.sonic_decoupled_wbc_agent as agent

    clock = agent.time
    if not isinstance(clock, _SimulationClock):
        clock = _SimulationClock()
        agent.time = clock
    policy.time_module = clock
    return clock


class SimpleController(Protocol):
    """Controller boundary owned by a SIMPLE runtime process."""

    last_base_height: float
    control_dt: float

    def reset(self) -> None: ...

    def stabilize(self, observation: dict[str, Any]) -> Any: ...

    def is_stabilized(self) -> bool: ...

    def finish_stabilization(self) -> None: ...

    def action(
        self, observation: dict[str, Any], high_level_action: np.ndarray
    ) -> Any: ...


def _load_wbc_onnx_policy(
    _policy: Any, model_path: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Load a WBC policy without ORT's automatic CPU affinity or thread pools."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Small control policies run per simulator. One thread avoids competing
    # pools and affinity assignments outside a container's allowed CPU set.
    options.intra_op_num_threads = 1
    model = ort.InferenceSession(model_path, sess_options=options)

    def run_inference(input_tensor: torch.Tensor) -> torch.Tensor:
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cpu")

    return run_inference


class SimpleTeleopController:
    """Drive the fixed SIMPLE decoupled-WBC runtime without an HTTP agent."""

    def __init__(self, robot: Any, sonic_config: dict[str, Any]) -> None:
        from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy
        from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

        self.robot = robot
        # Install before construction: the WBC policy stamps its last goal time.
        self._clock = _install_simulation_clock()
        # The pinned upstream factory has no session-options argument. Override
        # only its WBC loader during this synchronous construction, then restore
        # it even on failure. ONNX Runtime's global API remains unchanged.
        original_loader = G1GearWbcPolicy.load_onnx_policy
        try:
            G1GearWbcPolicy.load_onnx_policy = _load_wbc_onnx_policy
            self._agent = SonicDecoupledWbcAgent(robot, sonic_config=sonic_config)
        finally:
            G1GearWbcPolicy.load_onnx_policy = original_loader
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
        action = self._agent.get_stabilize_action(observation)
        self._clock.now += self.control_dt
        return action

    def is_stabilized(self) -> bool:
        return bool(self.robot.stabilized)

    def finish_stabilization(self) -> None:
        """Match SIMPLE's policy-start gait phase."""
        self._agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros(
            (1,), dtype=torch.float32
        )

    def action(self, observation: dict[str, Any], high_level_action: np.ndarray) -> Any:
        """Convert one high-level command to one frozen decoupled-WBC action."""
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
        now = self._clock.monotonic()
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
        self._clock.now += self.control_dt
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
