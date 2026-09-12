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

"""How a Piper is driven and what its policy sees, for any task on it.

A Piper takes absolute joint targets and, when a gripper is fitted, an opening
in ``0..1`` on the same CAN session. It reports its six joints and the tool pose
its own controller solves for, so a task may score either.
"""

from rlinf.envs.real.policy import (
    ActionLayout,
    ContinuousGripper,
    JointActionConfig,
    JointPositions,
    ObservationSpec,
    Source,
    StateKey,
)
from rlinf.envs.real.task_env import RegisteredTaskEnv
from rlinf.robotics import PiperConfig, PiperRobot
from rlinf.robotics.parts.arms.piper import PiperArm
from rlinf.robotics.parts.cameras import CameraInfo


class PiperEnv(RegisteredTaskEnv):
    """A task on a Piper, driven by absolute joint targets."""

    ROBOT = PiperRobot
    ACTION_CONFIG = JointActionConfig
    DEFAULTS = {
        "joint_limit_low": tuple(PiperArm.JOINT_LIMITS_LOWER),
        "joint_limit_high": tuple(PiperArm.JOINT_LIMITS_UPPER),
    }

    @classmethod
    def make_action(
        cls,
        hardware: PiperConfig,
        config: JointActionConfig,
        options: None = None,
        roles: tuple[str, ...] = ("arm",),
    ) -> ActionLayout:
        """Six joints, then the gripper's opening when one is fitted."""
        channels = [
            JointPositions(
                "arm",
                low=config.joint_limit_low,
                high=config.joint_limit_high,
                dof=PiperArm.DOF,
            )
        ]
        if hardware.with_gripper:
            channels.append(ContinuousGripper("arm"))
        return ActionLayout(channels)

    @classmethod
    def make_observation(
        cls,
        hardware: PiperConfig,
        cameras: tuple[CameraInfo, ...],
        roles: tuple[str, ...] = ("arm",),
    ) -> ObservationSpec:
        """Joints and tool pose, the gripper's opening, and the cameras."""
        state = [
            StateKey(
                "arm_joint_position",
                (PiperArm.DOF,),
                (Source("arm_joint_position"),),
            ),
            StateKey("tcp_pose", (7,), (Source("tcp_pose"),)),
        ]
        if hardware.with_gripper:
            state.append(
                StateKey(
                    "gripper_position",
                    (1,),
                    (Source("state", end_effector=True),),
                    low=0.0,
                    high=1.0,
                )
            )
        return ObservationSpec(tuple(state), cameras=cameras)
