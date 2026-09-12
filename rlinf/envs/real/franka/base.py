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

"""How a single-arm Franka is driven and what its policy sees, for any task.

A Franka moves its tool by Cartesian deltas under an impedance controller,
with a Franka Hand, another gripper, or a dexterous hand beside the arm on its
own connection. A policy reads the tool pose and twist, the wrench at the
tool, the gripper's opening or the fingers' pose, and at least one camera.
"""

from typing import Any

from rlinf.envs.real.policy import (
    ActionLayout,
    BinaryGripper,
    HandCommand,
    ObservationSpec,
    Phase,
    PoseActionConfig,
    PoseDelta,
    Source,
    StateKey,
)
from rlinf.envs.real.task_env import RegisteredTaskEnv
from rlinf.envs.real.tasks import CartesianTarget
from rlinf.robotics import FrankaConfig, FrankaRobot
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.robotics.parts.end_effectors import EndEffector

#: Default Cartesian impedance gains shared by Franka tasks.
COMPLIANCE_DEFAULTS: dict[str, float] = {
    "translational_stiffness": 1000,
    "translational_damping": 89,
    "rotational_stiffness": 150,
    "rotational_damping": 7,
    "translational_Ki": 0,
    "rotational_Ki": 0,
    "translational_clip_x": 0.003,
    "translational_clip_y": 0.003,
    "translational_clip_z": 0.01,
    "translational_clip_neg_x": 0.003,
    "translational_clip_neg_y": 0.003,
    "translational_clip_neg_z": 0.01,
    "rotational_clip_x": 0.02,
    "rotational_clip_y": 0.02,
    "rotational_clip_z": 0.02,
    "rotational_clip_neg_x": 0.02,
    "rotational_clip_neg_y": 0.02,
    "rotational_clip_neg_z": 0.02,
}


def compliance(**overrides: float) -> dict[str, float]:
    """Return :data:`COMPLIANCE_DEFAULTS` with ``overrides`` applied.

    Raises:
        KeyError: If an override is not a supported controller gain.
    """
    unknown = set(overrides) - set(COMPLIANCE_DEFAULTS)
    if unknown:
        raise KeyError(
            f"Unknown compliance gains {sorted(unknown)}. "
            f"Known: {sorted(COMPLIANCE_DEFAULTS)}."
        )
    return {**COMPLIANCE_DEFAULTS, **overrides}


class FrankaEnv(RegisteredTaskEnv):
    """A task on a single-arm Franka, driven by Cartesian deltas.

    On its own it runs :class:`~rlinf.envs.real.tasks.CartesianTarget`, which
    ``FrankaEnv-v1`` registers.
    """

    ROBOT = FrankaRobot
    ACTION_CONFIG = PoseActionConfig
    TASK = CartesianTarget
    TELEOP = ("spacemouse", "gello", "glove", "pico")
    TELEOP_DEFAULT = "spacemouse"
    DEFAULTS = {"joint_reset_qpos": (0.0, 0.0, 0.0, -1.9, -0.0, 2.0, 0.0)}
    RETIRED = {
        "hand_target_state": "No task scores the hand's pose.",
        "save_video_path": "Record episodes with the env's video_cfg instead.",
        "add_gripper_penalty": "Use enable_gripper_penalty.",
    }

    @classmethod
    def end_effector_class(cls, hardware: FrankaConfig) -> type[EndEffector]:
        """The end effector the hardware fits, checked for an action layout.

        Raises:
            ValueError: If the part is not exactly one of a gripper or a hand,
                or is a gripper that is not driven by one value.
        """
        part = FrankaRobot.end_effector_class(
            backend=hardware.backend,
            gripper_type=hardware.gripper_type,
            end_effector_type=hardware.end_effector_type,
        )
        if part.is_hand == part.is_gripper:
            raise ValueError(
                "FrankaEnv requires an end effector declaring exactly one of "
                "is_hand or is_gripper. Other tools need a task-specific action "
                "layout."
            )
        if part.is_gripper and (part.action_dim != 1 or part.state_dim != 1):
            raise ValueError("FrankaEnv requires scalar gripper actions and state.")
        return part

    @classmethod
    def make_action(
        cls,
        hardware: FrankaConfig,
        config: PoseActionConfig,
        options: None = None,
        roles: tuple[str, ...] = ("arm",),
    ) -> ActionLayout:
        """Tool deltas, then one gripper channel or one channel per finger."""
        part = cls.end_effector_class(hardware)
        if part.is_hand:
            end_effector: Any = HandCommand(
                "arm",
                dim=part.action_dim,
                scale=config.hand_action_scale,
                max_delta=config.hand_max_delta_per_step,
                reset_state=config.hand_reset_state,
            )
        else:
            # The grasp closes before the arm moves away from it.
            end_effector = BinaryGripper(
                "arm",
                threshold=config.binary_gripper_threshold,
                scale=float(config.action_scale[2]),
                phase=Phase.BEFORE,
            )
        return ActionLayout(
            (
                PoseDelta(
                    "arm",
                    scales=config.action_scale,
                    compliance=config.compliance_param,
                ),
                end_effector,
            ),
            wrappers=("GripperCloseEnv",),
            transforms=("RelativeFrame", "Quat2EulerWrapper"),
        )

    @classmethod
    def make_observation(
        cls,
        hardware: FrankaConfig,
        cameras: tuple[CameraInfo, ...],
        roles: tuple[str, ...] = ("arm",),
    ) -> ObservationSpec:
        """Tool pose, twist and wrench, the end effector, and the cameras."""
        part = cls.end_effector_class(hardware)
        if part.is_hand:
            effector = StateKey(
                "hand_position",
                (part.state_dim,),
                (Source("state", end_effector=True),),
                low=0.0,
                high=1.0,
            )
        else:
            effector = StateKey(
                "gripper_position",
                (1,),
                (Source("state", end_effector=True),),
                low=-1.0,
                high=1.0,
            )
        state = (
            StateKey("tcp_pose", (7,), (Source("tcp_pose"),)),
            StateKey("tcp_vel", (6,), (Source("tcp_vel"),)),
            effector,
            StateKey("tcp_force", (3,), (Source("tcp_force"),)),
            StateKey("tcp_torque", (3,), (Source("tcp_torque"),)),
        )
        # A Franka policy always reads at least one camera.
        return ObservationSpec(state, cameras=cameras, min_cameras=1)
