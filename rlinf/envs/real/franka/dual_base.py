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

"""How a dual-arm Franka is driven and what its policy sees, for any task.

Each arm is a role, ``left`` and ``right``, with its own channels and its own
entry in every per-arm table. Nothing here counts arms: a task states what it
needs of each role, and the layout drives whichever roles it has channels for.

The two ids differ only in how the arms are commanded. ``DualFrankaJointEnv``
sends joint targets, directly or as a change; ``DualFrankaTCPEnv`` sends tool
waypoints with a six-number rotation. Both grip through a binary gripper the
step does not wait for, because the fingers take longer to close than one tick
of a 10 Hz loop.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from rlinf.envs.real.policy import (
    ActionLayout,
    BinaryGripper,
    Channel,
    ObservationSpec,
    Phase,
    Source,
    StateKey,
)
from rlinf.envs.real.task_env import RegisteredTaskEnv
from rlinf.envs.real.tasks import MultiArmTarget
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.robotics.robots.dual_franka import DualFrankaConfig, DualFrankaRobot

#: The roles a dual-arm Franka fills, in the order every table indexes them.
SIDES: tuple[str, ...] = ("left", "right")


@dataclass
class DualArmActionConfig:
    """How each arm of a two-armed robot is driven."""

    action_scale: Sequence[float] = (1.0, 1.0, 1.0)
    """Metres and radians per unit of arm action, then the gripper
    multiplier. Only the third entry applies to an arm commanded by joints."""

    compliance_param: Mapping[str, float] = field(default_factory=dict)
    """Impedance gains applied to both arms at the start of every episode."""

    binary_gripper_threshold: float = 0.5
    """Gripper channel magnitude at which a gripper opens or closes."""

    gripper_settle_s: float = 0.0
    """Seconds a gripper is given to finish. The step does not wait for it, so
    this only spaces one gripper command from the next."""

    def __post_init__(self) -> None:
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)
        self.compliance_param = dict(self.compliance_param)


class DualFrankaEnv(RegisteredTaskEnv):
    """A task on a dual-arm Franka, one role per arm.

    Subclasses say how the arms are commanded by returning that side's arm
    channel from :meth:`arm_channel`; everything else -- the grippers, what the
    policy reads, the cameras -- is the same for both ids.
    """

    ROBOT = DualFrankaRobot
    ACTION_CONFIG = DualArmActionConfig
    TASK = MultiArmTarget
    TELEOP = ("gello_joint", "pico")
    TELEOP_DEFAULT = "none"
    TELEOP_MARK_FLAG = True
    # 'no_gripper' strips a single trailing gripper value, which a two-armed
    # action does not have, so a run that asks for it is refused rather than
    # quietly driven with the wrong action.
    REFUSE_FLAGS = ("no_gripper",)
    REFUSE_DEFAULTS = {"no_gripper": True}
    DEFAULTS = {
        "roles": SIDES,
        "reset_joint_qpos": [[0.0, 0.0, 0.0, -1.9, 0.0, 2.0, 0.0]] * 2,
        "enable_gripper_penalty": True,
    }
    REFUSED = {
        "teleop_direct_stream": (
            "Set it beside 'teleop' in the env config, not in 'override_cfg'."
        ),
    }
    RETIRED = {
        "reset_ee_pose": "Both arms return to 'reset_joint_qpos' between episodes.",
        "enable_random_reset": "A dual-arm reset does not randomise.",
        "random_xy_range": "A dual-arm reset does not randomise.",
        "random_rz_range": "A dual-arm reset does not randomise.",
        "joint_reset_cycle": "Every reset returns the joints to 'reset_joint_qpos'.",
        "save_video_path": "Record episodes with the env's video_cfg instead.",
    }

    @classmethod
    def arm_channel(
        cls, side: str, hardware: DualFrankaConfig, config: DualArmActionConfig
    ) -> Channel:
        """The channel driving one arm, named ``<side>.arm``.

        Raises:
            NotImplementedError: On :class:`DualFrankaEnv` itself, which does
                not say how the arms are commanded.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not command the arms. Use DualFrankaJointEnv "
            "or DualFrankaTCPEnv, or give this id an arm channel of its own."
        )

    @classmethod
    def make_action(
        cls,
        hardware: DualFrankaConfig,
        config: DualArmActionConfig,
        options: None = None,
        roles: tuple[str, ...] = SIDES,
    ) -> ActionLayout:
        """Each arm's channel, then its gripper, in the order the task names."""
        channels: list[Channel] = []
        for side in roles:
            channels.append(cls.arm_channel(side, hardware, config))
            channels.append(
                BinaryGripper(
                    side,
                    threshold=config.binary_gripper_threshold,
                    scale=float(config.action_scale[2]),
                    settle_s=config.gripper_settle_s,
                    # The fingers take longer than one tick, so the step asks
                    # for the grasp and goes on with the arm's motion.
                    phase=Phase.WITH,
                    awaited=False,
                    name=f"{side}.end_effector",
                )
            )
        return ActionLayout(tuple(channels))

    @classmethod
    def make_observation(
        cls,
        hardware: DualFrankaConfig,
        cameras: tuple[CameraInfo, ...],
        roles: tuple[str, ...] = SIDES,
    ) -> ObservationSpec:
        """Every driven arm's pose, twist, joints, wrench and gripper, in order."""
        arms = len(roles)
        state = (
            StateKey(
                "tcp_pose",
                (7 * arms,),
                tuple(Source("tcp_pose", role=r) for r in roles),
            ),
            StateKey(
                "tcp_vel", (6 * arms,), tuple(Source("tcp_vel", role=r) for r in roles)
            ),
            StateKey(
                "joint_position",
                (7 * arms,),
                tuple(Source("arm_joint_position", role=r) for r in roles),
            ),
            StateKey(
                "joint_velocity",
                (7 * arms,),
                tuple(Source("arm_joint_velocity", role=r) for r in roles),
            ),
            StateKey(
                "gripper_position",
                (arms,),
                tuple(Source("state", role=r, end_effector=True) for r in roles),
                low=-1.0,
                high=1.0,
            ),
            StateKey(
                "tcp_force",
                (3 * arms,),
                tuple(Source("tcp_force", role=r) for r in roles),
            ),
            StateKey(
                "tcp_torque",
                (3 * arms,),
                tuple(Source("tcp_torque", role=r) for r in roles),
            ),
        )
        # A dual-arm policy was trained on 224-pixel frames.
        return ObservationSpec(state, cameras=cameras, frame_size=(224, 224))

    @classmethod
    def camera_infos(
        cls, hardware: DualFrankaConfig, config: Any
    ) -> "list[CameraInfo]":
        """Declare the base cameras, then each wrist's, as the robot names them.

        A wrist camera's name carries the side, which is how the robot puts it
        in that arm's group and on that arm's node.
        """
        default_type = hardware.camera_type or "realsense"
        regions = config.camera_crop_regions or {}
        infos: list[CameraInfo] = []

        def add(name: str, serial: str, camera_type: str) -> None:
            from rlinf.envs.real.task_env import crop_region

            region = regions.get(serial)
            infos.append(
                CameraInfo(
                    name=name,
                    serial_number=serial,
                    camera_type=camera_type,
                    crop_region=None
                    if region is None
                    else crop_region(region, camera=name, serial=serial),
                    enable_depth=config.enable_camera_depth,
                )
            )

        for index, serial in enumerate(hardware.base_camera_serials or []):
            add(
                f"base_{index}_rgb",
                str(serial),
                hardware.base_camera_type or default_type,
            )
        for side in SIDES:
            serials = getattr(hardware, f"{side}_camera_serials") or []
            camera_type = getattr(hardware, f"{side}_camera_type") or default_type
            for index, serial in enumerate(serials):
                add(f"{side}_wrist_{index}_rgb", str(serial), camera_type)
        return infos

    # Teleoperation reads the layout, so nothing here needs a per-arm getter.

    @classmethod
    def compliance_for(
        cls, config: DualArmActionConfig
    ) -> Optional[Mapping[str, float]]:
        """The impedance gains an arm channel applies at each reset."""
        return config.compliance_param or None
