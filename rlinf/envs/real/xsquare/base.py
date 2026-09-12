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

"""How a Turtle2 is driven and what its policy sees, for any task.

Each arm is a role, ``left`` and ``right``, and a run names the ones it uses
in the task's ``roles``. The arms take tool-pose deltas whose rotation is
added to the current Euler angles, which is what this controller's own
smoothing loop was tuned against, and a gripper commanded by its opening.

Frames stay in the order the cameras deliver them, because this robot's
checkpoints were trained on them that way.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rlinf.envs.real.policy import (
    ActionLayout,
    Channel,
    GripperWidth,
    ObservationSpec,
    PoseDelta,
    Source,
    StateKey,
)
from rlinf.envs.real.task_env import RegisteredTaskEnv
from rlinf.envs.real.tasks import MultiArmTarget
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.robotics.robots.turtle2 import Turtle2Config, Turtle2Robot


@dataclass
class Turtle2ActionConfig:
    """How each Turtle2 arm is driven."""

    action_scale: Sequence[float] = (1.0, 1.0, 1.0)
    """Metres per unit of position action, radians per unit of rotation
    action, and the gripper multiplier, which this robot does not use."""

    gripper_width_limit_min: float = 0.0
    """Narrowest opening the grippers accept."""

    gripper_width_limit_max: float = 5.0
    """Widest opening they accept."""

    enforce_gripper_close: bool = True
    """Hold the grippers shut and ignore the channel, for a task that must
    not let go of what it is holding."""

    def __post_init__(self) -> None:
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)


@dataclass
class Turtle2Options:
    """Settings the preset reads itself."""

    smooth_frequency: int = 50
    """Ticks per second of the controller's own smoothing loop."""


class Turtle2Env(RegisteredTaskEnv):
    """A task on a Turtle2, one role per arm in use."""

    ROBOT = Turtle2Robot
    ACTION_CONFIG = Turtle2ActionConfig
    TASK = MultiArmTarget
    OPTIONS = Turtle2Options
    TELEOP = ("spacemouse", "gello", "pico")
    TELEOP_DEFAULT = "spacemouse"
    DEFAULTS = {
        "roles": ("right",),
        "score_dims": "xyz_rpy",
        "dense_gain": 200.0,
        "orientation_clip": "box",
        "pre_reset_ee_pose": [0.2, 0.0, 0.1, 0.0, 0.0, 0.0],
        "require_reset_arrival": True,
        # This controller interpolates toward its target, so the arm settles
        # near the rest pose rather than exactly on it.
        "reset_tolerance": 0.02,
        # This controller smooths toward its target rather than jumping, so
        # the arm is still moving when the last waypoint is sent.
        "reset_arrive_within": 10.0,
        "enable_gripper_penalty": True,
    }
    REFUSED = {
        "use_arm_ids": (
            'Name the arms in \'roles\' as "left" and "right", and give every '
            "per-arm table one row per named role."
        ),
    }
    RETIRED = {
        "save_video_path": "Record episodes with the env's video_cfg instead.",
        "add_gripper_penalty": "Use enable_gripper_penalty.",
    }

    @classmethod
    def robot_options(cls, options: Turtle2Options) -> Mapping[str, Any]:
        """The controller's smoothing rate, which the connection opens with."""
        return {"frequency": options.smooth_frequency}

    @classmethod
    def make_action(
        cls,
        hardware: Turtle2Config,
        config: Turtle2ActionConfig,
        options: Turtle2Options = None,
        roles: tuple[str, ...] = ("right",),
    ) -> ActionLayout:
        """Each arm's pose delta, then its gripper's opening.

        One arm keeps the plain part names, as every single-armed robot uses;
        two qualify them by side, so a teleoperation device can tell them
        apart.
        """
        channels: list[Channel] = []
        qualify = len(roles) > 1
        for role in roles:
            channels.append(
                PoseDelta(
                    role,
                    scales=config.action_scale,
                    # This controller adds the rotation to its own Euler
                    # angles rather than composing a rotation onto the pose.
                    rotation="euler_add",
                    name=f"{role}.arm" if qualify else "arm",
                )
            )
            channels.append(
                GripperWidth(
                    role,
                    low=config.gripper_width_limit_min,
                    high=config.gripper_width_limit_max,
                    action_range=(-1.0, 1.0),
                    enforce_close=config.enforce_gripper_close,
                    name=f"{role}.end_effector" if qualify else "end_effector",
                )
            )
        return ActionLayout(
            tuple(channels),
            wrappers=("GripperCloseEnv",),
            transforms=("RelativeFrame", "Quat2EulerWrapper"),
        )

    @classmethod
    def make_observation(
        cls,
        hardware: Turtle2Config,
        cameras: tuple[CameraInfo, ...],
        roles: tuple[str, ...] = ("right",),
    ) -> ObservationSpec:
        """Every driven arm's tool pose, and the controller's own cameras."""
        state = (
            StateKey(
                "tcp_pose",
                (7 * len(roles),),
                tuple(Source("tcp_pose", role=role) for role in roles),
            ),
        )
        return ObservationSpec(
            state,
            cameras=cameras,
            frame_size=(128, 128),
            # This robot's checkpoints were trained on the frames as the
            # cameras deliver them.
            frame_order="bgr",
        )

    @classmethod
    def camera_infos(cls, hardware: Turtle2Config, config: Any) -> "list[CameraInfo]":
        """The controller's own camera channels, in the order configured."""
        del config
        return [
            CameraInfo(name=f"wrist_{index + 1}", serial_number=str(camera_id))
            for index, camera_id in enumerate(hardware.camera_ids)
        ]
