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

"""A dual-arm Franka commanded by tool waypoints.

Each arm takes a position and a six-number rotation, the first two columns of
the rotation matrix. The policy reads its poses back in the same form, so a
checkpoint never sees a quaternion's sign ambiguity.
"""

from dataclasses import dataclass

from rlinf.envs.real.policy import (
    Channel,
    ObservationSpec,
    PoseTarget,
    Source,
    StateKey,
    pose_as_rot6d,
)
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.robotics.robots.dual_franka import DualFrankaConfig

from .dual_base import SIDES, DualArmActionConfig, DualFrankaEnv


@dataclass
class DualFrankaTCPActionConfig(DualArmActionConfig):
    """Settings for waypoint-commanded arms."""

    rotation_limit: float = 1.5
    """Bound on each rotation number, left wide of one so a policy's output
    survives normalisation."""


class DualFrankaTCPEnv(DualFrankaEnv):
    """A dual-arm Franka whose policy sends absolute tool waypoints."""

    # The two dual ids differ in how the arms are driven, not in the task,
    # so a run names the one its checkpoint was trained against.
    GENERIC_ID = False
    ACTION_CONFIG = DualFrankaTCPActionConfig
    RETIRED = {
        "rotation_repr": "Waypoints are position plus a six-number rotation.",
    }

    @classmethod
    def arm_channel(
        cls,
        side: str,
        hardware: DualFrankaConfig,
        config: DualFrankaTCPActionConfig,
    ) -> Channel:
        """A tool waypoint for one arm, bounded by the task's workspace."""
        return PoseTarget(
            side,
            rotation_limit=config.rotation_limit,
            compliance=cls.compliance_for(config),
            name=f"{side}.arm",
        )

    @classmethod
    def make_observation(
        cls,
        hardware: DualFrankaConfig,
        cameras: tuple[CameraInfo, ...],
        roles: tuple[str, ...] = SIDES,
    ) -> ObservationSpec:
        """Every tool pose in the form the action uses, and every gripper."""
        arms = len(roles)
        state = (
            StateKey(
                "gripper_position",
                (arms,),
                tuple(Source("state", role=r, end_effector=True) for r in roles),
                low=-1.0,
                high=1.0,
            ),
            StateKey(
                "tcp_pose_rot6d",
                (9 * arms,),
                tuple(Source("tcp_pose", role=r, encode=pose_as_rot6d) for r in roles),
            ),
        )
        # A dual-arm policy was trained on 224-pixel frames.
        return ObservationSpec(state, cameras=cameras, frame_size=(224, 224))
