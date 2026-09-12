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

"""A dual-arm Franka commanded by joint targets.

``joint_action_scale`` decides which: unset, a policy sends the joints it
wants, bounded by the arm's limits; set, it sends one unit either way and that
many radians are added to the joints as they were read.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from rlinf.envs.real.policy import Channel, JointPositions
from rlinf.robotics.parts.arms.franka import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
from rlinf.robotics.robots.dual_franka import DualFrankaConfig

from .dual_base import DualArmActionConfig, DualFrankaEnv


@dataclass
class DualFrankaJointActionConfig(DualArmActionConfig):
    """Joint bounds and command mode for both arms."""

    joint_position_limits_lower: Sequence[float] = field(
        default_factory=lambda: JOINT_LIMITS_LOWER.copy()
    )
    """Lowest target per joint, shared by both arms."""

    joint_position_limits_upper: Sequence[float] = field(
        default_factory=lambda: JOINT_LIMITS_UPPER.copy()
    )
    """Highest target per joint, shared by both arms."""

    joint_action_scale: Optional[float] = None
    """Radians per action unit. ``None`` commands the joints directly."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.joint_position_limits_lower = np.asarray(
            self.joint_position_limits_lower, dtype=np.float64
        )
        self.joint_position_limits_upper = np.asarray(
            self.joint_position_limits_upper, dtype=np.float64
        )


class DualFrankaJointEnv(DualFrankaEnv):
    """A dual-arm Franka whose policy sends joint targets."""

    # The two dual ids differ in how the arms are driven, not in the task,
    # so a run names the one its checkpoint was trained against.
    GENERIC_ID = False
    ACTION_CONFIG = DualFrankaJointActionConfig
    # Ignoring the mode would turn an absolute config into a delta one, so a
    # run that still names it is refused rather than driven the other way.
    REFUSED = {
        "joint_action_mode": (
            "Drop it. Leave 'joint_action_scale' unset for absolute targets, "
            "or set it for a change of that many radians per action unit."
        ),
    }

    @classmethod
    def arm_channel(
        cls,
        side: str,
        hardware: DualFrankaConfig,
        config: DualFrankaJointActionConfig,
    ) -> Channel:
        """Seven joint targets for one arm."""
        return JointPositions(
            side,
            low=config.joint_position_limits_lower,
            high=config.joint_position_limits_upper,
            scale=config.joint_action_scale,
            compliance=cls.compliance_for(config),
            name=f"{side}.arm",
        )
