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

"""What the policy sends and reads, declared in one place.

A robot preset builds an :class:`ActionLayout` and an :class:`ObservationSpec`.
Together they are the contract a checkpoint is trained against: the channels
of its action, the keys of its observation, the order they flatten in, and the
colour order of its frames.
"""

from .action import (
    ActionLayout,
    Applied,
    Channel,
    Command,
    Effect,
    Phase,
)
from .channels import (
    BinaryGripper,
    ContinuousGripper,
    GripperWidth,
    HandCommand,
    JointActionConfig,
    JointPositions,
    PoseActionConfig,
    PoseDelta,
    PoseTarget,
)
from .observation import (
    Encoder,
    ObservationSpec,
    Source,
    StateKey,
    pose_as_rot6d,
    pose_from_euler,
)

__all__ = [
    "ActionLayout",
    "Applied",
    "BinaryGripper",
    "Channel",
    "Command",
    "ContinuousGripper",
    "GripperWidth",
    "Effect",
    "Encoder",
    "HandCommand",
    "JointActionConfig",
    "JointPositions",
    "ObservationSpec",
    "Phase",
    "PoseActionConfig",
    "PoseDelta",
    "PoseTarget",
    "Source",
    "StateKey",
    "pose_as_rot6d",
    "pose_from_euler",
]
