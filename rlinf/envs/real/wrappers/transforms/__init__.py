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
"""Wrappers that change how an observation or action is expressed.

These rewrite the representation, never the intent: a relative frame moves the
same motion into the end-effector's coordinates, and a Euler conversion says the
same rotation with different numbers. A policy trained against one representation
needs the matching wrapper at rollout, which is why they belong together rather
than scattered among the wrappers that decide what the robot does.
"""

from .dual_relative_frame import DualRelativeFrame, DualRelativeTargetFrame
from .euler_obs import DualQuat2EulerWrapper, Quat2EulerWrapper
from .gripper_close import GripperCloseEnv
from .relative_frame import RelativeFrame, RelativeTargetFrame

__all__ = [
    "DualQuat2EulerWrapper",
    "DualRelativeFrame",
    "DualRelativeTargetFrame",
    "GripperCloseEnv",
    "Quat2EulerWrapper",
    "RelativeFrame",
    "RelativeTargetFrame",
]
