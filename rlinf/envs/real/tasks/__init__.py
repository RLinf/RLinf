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

"""Real-world tasks, written once and run on any robot that meets them.

Gymnasium ids are registered by the robot packages, which pair a task with
the control and observation layout of one robot.
"""

from .base import Evaluation, ResetContext, Task, TaskConfig
from .bimanual import MultiArmTarget, MultiArmTargetConfig
from .bin_relocation import BinRelocation, BinRelocationConfig
from .bottle import BottleCap, BottleCapConfig
from .button import Button, ButtonConfig
from .cartesian import CartesianTarget, CartesianTargetConfig, FixtureConfig
from .peg_insertion import PegInsertion, PegInsertionConfig
from .pick_place import PickPlace, PickPlaceConfig
from .reach import JointReach, JointReachConfig
from .requirements import Bound, Needs, Parts, Reading, RequirementError, bind
from .workspace import Box, Workspace

__all__ = [
    "BinRelocation",
    "BinRelocationConfig",
    "BottleCap",
    "BottleCapConfig",
    "Button",
    "ButtonConfig",
    "Bound",
    "Box",
    "CartesianTarget",
    "CartesianTargetConfig",
    "Evaluation",
    "FixtureConfig",
    "JointReach",
    "JointReachConfig",
    "Needs",
    "Parts",
    "MultiArmTarget",
    "MultiArmTargetConfig",
    "PegInsertion",
    "PegInsertionConfig",
    "PickPlace",
    "PickPlaceConfig",
    "Reading",
    "RequirementError",
    "ResetContext",
    "Task",
    "TaskConfig",
    "Workspace",
    "bind",
]
