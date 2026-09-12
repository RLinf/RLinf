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

"""Robot-free task ids, such as ``PegInsertion-v1``.

Importing this module imports every robot package, then registers one id per
task that runs on the robot a run is given. The robot-bound ids, such as
``PegInsertionEnv-v1``, stay registered and name one robot's preset.
"""

import importlib

from . import _ROBOT_PACKAGES
from .registry import register_task_ids

_ENTRY_POINTS = register_task_ids(
    __name__,
    globals(),
    [
        importlib.import_module(package, "rlinf.envs.real").TASKS
        for package in _ROBOT_PACKAGES
    ],
)

__all__ = [*_ENTRY_POINTS]
