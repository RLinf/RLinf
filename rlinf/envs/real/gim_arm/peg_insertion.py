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

"""Peg insertion on a GimArm, through the task Franka runs it with."""

from rlinf.envs.real.tasks import PegInsertion

from .base import GimArmEnv


class GimArmPegInsertionEnv(GimArmEnv):
    """Peg insertion, resting through joint configurations."""

    TASK = PegInsertion
    DEFAULTS = {"safe_retract_qpos": (0.0, -1.5, 1.5, 0.0, 0.0, 0.0)}
