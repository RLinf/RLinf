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

"""Peg insertion on a Franka."""

from rlinf.envs.real.tasks import PegInsertion

from .base import FrankaEnv, compliance


class PegInsertionEnv(FrankaEnv):
    """Peg insertion, with a stiffer arm and small steps."""

    TASK = PegInsertion
    DEFAULTS = {
        "compliance_param": compliance(translational_stiffness=2000),
        "action_scale": (0.02, 0.1, 1.0),
    }
