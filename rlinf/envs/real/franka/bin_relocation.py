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

"""Bin relocation on a Franka."""

from rlinf.envs.real.tasks import BinRelocation

from .base import FrankaEnv, compliance


class FrankaBinRelocationEnv(FrankaEnv):
    """Bin relocation, with a stiff arm and wider steps."""

    TASK = BinRelocation
    DEFAULTS = {
        "compliance_param": compliance(
            rotational_clip_neg_x=0.04,
            rotational_clip_neg_y=0.04,
            rotational_clip_x=0.04,
            rotational_clip_y=0.04,
            translational_clip_neg_x=0.004,
            translational_clip_neg_y=0.004,
            translational_clip_neg_z=0.004,
            translational_clip_x=0.004,
            translational_clip_y=0.004,
            translational_clip_z=0.004,
            translational_stiffness=2800,
        ),
        "action_scale": (0.03, 0.1, 1.0),
    }
