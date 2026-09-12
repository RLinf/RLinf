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

"""Cap tightening on a Franka."""

from rlinf.envs.real.tasks import BottleCap

from .base import FrankaEnv, compliance


class BottleEnv(FrankaEnv):
    """Cap tightening, with a free wrist yaw and fine translation."""

    TASK = BottleCap
    DEFAULTS = {
        "compliance_param": compliance(
            rotational_clip_neg_z=0.5,
            rotational_clip_z=0.5,
            translational_clip_neg_x=0.001,
            translational_clip_neg_y=0.001,
            translational_clip_neg_z=0.001,
            translational_clip_x=0.001,
            translational_clip_y=0.001,
            translational_clip_z=0.001,
        ),
        "action_scale": (0.01, 0.5, 1.0),
        "step_frequency": 5.0,
    }
