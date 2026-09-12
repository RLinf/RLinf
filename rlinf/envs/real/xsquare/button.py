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

"""Button pressing on a Turtle2."""

import numpy as np

from rlinf.envs.real.tasks import Button

from .base import Turtle2Env


class ButtonEnv(Turtle2Env):
    """Press a button with a Turtle2 arm."""

    TASK = Button
    DEFAULTS = {
        # A press holds the button's own orientation and lets go of nothing,
        # so the gripper channel is fixed shut.
        "enforce_gripper_close": True,
        "action_scale": (0.01, 0.05, 0.0),
        "enable_random_reset": True,
        "random_xy_range": 0.05,
        "random_rpy_range": np.pi / 9,
        "enable_gripper_penalty": False,
    }
