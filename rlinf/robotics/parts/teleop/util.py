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

"""Helpers shared by teleoperation devices."""

from __future__ import annotations

import numpy as np


def jittered_grip(is_open: bool) -> np.ndarray:
    """Return a binary grip command with bounded training noise."""
    if is_open:
        return np.random.uniform(0.9, 1.0, size=(1,))
    return np.random.uniform(-1.0, -0.9, size=(1,))
