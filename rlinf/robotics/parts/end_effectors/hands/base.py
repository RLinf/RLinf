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

"""Common interface for articulated robot hands."""

from abc import ABC
from typing import Any

from ..base import EndEffector


class BaseHand(EndEffector, ABC):
    """End effector that controls finger poses.

    Drivers supply vector dimensions, state, commands, and any device-specific
    reset behavior. Connection ownership follows the common part lifecycle.
    """

    is_hand = True

    @property
    def finger_names(self) -> list[str]:
        """Return labels for the hand's state coordinates."""
        return [f"dof_{i}" for i in range(self.state_dim)]

    def get_detailed_state(self) -> dict[str, Any]:
        """Return diagnostic positions with finger labels."""
        return {**super().get_detailed_state(), "finger_names": self.finger_names}
