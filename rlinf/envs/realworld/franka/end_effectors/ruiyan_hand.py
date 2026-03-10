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

"""Ruiyan dexterous five-finger hand end-effector.

This is a thin wrapper that delegates to the ``rlinf_dexhand``
package (``pip install rlinf_dexhand``) and adapts it to the
:class:`EndEffector` interface used by the Franka env.
"""

from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base import EndEffector


class RuiyanHand(EndEffector):
    """Ruiyan dexterous hand — thin wrapper around ``rlinf_dexhand``.

    Install the driver package first::

        pip install rlinf_dexhand

    Args:
        port: Serial device path, e.g. ``"/dev/ttyUSB0"``.
        baudrate: Serial baudrate (default 460800).
        motor_ids: Tuple of motor IDs corresponding to the 6 fingers.
        default_velocity: Default command velocity for all motors.
        default_current: Default command current for all motors.
        default_state: Default hand state used during ``reset()``.
    """

    _NUM_DOFS = 6
    _FINGER_NAMES = [
        "thumb_rotation",
        "thumb_bend",
        "index",
        "middle",
        "ring",
        "pinky",
    ]

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 460800,
        motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        default_velocity: int = 2000,
        default_current: int = 800,
        default_state: Optional[list[float]] = None,
    ):
        from rlinf_dexhand.ruiyan import RuiyanHandDriver

        self._driver = RuiyanHandDriver(
            port=port,
            baudrate=baudrate,
            motor_ids=motor_ids,
            default_velocity=default_velocity,
            default_current=default_current,
            default_state=default_state,
        )
        self._logger = get_logger()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def state_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def control_mode(self) -> str:
        return "continuous"

    @property
    def finger_names(self) -> list[str]:
        """Human-readable DOF names for the Ruiyan hand."""
        return list(self._FINGER_NAMES)

    def get_detailed_state(self) -> dict:
        """Return detailed per-motor diagnostic information."""
        return self._driver.get_detailed_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the serial port and start the background control loop."""
        self._driver.initialize()

    def shutdown(self) -> None:
        """Stop the background loop and close the serial port."""
        self._driver.shutdown()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return the latest finger positions (normalised ``[0, 1]``)."""
        return self._driver.get_state()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command(self, action: np.ndarray) -> bool:
        """Set target finger positions (normalised ``[0, 1]``)."""
        return self._driver.command(action)

    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset hand to the default or specified state."""
        self._driver.reset(target_state)
