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

"""Aoyi dexterous five-finger hand end-effector.

This is a thin wrapper that delegates to the ``rlinf_dexhand``
package (``pip install rlinf_dexhand[aoyi]``) and adapts it to the
:class:`EndEffector` interface used by the Franka env.
"""

from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base import EndEffector


class AoyiHand(EndEffector):
    """Aoyi modular dexterous hand — thin wrapper around ``rlinf_dexhand``.

    Install the driver package first::

        pip install "rlinf_dexhand[aoyi]"

    Args:
        port: Serial device path, e.g. ``"/dev/ttyUSB0"``.
        node_id: Modbus slave node ID.
        baudrate: Serial baudrate.
        default_state: Default hand state used during ``reset()``.
    """

    _NUM_DOFS = 6

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        node_id: int = 2,
        baudrate: int = 115200,
        default_state: Optional[list[float]] = None,
    ):
        from rlinf_dexhand.aoyi import AoyiHandDriver

        self._driver = AoyiHandDriver(
            port=port,
            node_id=node_id,
            baudrate=baudrate,
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the Modbus RTU connection and initialise the hand."""
        self._driver.initialize()

    def shutdown(self) -> None:
        """Close the Modbus serial connection."""
        self._driver.shutdown()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Read current finger positions and return normalised ``[0, 1]`` array."""
        return self._driver.get_state()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command(self, action: np.ndarray) -> bool:
        """Write target finger positions."""
        return self._driver.command(action)

    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset hand to the default or specified state."""
        self._driver.reset(target_state)
