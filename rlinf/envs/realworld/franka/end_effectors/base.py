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

"""Abstract base class for robot end-effectors."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class EndEffectorType(str, Enum):
    """Supported end-effector types for the Franka robot arm."""

    FRANKA_GRIPPER = "franka_gripper"
    AOYI_HAND = "aoyi_hand"
    RUIYAN_HAND = "ruiyan_hand"


class EndEffector(ABC):
    """Abstract interface for a robot end-effector.

    Every end-effector must expose its state and action dimensions so
    that ``FrankaEnv`` can build the correct Gymnasium spaces dynamically.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the end-effector action vector."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the end-effector state vector."""

    @property
    @abstractmethod
    def control_mode(self) -> str:
        """Control mode: ``"binary"`` (open/close) or ``"continuous"``."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Perform any hardware-level initialization (serial open, etc.)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully release hardware resources."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current end-effector state as a 1-D array.

        The length of the returned array must equal :pyattr:`state_dim`.
        """

    @property
    def finger_names(self) -> list[str]:
        """Human-readable names for each DOF.

        Subclasses may override this to provide meaningful labels.
        The default returns generic names ``["dof_0", "dof_1", ...]``.
        """
        return [f"dof_{i}" for i in range(self.state_dim)]

    def get_detailed_state(self) -> dict:
        """Return a detailed status dictionary for diagnostic purposes.

        The default implementation wraps :meth:`get_state` into a dict.
        Subclasses (e.g. dexterous hands) should override this to expose
        per-motor velocity, current, error status, etc.
        """
        state = self.get_state()
        return {
            "positions": state.tolist(),
            "finger_names": self.finger_names,
        }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @abstractmethod
    def command(self, action: np.ndarray) -> bool:
        """Send a command to the end-effector.

        Args:
            action: Action vector whose length equals :pyattr:`action_dim`.

        Returns:
            ``True`` if the command caused a meaningful state change
            (e.g. gripper opened/closed), ``False`` otherwise.
        """

    @abstractmethod
    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a default or specified state.

        Args:
            target_state: Optional target state. If ``None``, reset to the
                implementation-defined default.
        """
