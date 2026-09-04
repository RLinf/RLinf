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

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.end_effectors.base import BaseEndEffector


class BaseGripper(BaseEndEffector, ABC):
    """One-axis end-effector interface for parallel grippers.

    Drivers implement :meth:`open`, :meth:`close`, and :meth:`move`; this class
    maps them to the common end-effector state and command interface. Connection
    state and activation readiness are reported separately.
    """

    @classmethod
    def declare(
        cls,
        *,
        ros: Optional[Any] = None,
        port: Optional[str] = None,
        robot_ip: Optional[str] = None,
        **settings: Any,
    ) -> "BaseGripper":
        """Declare a gripper from the attachment settings offered by its arm.

        An arm offers every attachment it can reach a gripper through -- a ROS
        session, a serial port, its own IP -- and each backend takes the one it
        uses and ignores the rest.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not say which attachment it is reached "
            "through. Override declare() to take the one it uses."
        )

    @property
    def is_gripper(self) -> bool:
        """Every driver in this category opens and closes on one axis."""
        return True

    @abstractmethod
    def open(self, speed: float = 0.3) -> None:
        """Fully open the gripper.

        Args:
            speed: Opening speed, normalized to [0, 1].
        """
        raise NotImplementedError

    @abstractmethod
    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        """Fully close the gripper (or grasp).

        Args:
            speed: Closing speed, normalized to [0, 1].
            force: Grasping force (unit depends on implementation).
        """
        raise NotImplementedError

    @abstractmethod
    def move(self, width: float, speed: float = 0.3) -> None:
        """Move the fingers to an absolute opening width.

        Args:
            width: Target opening in metres, clamped to ``[0, max_width]``.
            speed: Movement speed, normalized to [0, 1].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> float:
        """Current opening width, in metres."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_width(self) -> float:
        """Return the fully open width in metres."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the gripper is currently in the *open* state."""
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the gripper is activated and ready to accept commands."""
        raise NotImplementedError

    # End-effector interface derived from the gripper primitives.

    @property
    def state_dim(self) -> int:
        """Return the one-dimensional gripper state size."""
        return 1

    @property
    def action_dim(self) -> int:
        """Return the one-dimensional gripper action size."""
        return 1

    @property
    def control_mode(self) -> str:
        """Return ``"continuous"`` for absolute-width control."""
        return "continuous"

    def get_state(self) -> np.ndarray:
        """Return the opening width in metres as a one-element vector."""
        return np.asarray([self.position], dtype=np.float32)

    def command(self, action: np.ndarray) -> bool:
        """Move to an absolute width and report whether open state changed."""
        target = np.asarray(action, dtype=np.float32).reshape(-1)
        if target.size != self.action_dim:
            raise ValueError(f"Gripper target must have one value, got {target.size}.")
        was_open = self.is_open
        self.move(float(target[0]))
        return self.is_open != was_open

    def reset(self, target_state: Optional[np.ndarray] = None) -> None:
        """Open the gripper, or move it to ``target_state`` when one is given."""
        if target_state is None:
            self.open()
            return
        self.command(target_state)
