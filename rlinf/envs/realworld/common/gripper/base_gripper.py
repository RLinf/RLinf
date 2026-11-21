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

from abc import ABC, abstractmethod


class BaseGripper(ABC):
    """Abstract base class for robot gripper control.

    All gripper implementations (Franka parallel gripper, Robotiq 2F, …)
    must implement this interface so that :class:`FrankaController` can
    use them interchangeably.
    """

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
    def move(self, position: float, speed: float = 0.3) -> None:
        """Move gripper to an absolute position.

        Args:
            position: Target position. Semantics are implementation-specific
                (e.g. Franka uses width in metres, Robotiq uses 0–255).
            speed: Movement speed.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> float:
        """Current gripper opening width / position."""
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

    def cleanup(self) -> None:
        """Release hardware resources (serial port, ROS channels, …)."""
