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
from typing import Any

import numpy as np

from rlinf.robotics.parts.base import EndEffector


class BaseGripper(EndEffector, ABC):
    """Abstract base class for robot gripper control.

    All gripper implementations (Franka parallel gripper, Robotiq 2F, …)
    must implement this interface so that the arm part can
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

    @property
    def is_connected(self) -> bool:
        """Whether the backend is ready to accept commands."""
        return self.is_ready()

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the scalar gripper position."""
        return {"position": {"shape": (1,), "dtype": "float32"}}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe the scalar absolute-position command."""
        return {"target": {"shape": (1,), "dtype": "float32"}}

    def connect(self) -> None:
        """Validate the connection established by the backend constructor."""
        if not self.is_ready():
            raise RuntimeError(f"{type(self).__name__} is not ready.")

    def disconnect(self) -> None:
        """Release backend resources."""
        self.cleanup()

    def reset(self) -> None:
        """Reset the gripper to its open state."""
        self.open()

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return the current gripper position."""
        return {"position": np.asarray([self.position], dtype=np.float32)}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply one absolute-position target."""
        if set(action) != {"target"}:
            raise KeyError("Gripper action must contain only 'target'.")
        target = np.asarray(action["target"], dtype=np.float32).reshape(-1)
        if target.size != 1:
            raise ValueError(f"Gripper target must have one value, got {target.size}.")
        self.move(float(target[0]))
        return {"target": target}
