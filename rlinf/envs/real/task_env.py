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
from typing import Any, Optional, Protocol

import gymnasium as gym

from rlinf.robotics import Robot


class ValueAdapter(Protocol):
    """Convert values between canonical and public representations."""

    def adapt(self, value: Any) -> Any:
        """Translate ``value`` into the destination representation."""


class RobotTask(ABC):
    """Task logic evaluated against a composed physical robot."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language task description."""

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        """Policy-facing observation space."""

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """Policy-facing action space."""

    @abstractmethod
    def reset(
        self,
        robot: Robot,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset task and robot state and return a canonical observation."""

    @abstractmethod
    def step(
        self,
        robot: Robot,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply a canonical action and return a canonical transition."""


class RobotTaskEnv(gym.Env):
    """Gymnasium adapter for task logic and a composed robot runtime."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: Robot,
        task: RobotTask,
        observation_adapter: Optional[ValueAdapter] = None,
        action_adapter: Optional[ValueAdapter] = None,
    ) -> None:
        self.robot = robot
        self.task = task
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self.observation_space = task.observation_space
        self.action_space = task.action_space
        if not robot.is_connected:
            robot.connect()

    @property
    def task_description(self) -> str:
        """Return the task's natural-language description."""
        return self.task.description

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the task and adapt its canonical observation."""
        super().reset(seed=seed)
        observation, info = self.task.reset(
            self.robot,
            seed=seed,
            options=options,
        )
        if self.observation_adapter is not None:
            observation = self.observation_adapter.adapt(observation)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Adapt one action, execute the task, and adapt the observation."""
        if self.action_adapter is not None:
            action = self.action_adapter.adapt(action)
        observation, reward, terminated, truncated, info = self.task.step(
            self.robot,
            action,
        )
        if self.observation_adapter is not None:
            observation = self.observation_adapter.adapt(observation)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Disconnect all robot components."""
        self.robot.disconnect()
