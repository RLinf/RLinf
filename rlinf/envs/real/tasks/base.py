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

"""A real-world task: what counts as doing the job, on whatever robot does it.

A task states what it needs from the robot, puts the scene back between
episodes, and scores each step. It never builds an action: how a policy's
action reaches the arm belongs to the :class:`~rlinf.envs.real.control.Control`
the env is driven with, so one task runs under joint and Cartesian control
alike.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import numpy as np

from .requirements import Needs, Parts, Reading
from .workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.envs.real.policy import ActionLayout, Applied


@dataclass
class TaskConfig:
    """Settings every task has; a task adds its own by subclassing."""

    task_description: str = ""
    """Language instruction for policies that condition on one. Empty uses
    the task's own :attr:`Task.DESCRIPTION`."""

    success_hold_steps: int = 1
    """Consecutive successful steps before the episode terminates."""

    use_dense_reward: bool = False
    """Report a shaped reward instead of a sparse hit."""

    enable_gripper_penalty: bool = False
    """Charge :attr:`gripper_penalty` whenever a gripper opens or closes."""

    gripper_penalty: float = 0.1
    """Reward subtracted for one gripper change."""


@dataclass
class Evaluation:
    """How one step went.

    Attributes:
        reward: The task's reward, before any gripper penalty or scaling.
        in_zone: Whether this step counts toward the successful streak that
            ends an episode.
    """

    reward: float
    in_zone: bool


@dataclass
class ResetContext:
    """What a task's reset may use beyond the robot itself."""

    rng: np.random.Generator
    """The env's generator, seeded by ``reset(seed=...)``."""

    action: "ActionLayout"
    """The action layout the env is driven with: its limits, and the
    end-effector verbs a reset asks for."""

    options: Mapping[str, Any] = field(default_factory=dict)
    """The ``options`` passed to ``reset``."""

    rate_hz: float = 10.0
    """The env's control rate, for motions a reset streams to the arm."""


class Task(ABC):
    """Task logic evaluated against a composed robot.

    Subclasses set :attr:`CONFIG`, declare :meth:`requirements`, and implement
    :meth:`evaluate`; :meth:`home` and :meth:`reset` default to doing nothing.
    """

    #: The config dataclass built from a run's ``override_cfg``.
    CONFIG: ClassVar[type[TaskConfig]] = TaskConfig

    #: Instruction used when the config leaves ``task_description`` empty.
    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, config: Optional[TaskConfig] = None) -> None:
        self.config = config if config is not None else self.CONFIG()

    @property
    def description(self) -> str:
        """Language instruction for policies that condition on one."""
        return self.config.task_description or self.DESCRIPTION

    @abstractmethod
    def requirements(self) -> Mapping[str, Needs]:
        """What each role must report, accept, and carry."""

    @property
    def workspace(self) -> "Optional[Union[Workspace, Mapping[str, Workspace]]]":
        """Where a control that commands poses may send the tool.

        ``None`` leaves the tool unconstrained, as for a task that never asks
        for a pose. A task whose roles reach different parts of the bench
        returns one workspace per role instead of one for all of them.
        """
        return None

    def validate(self, dof: Mapping[str, Optional[int]]) -> None:
        """Check settings that depend on the arm, before anything connects.

        Args:
            dof: Joints of the arm filling each role, ``None`` where unknown.
        """

    def home(self, parts: Parts, context: ResetContext) -> None:
        """Bring the robot to where an episode can start, once, after connecting."""

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Put the robot and scene back between episodes."""

    @abstractmethod
    def evaluate(self, reading: Reading, applied: "Applied") -> Evaluation:
        """Score the robot as it was read after one step."""
