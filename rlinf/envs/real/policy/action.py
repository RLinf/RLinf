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

"""What a policy sends, channel by channel.

A policy produces one flat vector. An :class:`ActionLayout` cuts it into
channels, each of which owns one contiguous slice: which role it drives, how
wide it is, what its numbers mean, and what command they become. A robot with
two arms is two arms' worth of channels, so nothing here has to know how many
arms a robot has.

Ordering within a step is a channel's :class:`Phase`, not a flag on the env: a
Franka grasps before the arm moves, a GimArm after it, and a gripper on the
arm's own bus goes out in the same command as the joints.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np

from rlinf.envs.real.tasks.requirements import Needs, Parts, Reading, combine, merge
from rlinf.envs.real.tasks.workspace import Workspace
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.utils.logging import get_logger


class Phase(Enum):
    """When in a step a channel acts, relative to the motion command."""

    BEFORE = "before"
    """Before the motion goes out, as a grasp that must close first."""

    WITH = "with"
    """In the same command as the motion, as a gripper on the arm's bus."""

    AFTER = "after"
    """After the motion, as a gripper the arm's controller does not carry."""


@dataclass(frozen=True)
class Effect:
    """What one channel's command did, beyond asking for a position.

    Attributes:
        role: The role the channel drives.
        changed: The end effector changed state, which a task may be charged
            for.
        is_hand: The end effector is a hand, which is never charged.
    """

    role: str
    changed: bool = False
    is_hand: bool = False


@dataclass(frozen=True)
class Applied:
    """What a whole action did, one entry per channel that reports anything."""

    effects: tuple[Effect, ...] = ()

    @property
    def penalties(self) -> int:
        """End effectors that changed state and may be charged for it."""
        return sum(
            1 for effect in self.effects if effect.changed and not effect.is_hand
        )

    def effect(self, role: str = "arm") -> Optional[Effect]:
        """What the channel driving ``role`` did, if it reported anything."""
        return next((effect for effect in self.effects if effect.role == role), None)


@dataclass(frozen=True)
class Command:
    """A channel's answer for one step.

    Attributes:
        send: Nested part action, merged with every other command of the same
            phase into one ``send_action``.
        effect: What the command did, for a task that scores it.
        defer: Work the step must not wait for, run on the role's own queue in
            the order it was handed over. A channel driving a part whose
            command takes longer than the control period returns the decision
            here and the effect above, so the step still scores it.
    """

    send: Mapping[str, Any] = field(default_factory=dict)
    effect: Optional[Effect] = None
    defer: Optional[Callable[[], None]] = None


class Channel(ABC):
    """One contiguous slice of the policy's action.

    Args:
        role: The role whose parts this channel drives.
        name: Name of the action part, as teleoperation and action wrappers
            refer to it.
        width: Numbers this channel takes from the action.
        kind: What those numbers mean.
        phase: When the channel acts within a step.
    """

    def __init__(
        self,
        *,
        role: str,
        name: str,
        width: int,
        kind: ActionKind,
        phase: Phase,
    ) -> None:
        self.role = role
        self.name = name
        self.width = width
        self.kind = kind
        self.phase = phase

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lowest and highest value of each number in the slice."""

    def requirements(self) -> Mapping[str, Needs]:
        """What the role's parts must accept for this channel to drive them."""
        return {}

    def confine(self, workspace: Optional[Workspace]) -> None:
        """Keep whatever this channel commands inside a task's workspace."""

    def reset(self, parts: Optional[Parts]) -> None:
        """Forget the previous episode, and set the parts up for the next."""

    def prepare(self, parts: Parts) -> None:
        """Ready the parts for this phase's command, as clearing a fault."""

    @abstractmethod
    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Turn this channel's slice into a command."""

    def teleop_context(self, parts: Optional[Parts]) -> Mapping[str, Any]:
        """What a teleoperation device needs to line its commands up with this."""
        return {}

    # End-effector verbs a task's reset uses, through the policy's own channel.

    def grasp(self, parts: Parts) -> bool:
        """Close, as a fully closing action would. ``False`` if not applicable."""
        return False

    def release(self, parts: Parts) -> bool:
        """Open, as a fully opening action would. ``False`` if not applicable."""
        return False

    def rest(self, parts: Parts) -> None:
        """Go to the resting state, where the channel has one."""


class ActionLayout:
    """The channels a policy's action is cut into, in order.

    Args:
        channels: The channels, in the order they occupy the action vector.
        wrappers: Registered action wrappers that fit this layout.
        transforms: Registered observation and action transforms that fit it.
    """

    def __init__(
        self,
        channels: Sequence[Channel],
        *,
        wrappers: Sequence[str] = (),
        transforms: Sequence[str] = (),
    ) -> None:
        if not channels:
            raise ValueError("An action layout needs at least one channel.")
        self.channels = tuple(channels)
        self.wrappers = tuple(wrappers)
        self.transforms = tuple(transforms)
        # One worker per role, so deferred commands to a part stay in the
        # order they were handed over while different roles run at once.
        self._queues: dict[str, ThreadPoolExecutor] = {}
        self._pending: list[Future[None]] = []
        self._suspended: frozenset[str] = frozenset()

    @property
    def width(self) -> int:
        """Numbers the policy produces."""
        return sum(channel.width for channel in self.channels)

    def space(self) -> gym.spaces.Box:
        """Bounds of the whole action, channel by channel."""
        low = np.concatenate([channel.bounds()[0] for channel in self.channels])
        high = np.concatenate([channel.bounds()[1] for channel in self.channels])
        return gym.spaces.Box(low.astype(np.float32), high.astype(np.float32))

    def parts(self) -> tuple[ActionPart, ...]:
        """Named slices of the action, in order, tiling its whole width."""
        return tuple(
            ActionPart(channel.name, channel.width, channel.kind)
            for channel in self.channels
        )

    def requirements(self) -> dict[str, Needs]:
        """What every channel needs of the robot, merged by role."""
        return combine(*(channel.requirements() for channel in self.channels))

    def confine(
        self, workspace: "Optional[Union[Workspace, Mapping[str, Workspace]]]"
    ) -> None:
        """Hand the task's workspace to the channels that command poses.

        One workspace confines every role. A task whose roles reach different
        parts of the bench keys them by role instead, and a role it leaves out
        is unconfined.
        """
        for channel in self.channels:
            if isinstance(workspace, Mapping):
                channel.confine(workspace.get(channel.role))
            else:
                channel.confine(workspace)

    def reset(self, parts: Optional[Parts] = None) -> None:
        """Forget the previous episode and set the parts up for the next."""
        # A reset commands the same parts, so let the last episode's deferred
        # commands land before it reads or moves them.
        self.drain()
        for channel in self.channels:
            channel.reset(parts)

    def defer(self, role: str, call: Callable[[], None]) -> None:
        """Run ``call`` on ``role``'s queue without waiting for it.

        Commands handed to one role run in order. A failure is logged rather
        than raised, because by then the step that caused it has returned.
        """
        queue = self._queues.get(role)
        if queue is None:
            queue = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"action-{role}"
            )
            self._queues[role] = queue
        future = queue.submit(call)
        future.add_done_callback(lambda done: self._report(role, done))
        self._pending.append(future)

    @staticmethod
    def _report(role: str, future: "Future[None]") -> None:
        """Log a deferred command that failed after its step returned."""
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            get_logger().error("Deferred command for role %r failed: %r", role, error)

    def drain(self) -> None:
        """Wait for every deferred command handed over so far."""
        pending, self._pending = self._pending, []
        for future in pending:
            try:
                future.result()
            except BaseException:  # noqa: BLE001 - already logged by _report
                pass

    def close(self) -> None:
        """Stop the deferral queues, letting queued commands finish."""
        self.drain()
        for queue in self._queues.values():
            queue.shutdown(wait=True)
        self._queues.clear()

    def suspend(self, names: Sequence[str] = ()) -> None:
        """Leave these action parts to whoever else is commanding them.

        A teleoperation device that streams its own commands between steps
        names the parts it delivers, and the step stops commanding those so
        the two do not fight over one controller. Calling this with nothing
        hands them back.
        """
        unknown = set(names) - {channel.name for channel in self.channels}
        if unknown:
            raise KeyError(
                f"This layout has no action parts {sorted(unknown)}; it has "
                f"{sorted(channel.name for channel in self.channels)}."
            )
        self._suspended = frozenset(names)

    def apply(self, parts: Parts, action: np.ndarray, reading: Reading) -> Applied:
        """Command every channel from one action, phase by phase."""
        effects: list[Effect] = []
        for phase in Phase:
            pending: dict[str, Any] = {}
            for channel, values in self._slices(action):
                if channel.phase is not phase or channel.name in self._suspended:
                    continue
                channel.prepare(parts)
                command = channel.command(parts, values, reading)
                if command.send:
                    merge(pending, dict(command.send))
                if command.effect is not None:
                    effects.append(command.effect)
                if command.defer is not None:
                    self.defer(channel.role, command.defer)
            if pending:
                parts.robot.send_action(pending)
        # Deferred commands that are already done should not pile up.
        self._pending = [future for future in self._pending if not future.done()]
        return Applied(tuple(effects))

    # What a task's reset and a teleoperation device ask of the channels.

    def grasp(self, parts: Parts) -> bool:
        """Close every end effector, through the policy's own channel."""
        # Every channel acts; a robot with two grippers closes both.
        closed = [channel.grasp(parts) for channel in self.channels]
        return any(closed)

    def release(self, parts: Parts) -> bool:
        """Open every end effector, through the policy's own channel."""
        opened = [channel.release(parts) for channel in self.channels]
        return any(opened)

    def rest_end_effectors(self, parts: Parts) -> None:
        """Put every end effector at its resting state."""
        for channel in self.channels:
            channel.rest(parts)

    def teleop_context(self, parts: Optional[Parts]) -> dict[str, Any]:
        """Everything the channels offer a teleoperation device."""
        context: dict[str, Any] = {}
        for channel in self.channels:
            context.update(channel.teleop_context(parts))
        return context

    def dof(self) -> dict[str, Optional[int]]:
        """Joints each role's arm drives, where a channel commands joints."""
        return {
            channel.role: channel.width
            for channel in self.channels
            if channel.kind is ActionKind.JOINT_POSITION
        }

    def joint_limits(
        self, role: str = "arm"
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Joint bounds for ``role`` in radians, where a channel keeps them."""
        for channel in self.channels:
            if channel.role == role and channel.kind is ActionKind.JOINT_POSITION:
                return channel.bounds()
        return None

    def _slices(self, action: np.ndarray) -> Iterator[tuple[Channel, np.ndarray]]:
        start = 0
        for channel in self.channels:
            yield channel, action[start : start + channel.width]
            start += channel.width
