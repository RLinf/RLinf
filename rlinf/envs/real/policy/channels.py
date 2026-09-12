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

"""The channels a robot's action is made of, and the settings a run gives them.

Each channel drives one role's part: joints, a tool pose, a gripper, a hand.
A robot's layout is a list of them, so a second arm is a second set of the
same channels rather than a second env.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.tasks.requirements import Needs, Parts, Reading, nest
from rlinf.envs.real.tasks.workspace import Workspace
from rlinf.robotics.actions import ActionKind
from rlinf.utils.rot6d import rot6d_to_quat_xyzw_safe

from .action import Channel, Command, Effect, Phase

#: How a rotation delta reaches the tool's orientation.
ROTATIONS = ("compose", "euler_add")


@dataclass
class JointActionConfig:
    """Joint bounds for an absolute joint-position action, in radians."""

    joint_limit_low: Optional[Sequence[float]] = None
    """Lowest joint target a policy may send. A robot preset fills this in."""

    joint_limit_high: Optional[Sequence[float]] = None
    """Highest joint target a policy may send. A robot preset fills this in."""


@dataclass
class PoseActionConfig:
    """How far one action moves the tool, and how its end effector is driven."""

    action_scale: Sequence[float] = (1.0, 1.0, 1.0)
    """Metres per unit of position action, radians per unit of rotation
    action, and the multiplier on the gripper channel."""

    compliance_param: Mapping[str, float] = field(default_factory=dict)
    """Impedance gains applied at the start of every episode. Empty leaves the
    arm's controller as it is."""

    binary_gripper_threshold: float = 0.5
    """Gripper channel magnitude at which the gripper opens or closes."""

    hand_action_scale: float = 1.0
    """Multiplier from a hand channel to a finger target."""

    hand_max_delta_per_step: float = float("inf")
    """Largest change of one finger target between two steps."""

    hand_reset_state: Sequence[float] = (0.0,) * 6
    """Finger pose a hand rests at between episodes."""

    def __post_init__(self) -> None:
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)
        self.hand_reset_state = np.asarray(self.hand_reset_state, dtype=np.float64)
        self.compliance_param = dict(self.compliance_param)


class JointPositions(Channel):
    """Joint targets for one arm, in radians.

    The channel commands the joints directly, so its bounds are the arm's
    joint limits. With ``scale`` set it commands a change instead: the action
    is one unit either way, multiplied by ``scale`` radians and added to the
    joints as they were read, then held inside the same limits.

    Args:
        role: The role whose arm this channel drives.
        low: Lowest target per joint.
        high: Highest target per joint.
        dof: Joints the arm has, when the robot knows; the bounds are checked
            against it so a mis-sized limit fails before anything connects.
        scale: Radians per action unit, for an arm commanded by change. ``None``
            commands the joints directly.
        compliance: Impedance gains applied to the arm at each reset. An arm
            running a Cartesian impedance controller underneath keeps them
            whichever way a policy commands it.
        name: Action part name; the role's own name by default.
    """

    def __init__(
        self,
        role: str,
        *,
        low: Optional[Sequence[float]],
        high: Optional[Sequence[float]],
        dof: Optional[int] = None,
        scale: Optional[float] = None,
        compliance: Optional[Mapping[str, float]] = None,
        name: Optional[str] = None,
    ) -> None:
        if low is None or high is None:
            raise ValueError(
                "A joint channel needs 'joint_limit_low' and 'joint_limit_high'."
            )
        self._low = np.asarray(low, dtype=np.float64)
        self._high = np.asarray(high, dtype=np.float64)
        if self._low.shape != self._high.shape or self._low.ndim != 1:
            raise ValueError(
                "Joint bounds must be one value per joint on both sides, got "
                f"{self._low.shape} and {self._high.shape}."
            )
        for label, bound in (
            ("joint_limit_low", self._low),
            ("joint_limit_high", self._high),
        ):
            if dof is not None and bound.shape != (dof,):
                raise ValueError(
                    f"The arm has {dof} joints, so {label!r} needs {dof} values, "
                    f"got {bound.size}."
                )
        self.scale = scale
        self.compliance = dict(compliance or {})
        super().__init__(
            role=role,
            name=name or role,
            width=int(self._low.size),
            kind=ActionKind.JOINT_DELTA
            if scale is not None
            else ActionKind.JOINT_POSITION,
            phase=Phase.WITH,
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The joint limits, or one unit either way when commanded by change."""
        if self.scale is None:
            return self._low, self._high
        return -np.ones(self._low.size), np.ones(self._low.size)

    def requirements(self) -> Mapping[str, Needs]:
        """An arm that accepts joint targets, and reports them when relative."""
        needs = Needs(commands=frozenset({"joint_position"}))
        if self.scale is not None:
            needs = Needs(
                observes=frozenset({"arm_joint_position"}),
                commands=frozenset({"joint_position"}),
            )
        return {self.role: needs}

    def reset(self, parts: Optional[Parts]) -> None:
        """Apply the episode's impedance gains."""
        if parts is not None and self.compliance:
            parts.arm(self.role).reconfigure_compliance_params(self.compliance)

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Ask the arm for these joints, or for this change to them."""
        target = values
        if self.scale is not None:
            current = np.asarray(
                reading.arm(self.role)["arm_joint_position"], dtype=np.float64
            )
            target = np.clip(current + values * self.scale, self._low, self._high)
        path = parts.bound[self.role].part
        return Command(send=nest(path, {"joint_position": target}))


class PoseDelta(Channel):
    """A tool-pose delta for one arm: three positions, then three rotations.

    The position delta is scaled and added to the tool's position. The
    rotation delta is scaled and either composed onto the tool's orientation
    or added to its Euler angles, which is what an arm whose controller
    thinks in Euler angles was trained against. The result is kept inside the
    task's workspace and sent as a pose target.

    Args:
        role: The role whose arm this channel drives.
        scales: Position, rotation, and gripper scales, as a run gives them.
        rotation: ``"compose"`` or ``"euler_add"``.
        compliance: Impedance gains to apply at the start of an episode.
        clear_errors: Clear a latched fault before every command, for a
            controller that stops accepting targets after one.
        name: Action part name; the role's own name by default.
    """

    def __init__(
        self,
        role: str,
        *,
        scales: Sequence[float] = (1.0, 1.0, 1.0),
        rotation: str = "compose",
        compliance: Optional[Mapping[str, float]] = None,
        clear_errors: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}, got {rotation!r}.")
        super().__init__(
            role=role,
            name=name or role,
            width=6,
            kind=ActionKind.CARTESIAN_DELTA,
            phase=Phase.WITH,
        )
        self.scales = np.asarray(scales, dtype=np.float64)
        self.rotation = rotation
        self.compliance = dict(compliance or {})
        self.clear_errors = clear_errors
        self.workspace: Optional[Workspace] = None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """One unit either way on every axis."""
        return -np.ones(6), np.ones(6)

    def requirements(self) -> Mapping[str, Needs]:
        """An arm that reports a tool pose and accepts one."""
        return {
            self.role: Needs(
                observes=frozenset({"tcp_pose"}),
                commands=frozenset({"tcp_pose"}),
            )
        }

    def confine(self, workspace: Optional[Workspace]) -> None:
        """Keep every commanded pose inside ``workspace``."""
        self.workspace = workspace

    def reset(self, parts: Optional[Parts]) -> None:
        """Apply the episode's impedance gains."""
        if parts is not None:
            parts.arm(self.role).reconfigure_compliance_params(self.compliance)

    def prepare(self, parts: Parts) -> None:
        """Clear a fault the last motion latched."""
        if self.clear_errors:
            parts.arm(self.role).clear_errors()

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Move the tool one delta from where it was read."""
        current = np.asarray(reading.arm(self.role)["tcp_pose"], dtype=np.float64)
        target = current.copy()
        target[:3] += values[:3] * self.scales[0]
        delta = values[3:6] * self.scales[1]
        if self.rotation == "compose":
            target[3:] = (
                R.from_euler("xyz", delta) * R.from_quat(current[3:])
            ).as_quat()
        else:
            euler = R.from_quat(current[3:]).as_euler("xyz") + delta
            target[3:] = R.from_euler("xyz", euler).as_quat()
        if self.workspace is not None:
            target = self.workspace.clip(target, current)
        path = parts.bound[self.role].part
        return Command(send=nest(path, {"tcp_pose": target.astype(np.float32)}))

    def teleop_context(self, parts: Optional[Parts]) -> Mapping[str, Any]:
        """The scales a delta device multiplies its own reading by."""
        return {"action_scale": self.scales}


class PoseTarget(Channel):
    """A tool pose for one arm, commanded directly rather than as a change.

    The channel takes a position and a six-number rotation, the first two
    columns of the rotation matrix, which is what a policy trained to emit
    waypoints regresses. The rotation is turned back into a quaternion on the
    hemisphere the last command used, so a controller interpolating between
    two steps takes the short way round.

    Because the channel asks for a position rather than a change to one, its
    bounds are the reachable positions themselves. Those are the task's, so
    they come from the workspace the task confines this channel to, and
    ``low`` and ``high`` are only for a layout used without one.

    Args:
        role: The role whose arm this channel drives.
        low: Lowest position the channel may ask for. ``None`` takes the
            task's workspace.
        high: Highest position the channel may ask for. ``None`` likewise.
        rotation_limit: Bound on each of the six rotation numbers, left wide of
            one so a policy's output survives normalisation.
        compliance: Impedance gains applied to the arm at each reset.
        clear_errors: Clear a latched fault before each command. Off by
            default: a controller that needs it says so, and asking every arm
            to recover at the control rate is traffic most of them do not
            need.
        name: Action part name; the role's own name by default.
    """

    def __init__(
        self,
        role: str,
        *,
        low: Optional[Sequence[float]] = None,
        high: Optional[Sequence[float]] = None,
        rotation_limit: float = 1.5,
        compliance: Optional[Mapping[str, float]] = None,
        clear_errors: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self._low = None if low is None else np.asarray(low, dtype=np.float64)
        self._high = None if high is None else np.asarray(high, dtype=np.float64)
        for label, bound in (("low", self._low), ("high", self._high)):
            if bound is not None and bound.shape != (3,):
                raise ValueError(
                    f"A pose target's {label!r} is three position values, got "
                    f"shape {bound.shape}."
                )
        self.rotation_limit = float(rotation_limit)
        self.compliance = dict(compliance or {})
        self.clear_errors = clear_errors
        self.workspace: Optional[Workspace] = None
        self._last_quat: Optional[np.ndarray] = None
        super().__init__(
            role=role,
            name=name or role,
            width=9,
            kind=ActionKind.CARTESIAN_POSE,
            phase=Phase.WITH,
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The position box, then the rotation numbers either way.

        Raises:
            ValueError: If neither this channel nor the task bounds the
                positions it may ask for.
        """
        low, high = self._low, self._high
        if low is None or high is None:
            if self.workspace is None:
                raise ValueError(
                    f"PoseTarget({self.role!r}) asks for positions directly, so "
                    "it needs bounds: give the task a workspace, or pass 'low' "
                    "and 'high'."
                )
            low = np.asarray(self.workspace.low[:3], dtype=np.float64)
            high = np.asarray(self.workspace.high[:3], dtype=np.float64)
        limit = np.full(6, self.rotation_limit)
        return (
            np.concatenate([low, -limit]),
            np.concatenate([high, limit]),
        )

    def requirements(self) -> Mapping[str, Needs]:
        """An arm that reports a tool pose and accepts one."""
        return {
            self.role: Needs(
                observes=frozenset({"tcp_pose"}),
                commands=frozenset({"tcp_pose"}),
            )
        }

    def confine(self, workspace: Optional[Workspace]) -> None:
        """Keep every commanded pose inside ``workspace``."""
        self.workspace = workspace

    def reset(self, parts: Optional[Parts]) -> None:
        """Forget the hemisphere and apply the episode's impedance gains."""
        self._last_quat = None
        if parts is not None:
            parts.arm(self.role).reconfigure_compliance_params(self.compliance)

    def prepare(self, parts: Parts) -> None:
        """Clear a fault the last motion latched."""
        if self.clear_errors:
            parts.arm(self.role).clear_errors()

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Send this pose, on the hemisphere the last command used."""
        current = np.asarray(reading.arm(self.role)["tcp_pose"], dtype=np.float64)
        previous = self._last_quat if self._last_quat is not None else current[3:]
        quat = rot6d_to_quat_xyzw_safe(values[3:9], fallback_quat_xyzw=previous)
        if float(np.dot(quat, previous)) < 0.0:
            quat = -quat
        self._last_quat = quat
        target = np.concatenate([values[:3], quat]).astype(np.float64)
        if self.workspace is not None:
            target = self.workspace.clip(target, current)
        path = parts.bound[self.role].part
        return Command(send=nest(path, {"tcp_pose": target}))


class ContinuousGripper(Channel):
    """A gripper opened to any fraction of its stroke, on the arm's command.

    Args:
        role: The role whose end effector this channel drives.
        low: Opening at the bottom of the channel.
        high: Opening at the top of it.
        moved_tolerance: Change in opening between two steps that counts as
            the gripper moving, for a task's gripper penalty.
        name: Action part name.
    """

    def __init__(
        self,
        role: str = "arm",
        *,
        low: float = 0.0,
        high: float = 1.0,
        moved_tolerance: float = 0.05,
        name: str = "end_effector",
    ) -> None:
        super().__init__(
            role=role, name=name, width=1, kind=ActionKind.GRIPPER, phase=Phase.WITH
        )
        self.low = low
        self.high = high
        self.moved_tolerance = moved_tolerance
        self._last: Optional[float] = None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The stroke this channel spans."""
        return np.array([self.low]), np.array([self.high])

    def requirements(self) -> Mapping[str, Needs]:
        """A gripper to open."""
        return {self.role: Needs(end_effector="gripper")}

    def reset(self, parts: Optional[Parts]) -> None:
        """Forget the previous episode's opening."""
        self._last = None

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Open the gripper to this fraction, alongside the arm's command."""
        opening = float(np.clip(values[0], self.low, self.high))
        moved = (
            self._last is not None and abs(opening - self._last) > self.moved_tolerance
        )
        self._last = opening
        path = parts.bound[self.role].end_effector
        return Command(
            send=nest(path, {"target": np.array([opening])}),
            effect=Effect(self.role, changed=moved),
        )


class GripperWidth(Channel):
    """A gripper opened to a width the policy asks for, in the driver's units.

    Unlike :class:`ContinuousGripper`, which spans a normalised stroke, this
    channel's numbers are the opening itself, which is what a policy trained
    against a controller that reports a width was trained on.

    Args:
        role: The role whose end effector this channel drives.
        low: Narrowest opening the gripper accepts.
        high: Widest opening it accepts.
        action_range: What the policy emits, when that is not the opening
            range itself. A checkpoint trained with every channel normalised
            sends one unit either way, and the opening it asks for is then
            held inside ``low`` and ``high`` like any other.
        enforce_close: Ignore the channel and hold the gripper at ``low``, for
            a task that must not let go of what it is holding.
        moved_tolerance: Change in opening between two steps that counts as
            the gripper moving, for a task's gripper penalty.
        name: Action part name.
    """

    def __init__(
        self,
        role: str = "arm",
        *,
        low: float = 0.0,
        high: float = 1.0,
        action_range: Optional[tuple[float, float]] = None,
        enforce_close: bool = False,
        moved_tolerance: float = 0.05,
        name: str = "end_effector",
    ) -> None:
        super().__init__(
            role=role, name=name, width=1, kind=ActionKind.GRIPPER, phase=Phase.WITH
        )
        self.low = low
        self.high = high
        self.action_range = action_range
        self.enforce_close = enforce_close
        self.moved_tolerance = moved_tolerance

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """What the policy emits: the opening range, or its own."""
        low, high = self.action_range or (self.low, self.high)
        return np.array([low]), np.array([high])

    def requirements(self) -> Mapping[str, Needs]:
        """A gripper that reports its opening and takes one."""
        return {self.role: Needs(end_effector="gripper")}

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Open the gripper this wide, or hold it shut."""
        target = self.low if self.enforce_close else float(values[0])
        target = float(np.clip(target, self.low, self.high))
        effector = parts.end_effector(self.role)
        before = reading.end_effector(self.role)
        opening = None if before is None else np.asarray(before["state"]).reshape(-1)[0]
        effector.command(np.array([target], dtype=np.float32))
        moved = opening is not None and abs(target - float(opening)) > (
            self.moved_tolerance
        )
        return Command(effect=Effect(self.role, changed=bool(moved)))


class BinaryGripper(Channel):
    """A gripper that is either open or closed.

    The channel closes the gripper at ``-threshold`` or below and opens it at
    ``threshold`` or above; anything between leaves it as it is. A command
    that would not change the gripper is not sent, so a policy holding the
    channel down does not re-grasp every step.

    Args:
        role: The role whose end effector this channel drives.
        threshold: Magnitude at which the channel acts.
        scale: Multiplier applied to the channel before the threshold, as a
            run's gripper action scale.
        settle_s: Seconds to wait after a change for the fingers to finish.
        phase: Whether the gripper acts before or after the arm's motion.
        fitted: Whether the robot has the gripper. A rig without one keeps the
            channel, so the policy's action layout does not depend on the rig,
            and ignores it.
        awaited: Whether a step waits for the fingers. A gripper slower than
            the control period is commanded on the role's queue instead, and
            the step reports what it asked for without blocking on it.
        name: Action part name.
    """

    def __init__(
        self,
        role: str = "arm",
        *,
        threshold: float = 0.5,
        scale: float = 1.0,
        settle_s: float = 0.6,
        phase: Phase = Phase.BEFORE,
        fitted: bool = True,
        awaited: bool = True,
        name: str = "end_effector",
    ) -> None:
        super().__init__(
            role=role, name=name, width=1, kind=ActionKind.GRIPPER, phase=phase
        )
        self.threshold = threshold
        self.scale = scale
        self.settle_s = settle_s
        self.fitted = fitted
        self.awaited = awaited

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """One unit either way: closed at the bottom, open at the top."""
        return np.array([-1.0]), np.array([1.0])

    def requirements(self) -> Mapping[str, Needs]:
        """A gripper to latch, when the rig has one."""
        return {self.role: Needs(end_effector="gripper")} if self.fitted else {}

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Open or close the gripper, if this value asks for the other state."""
        act, effect = self._decide(parts, float(values[0]) * self.scale)
        if act is None:
            return Command(effect=effect)
        if self.awaited:
            act()
            return Command(effect=effect)
        return Command(effect=effect, defer=act)

    def grasp(self, parts: Parts) -> bool:
        """Close the gripper as a fully closing action would."""
        effect = self._latch(parts, -1.0 * self.scale)
        return bool(effect and effect.changed)

    def release(self, parts: Parts) -> bool:
        """Open the gripper as a fully opening action would."""
        effect = self._latch(parts, 1.0 * self.scale)
        return bool(effect and effect.changed)

    def teleop_context(self, parts: Optional[Parts]) -> Mapping[str, Any]:
        """Whether the gripper is open, for a device that toggles it."""
        if parts is None or not self.fitted:
            return {}
        return {"gripper_open": bool(parts.end_effector(self.role).is_open)}

    def _decide(
        self, parts: Parts, value: float
    ) -> tuple[Optional[Callable[[], None]], Optional[Effect]]:
        """What this value asks of the gripper, and the effect it has.

        Deciding is separate from acting so that a caller which does not wait
        for the fingers still learns, in the step that asked, whether the
        gripper was asked to change.
        """
        if not self.fitted:
            return None, None
        effector = parts.end_effector(self.role)
        if value <= -self.threshold and effector.is_open:
            latch = effector.close
        elif value >= self.threshold and not effector.is_open:
            latch = effector.open
        else:
            return None, Effect(self.role, changed=False)

        def act() -> None:
            latch()
            time.sleep(self.settle_s)

        return act, Effect(self.role, changed=True)

    def _latch(self, parts: Parts, value: float) -> Optional[Effect]:
        """Act on ``value`` and wait for the fingers, as a reset does."""
        act, effect = self._decide(parts, value)
        if act is not None:
            act()
        return effect


class HandCommand(Channel):
    """Finger targets for a dexterous hand, scaled and rate limited.

    Args:
        role: The role whose end effector this channel drives.
        dim: Fingers the hand drives, one number each.
        scale: Multiplier from a channel value to a finger target.
        max_delta: Largest change of one finger target between two commands.
        reset_state: Finger pose the hand rests at between episodes.
        name: Action part name.
    """

    def __init__(
        self,
        role: str = "arm",
        *,
        dim: int,
        scale: float = 1.0,
        max_delta: float = float("inf"),
        reset_state: Optional[Sequence[float]] = None,
        name: str = "hand",
    ) -> None:
        super().__init__(
            role=role, name=name, width=dim, kind=ActionKind.HAND, phase=Phase.BEFORE
        )
        self.scale = scale
        self.max_delta = max_delta
        self.reset_state = np.asarray(
            np.zeros(dim) if reset_state is None else reset_state, dtype=np.float64
        )
        self._last: Optional[np.ndarray] = None

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """One unit either way on every finger."""
        return -np.ones(self.width), np.ones(self.width)

    def requirements(self) -> Mapping[str, Needs]:
        """A hand to pose."""
        return {self.role: Needs(end_effector="hand")}

    def command(self, parts: Parts, values: np.ndarray, reading: Reading) -> Command:
        """Pose the fingers, moving each at most ``max_delta`` from the last."""
        return Command(effect=self._pose(parts, values))

    def release(self, parts: Parts) -> bool:
        """Bring the fingers to their resting pose, as letting go."""
        self._pose(parts, self.reset_state)
        return False

    def rest(self, parts: Parts) -> None:
        """Put the hand at its resting pose and limit the next command from it."""
        parts.end_effector(self.role).reset(self.reset_state)
        self._last = self.reset_state * self.scale

    def teleop_context(self, parts: Optional[Parts]) -> Mapping[str, Any]:
        """The pose a glove starts its own fingers from."""
        return {"hand_reset_pose": self.reset_state}

    def _pose(self, parts: Parts, values: np.ndarray) -> Effect:
        target = np.asarray(values, dtype=np.float64) * self.scale
        if self._last is not None:
            step = np.clip(target - self._last, -self.max_delta, self.max_delta)
            target = self._last + step
        self._last = target.copy()
        parts.end_effector(self.role).command(target)
        return Effect(self.role, changed=True, is_hand=True)
