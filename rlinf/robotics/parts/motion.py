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

"""Joint motion shaping shared by drivers that stream position setpoints.

``MotionProfile`` and ``JointGroupSpec`` hold a connection's rate and physical
limits. ``SetpointStreamer`` turns irregular commands into a continuous,
limited setpoint sequence, and ``JointStream`` runs it from a fixed-rate
background loop that a driver feeds through a few callables.
"""

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

import numpy as np

from rlinf.utils.logging import get_logger

#: Time constant of the feedback that pulls a setpoint onto its goal.
_FEEDBACK_TIME_S = 0.05

#: Goals closer together than this are one command sent part by part, such as
#: two arms updated one after the other.
_COALESCE_S = 0.002


def _limit_vector(
    value: "float | Sequence[float] | np.ndarray", dim: int
) -> np.ndarray:
    """Broadcast a scalar or per-joint limit to a positive ``(dim,)`` float64 array."""
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (dim,)).copy()
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"Motion limits must be finite and positive, got {arr!r}.")
    return arr


@dataclass
class MotionProfile:
    """Control rate and physical limits for a connection's joint motion.

    Limits are in physical units, so the motion a profile allows does not
    depend on how often frames are sent.

    Attributes:
        control_rate_hz: Frequency of the streamed setpoint loop in Hz.
        max_joint_velocity: Joint speed limit in rad/s.
        max_joint_acceleration: Joint acceleration limit in rad/s^2.
        max_extrapolation_s: Longest a streamed goal is carried forward at its
            last velocity while the next command is awaited, in seconds.
        effector_settle_s: Time a point-to-point move keeps holding after an
            end effector changes target, in seconds.
        pulse_rate_hz: Streaming frequency in Hz for base velocity pulses.
        pulse_ramp_up: Longest acceleration ramp of a velocity pulse, in seconds.
        pulse_ramp_down: Longest deceleration ramp of a velocity pulse, in seconds.
        pulse_ramp_up_ratio: Largest fraction of a pulse spent ramping up.
        pulse_ramp_down_ratio: Largest fraction of a pulse spent ramping down.
    """

    control_rate_hz: float = 100.0
    max_joint_velocity: float = 1.0
    max_joint_acceleration: float = 4.0
    max_extrapolation_s: float = 0.1
    effector_settle_s: float = 0.4
    pulse_rate_hz: float = 50.0
    pulse_ramp_up: float = 0.25
    pulse_ramp_down: float = 0.35
    pulse_ramp_up_ratio: float = 0.3
    pulse_ramp_down_ratio: float = 0.4

    def __post_init__(self) -> None:
        """Reject limits that would stall or unbound the motion they shape."""
        for name in (
            "control_rate_hz",
            "max_joint_velocity",
            "max_joint_acceleration",
            "pulse_rate_hz",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"MotionProfile.{name} must be finite and positive, got {value!r}."
                )
        for name in ("max_extrapolation_s", "effector_settle_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"MotionProfile.{name} must be finite and non-negative, "
                    f"got {value!r}."
                )

    @property
    def dt(self) -> float:
        """Control cycle period in seconds derived from :attr:`control_rate_hz`."""
        return 1.0 / float(self.control_rate_hz)

    @property
    def pulse_dt(self) -> float:
        """Velocity pulse period in seconds derived from :attr:`pulse_rate_hz`."""
        return 1.0 / float(self.pulse_rate_hz)

    @classmethod
    def from_config(
        cls,
        value: "Optional[MotionProfile | Mapping[str, Any]]" = None,
        *,
        defaults: "Optional[MotionProfile]" = None,
        **overrides: Any,
    ) -> "MotionProfile":
        """Resolve a profile or mapping against ``defaults`` and keyword overrides."""
        base = replace(defaults) if defaults is not None else cls()
        given: dict[str, Any] = {}
        if isinstance(value, cls):
            base = replace(value)
        elif value is not None:
            given.update(dict(value))
        for key, val in overrides.items():
            if val is not None:
                given[key] = val
        if not given:
            return base
        known = {f.name for f in fields(cls)}
        unknown = set(given) - known
        if unknown:
            raise KeyError(
                f"Unknown MotionProfile settings {sorted(unknown)}. "
                f"Known: {sorted(known)}."
            )
        return replace(base, **{k: float(v) for k, v in given.items()})

    @classmethod
    def raised_cosine_alpha(cls, step_index: int, total_steps: int) -> float:
        """Return the raised-cosine ease-in/ease-out factor in ``[0.0, 1.0]``."""
        if total_steps <= 1:
            return 1.0
        clamped = min(max(int(step_index), 0), int(total_steps))
        return 0.5 * (1.0 - float(np.cos(np.pi * (clamped / float(total_steps)))))

    @classmethod
    def raised_cosine_decay(cls, step_index: int, total_steps: int) -> float:
        """Return the raised-cosine ease-out decay factor from ``1.0`` to ``0.0``."""
        if total_steps <= 0:
            return 0.0
        clamped = min(max(int(step_index), 0), int(total_steps))
        return 0.5 * (1.0 + float(np.cos(np.pi * (clamped / float(total_steps)))))

    def move_duration(
        self,
        delta: "Sequence[float] | np.ndarray",
        *,
        max_velocity: "Optional[float | Sequence[float] | np.ndarray]" = None,
        max_acceleration: "Optional[float | Sequence[float] | np.ndarray]" = None,
        min_duration: float = 0.0,
    ) -> float:
        """Return the shortest raised-cosine duration for ``delta`` within the limits.

        A raised cosine over ``T`` seconds peaks at ``pi*d/(2T)`` velocity and
        ``pi**2*d/(2T**2)`` acceleration for a joint moving ``d`` radians.
        Per-joint limits default to this profile's and the result is never
        shorter than ``min_duration``.
        """
        d = np.abs(np.asarray(delta, dtype=np.float64).reshape(-1))
        if d.size == 0:
            return max(0.0, float(min_duration))
        v = _limit_vector(
            self.max_joint_velocity if max_velocity is None else max_velocity, d.size
        )
        a = _limit_vector(
            self.max_joint_acceleration
            if max_acceleration is None
            else max_acceleration,
            d.size,
        )
        by_velocity = float(np.max(np.pi * d / (2.0 * v)))
        by_acceleration = float(np.max(np.pi * np.sqrt(d / (2.0 * a))))
        return max(float(min_duration), by_velocity, by_acceleration)

    def generate_velocity_pulse(
        self,
        velocity: Sequence[float] | np.ndarray,
        duration: float,
        *,
        rate_hz: Optional[float] = None,
        ramp_up: Optional[float] = None,
        ramp_down: Optional[float] = None,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Generate ``(dt, up_frames, cruise_frames, down_frames)`` for a smooth velocity pulse."""
        vel = np.asarray(velocity, dtype=np.float32).copy()
        total_duration = max(float(duration), 0.0)
        eff_rate_hz = (
            max(float(rate_hz), float(self.pulse_rate_hz))
            if rate_hz is not None
            else float(self.pulse_rate_hz)
        )
        dt = 1.0 / max(eff_rate_hz, 1.0)
        up_dur = (
            float(ramp_up)
            if ramp_up is not None
            else min(
                float(self.pulse_ramp_up),
                float(self.pulse_ramp_up_ratio) * total_duration,
            )
        )
        down_dur = (
            float(ramp_down)
            if ramp_down is not None
            else min(
                float(self.pulse_ramp_down),
                float(self.pulse_ramp_down_ratio) * total_duration,
            )
        )
        up_steps = max(1, int(round(up_dur / dt))) if up_dur > 1e-3 else 0
        cruise_steps = max(1, int(round(total_duration / dt)))
        down_steps = max(1, int(round(down_dur / dt))) if down_dur > 1e-3 else 0

        up_frames = [
            (vel * self.raised_cosine_alpha(idx, up_steps)).astype(np.float32)
            for idx in range(1, up_steps + 1)
        ]
        cruise_frames = [vel.copy() for _ in range(cruise_steps)]
        down_frames = [
            (vel * self.raised_cosine_decay(idx, down_steps)).astype(np.float32)
            for idx in range(1, down_steps + 1)
        ]
        return dt, up_frames, cruise_frames, down_frames

    def to_dict(self) -> dict[str, Any]:
        """Serialize this motion profile into a plain dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class JointGroupSpec:
    """Position limits and optional tighter motion limits for a named joint group.

    Attributes:
        name: Logical group identifier (e.g. ``"left_arm"``).
        dim: Number of joints in this group.
        low: Optional lower joint limits in radians.
        high: Optional upper joint limits in radians.
        max_velocity: Optional speed limit in rad/s; the group moves at the
            lower of this and the profile's limit.
        max_acceleration: Optional acceleration limit in rad/s^2; the group
            uses the lower of this and the profile's limit.
    """

    name: str
    dim: int
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None

    def clamp(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Reshape ``values`` to ``(self.dim,)`` and clip to ``[self.low, self.high]``."""
        arr = np.asarray(values, dtype=np.float32).reshape(self.dim)
        if self.low is not None and self.high is not None:
            low_b = np.asarray(self.low, dtype=np.float32).reshape(-1)[: self.dim]
            high_b = np.asarray(self.high, dtype=np.float32).reshape(-1)[: self.dim]
            return np.clip(arr, low_b, high_b).astype(np.float32)
        return arr.copy()

    def velocity_limit(self, profile: MotionProfile) -> float:
        """Return this group's speed limit in rad/s under ``profile``."""
        limit = float(profile.max_joint_velocity)
        return limit if self.max_velocity is None else min(limit, self.max_velocity)

    def acceleration_limit(self, profile: MotionProfile) -> float:
        """Return this group's acceleration limit in rad/s^2 under ``profile``."""
        limit = float(profile.max_joint_acceleration)
        return (
            limit
            if self.max_acceleration is None
            else min(limit, self.max_acceleration)
        )


class SetpointStreamer:
    """Turn irregular joint commands into a continuous, limited setpoint stream.

    A connection calls :meth:`step` once per control period and sends the
    returned setpoint, while commands only move the goal (:meth:`set_goal`) or
    start a timed point-to-point move (:meth:`start_move`). The setpoint
    approaches the goal with per-joint velocity and acceleration limits,
    scaled together so the joints keep their direction in joint space.

    Commands usually arrive slower and less regularly than the control rate,
    so between two goals the latest one is carried forward at the velocity the
    last two implied, for at most one and a half command intervals and
    ``max_extrapolation_s``. That removes the staircase a setpoint-per-command
    stream shows. A goal that moves faster than the velocity limit is a jump,
    not a stream, and is approached without being carried forward. Goals set
    within a couple of milliseconds of each other count as one command, so a
    driver may update its parts one at a time.

    Not thread-safe: the owning connection serializes every call.

    Args:
        setpoint: The joints to hold at first.
        max_velocity: Per-joint (or shared) speed limit in rad/s.
        max_acceleration: Per-joint (or shared) acceleration limit in rad/s^2.
        max_extrapolation_s: Longest a goal is carried forward, in seconds.
    """

    def __init__(
        self,
        setpoint: "Sequence[float] | np.ndarray",
        *,
        max_velocity: "float | Sequence[float] | np.ndarray",
        max_acceleration: "float | Sequence[float] | np.ndarray",
        max_extrapolation_s: float,
    ) -> None:
        self._q = np.asarray(setpoint, dtype=np.float64).reshape(-1).copy()
        self._v = np.zeros_like(self._q)
        self.set_limits(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_extrapolation_s=max_extrapolation_s,
        )
        self._goal = self._q.copy()
        self._goal_time: Optional[float] = None
        self._goal_velocity = np.zeros_like(self._q)
        self._goal_interval = 0.0
        # The command before the latest one, which its velocity is measured from.
        self._last_goal: Optional[np.ndarray] = None
        self._last_goal_time: Optional[float] = None
        self._move: Optional[tuple[np.ndarray, np.ndarray, float, float]] = None
        self._move_done = threading.Event()
        self._move_done.set()
        self._move_completed = True

    @property
    def dim(self) -> int:
        """Number of joints this streamer drives."""
        return int(self._q.size)

    @property
    def setpoint(self) -> np.ndarray:
        """The setpoint the last :meth:`step` produced."""
        return self._q.copy()

    @property
    def goal(self) -> np.ndarray:
        """The latest goal, or the target of the running move."""
        return self._goal.copy()

    @property
    def moving(self) -> bool:
        """Whether a point-to-point move is in progress."""
        return self._move is not None

    @property
    def move_completed(self) -> bool:
        """Whether the last point-to-point move reached its target."""
        return self._move_completed

    def set_limits(
        self,
        *,
        max_velocity: "float | Sequence[float] | np.ndarray",
        max_acceleration: "float | Sequence[float] | np.ndarray",
        max_extrapolation_s: float,
    ) -> None:
        """Replace the motion limits; the current setpoint and goal are kept."""
        self._vmax = _limit_vector(max_velocity, self.dim)
        self._amax = _limit_vector(max_acceleration, self.dim)
        self._max_extrapolation_s = max(0.0, float(max_extrapolation_s))

    def reset(self, setpoint: "Sequence[float] | np.ndarray") -> None:
        """Hold still at ``setpoint``, forgetting the goal history and any move."""
        q = np.asarray(setpoint, dtype=np.float64).reshape(self.dim)
        self._q = q.copy()
        self._v = np.zeros_like(q)
        self._goal = q.copy()
        self._goal_time = None
        self._goal_velocity = np.zeros_like(q)
        self._goal_interval = 0.0
        self._last_goal = None
        self._last_goal_time = None
        self._finish_move(completed=False)

    def set_goal(self, goal: "Sequence[float] | np.ndarray", now: float) -> None:
        """Track ``goal`` from time ``now``, preempting a running move."""
        g = np.asarray(goal, dtype=np.float64).reshape(self.dim)
        self._finish_move(completed=False)
        coalesced = self._goal_time is not None and now - self._goal_time < _COALESCE_S
        if not coalesced:
            self._last_goal, self._last_goal_time = self._goal, self._goal_time
        if self._last_goal_time is not None and now > self._last_goal_time:
            assert self._last_goal is not None
            interval = now - self._last_goal_time
            implied = (g - self._last_goal) / interval
            jump = bool(np.any(np.abs(implied) > self._vmax))
            self._goal_velocity = np.zeros_like(g) if jump else implied
            self._goal_interval = interval
        else:
            self._goal_velocity = np.zeros_like(g)
            self._goal_interval = 0.0
        self._goal = g.copy()
        self._goal_time = now

    def start_move(
        self,
        target: "Sequence[float] | np.ndarray",
        now: float,
        duration: float,
    ) -> threading.Event:
        """Move to ``target`` along a raised cosine of ``duration`` seconds.

        Returns:
            An event set when the move ends, by arriving or by being
            preempted, aborted, or reset; :attr:`move_completed` says which.
        """
        t = np.asarray(target, dtype=np.float64).reshape(self.dim)
        self._finish_move(completed=False)
        self._move = (self._q.copy(), t.copy(), float(now), max(float(duration), 1e-6))
        self._move_done.clear()
        self._move_completed = False
        self._goal = t.copy()
        self._goal_time = None
        self._goal_velocity = np.zeros_like(t)
        self._last_goal = None
        self._last_goal_time = None
        return self._move_done

    def brake(self) -> np.ndarray:
        """End a running move or tracked goal and come to rest within the limits.

        The joints decelerate together, so the motion keeps its direction while
        it stops.

        Returns:
            The setpoint the stream comes to rest at.
        """
        # Scaled together, every joint stops when the slowest-braking one does.
        rest = self._q + self._v * float(np.max(np.abs(self._v) / self._amax)) / 2.0
        self._finish_move(completed=False)
        self._goal = rest
        self._goal_time = None
        self._goal_velocity = np.zeros_like(rest)
        self._goal_interval = 0.0
        self._last_goal = None
        self._last_goal_time = None
        return rest.copy()

    def abort_move(self) -> None:
        """End a running move where it stands, as not completed."""
        if self._move is not None:
            self._v = np.zeros_like(self._q)
            self._goal = self._q.copy()
        self._finish_move(completed=False)

    def step(self, now: float, dt: float) -> np.ndarray:
        """Advance one control period of ``dt`` seconds and return the setpoint."""
        if self._move is not None:
            start, target, t0, duration = self._move
            s = min(max((now - t0) / duration, 0.0), 1.0)
            span = target - start
            self._q = start + 0.5 * (1.0 - np.cos(np.pi * s)) * span
            self._v = 0.5 * np.pi * np.sin(np.pi * s) / duration * span
            if s >= 1.0:
                self._q = target.copy()
                self._v = np.zeros_like(target)
                self._finish_move(completed=True)
            return self._q.copy()

        goal, feedforward = self._goal, self._goal_velocity
        if self._goal_time is not None and np.any(feedforward):
            # Half an interval of slack covers commands that arrive late.
            horizon = min(self._max_extrapolation_s, 1.5 * self._goal_interval)
            age = max(0.0, now - self._goal_time)
            if age < horizon:
                goal = goal + feedforward * age
            else:
                goal = goal + feedforward * horizon
                feedforward = np.zeros_like(feedforward)
        self._track(goal, feedforward, float(dt))
        return self._q.copy()

    def _track(self, goal: np.ndarray, feedforward: np.ndarray, dt: float) -> None:
        """Move one period toward ``goal`` within the velocity and acceleration limits."""
        error = goal - self._q
        dist = np.abs(error)
        step_a = self._amax * dt
        # Fastest speed from which each joint can still brake to rest within
        # its remaining error, one control period at a time.
        braking = step_a * (np.sqrt(0.25 + 2.0 * dist / (step_a * dt)) - 0.5)
        moving = dist > 1e-12
        # Feed-forward carries a streamed motion, so feedback only removes the
        # residual; a dead-beat gain would turn sub-milliradian mismatches
        # between consecutive goals into velocity spikes.
        gain = min(1.0 / dt, 1.0 / _FEEDBACK_TIME_S)
        if np.any(moving):
            gain = min(
                gain,
                float(
                    np.min(
                        np.minimum(braking[moving], self._vmax[moving]) / dist[moving]
                    )
                ),
            )
        desired = feedforward + gain * error
        over = float(np.max(np.abs(desired) / self._vmax))
        if over > 1.0:
            desired = desired / over
        change = desired - self._v
        over = float(np.max(np.abs(change) / step_a))
        if over > 1.0:
            change = change / over
        self._v = self._v + change
        self._q = self._q + self._v * dt

    def _finish_move(self, *, completed: bool) -> None:
        if self._move is None:
            return
        self._move = None
        self._move_completed = completed
        self._move_done.set()


class JointStream:
    """Stream a connection's joint setpoints from a fixed-rate background loop.

    A driver that commands joint positions over a streaming interface owns one
    ``JointStream`` and feeds it through callables, so the timing, limits, and
    fault handling live here rather than in each driver:

    - ``send(setpoint)`` publishes one frame, called from the loop thread.
    - ``measure()`` returns the measured joints; streaming starts from them
      whenever it was idle, so resuming after a hold never jumps.
    - ``faulted()`` reports a latched fault from memory; the loop stops
      sending while it is true.
    - ``limits()`` returns per-joint ``(max_velocity, max_acceleration)``.
    - ``profile()`` returns the connection's :class:`MotionProfile`, read each
      tick for the control rate and extrapolation horizon.
    - ``on_start()`` optionally prepares the controller (such as switching it
      to streaming mode) before streaming starts, outside the stream's lock.
    - ``stopping()`` optionally reports that the process is shutting down, so
      a send failing then is logged as shutdown rather than as an error.

    Commands return at once (:meth:`track`) or block until a timed move ends
    (:meth:`move`); :meth:`hold` stops sending until the next command.
    """

    def __init__(
        self,
        *,
        send: Callable[[np.ndarray], None],
        measure: Callable[[], np.ndarray],
        faulted: Callable[[], bool],
        limits: Callable[[], tuple[np.ndarray, np.ndarray]],
        profile: Callable[[], MotionProfile],
        on_start: Optional[Callable[[], None]] = None,
        stopping: Optional[Callable[[], bool]] = None,
        name: str = "joint-stream",
    ) -> None:
        self._send = send
        self._measure = measure
        self._faulted = faulted
        self._limits = limits
        self._profile = profile
        self._on_start = on_start
        self._stopping = stopping
        self._name = name
        self._lock = threading.Lock()
        self._streamer: Optional[SetpointStreamer] = None
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._logger = get_logger()

    @property
    def active(self) -> bool:
        """Whether the loop is sending setpoints."""
        return self._active

    def start(self) -> None:
        """Start the loop thread; it sends nothing until the first command."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop sending and join the loop thread. Safe to call more than once."""
        self.hold()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def track(self, goal: "Sequence[float] | np.ndarray") -> None:
        """Stream toward ``goal`` from now on, without blocking."""
        if not self._active and self._on_start is not None:
            self._on_start()
        with self._lock:
            self._ready_locked().set_goal(goal, time.monotonic())

    def move(
        self,
        target: "Sequence[float] | np.ndarray",
        *,
        duration: float = 0.0,
        max_velocity: "Optional[float | Sequence[float] | np.ndarray]" = None,
    ) -> bool:
        """Move to ``target`` along a raised cosine and block until the move ends.

        The move lasts at least ``duration`` seconds, longer when the joint
        limits or ``max_velocity`` (rad/s; the smallest value applies) require.

        Returns:
            ``True`` when the move arrived; ``False`` when a fault, a newer
            command, or a hold ended it first.
        """
        if not self._active and self._on_start is not None:
            self._on_start()
        speed, acceleration = self._limits()
        if max_velocity is not None:
            cap = float(np.min(np.abs(np.asarray(max_velocity, dtype=np.float64))))
            speed = np.minimum(speed, cap)
        with self._lock:
            streamer = self._ready_locked()
            target_arr = np.asarray(target, dtype=np.float64).reshape(streamer.dim)
            move_s = self._profile().move_duration(
                target_arr - streamer.setpoint,
                max_velocity=speed,
                max_acceleration=acceleration,
                min_duration=float(duration),
            )
            done = streamer.start_move(target_arr, time.monotonic(), move_s)
        # The slack covers loop ticks that run late.
        finished = done.wait(timeout=move_s + 1.0)
        with self._lock:
            if not finished:
                streamer.abort_move()
            return finished and streamer.move_completed

    def brake(self) -> Optional[np.ndarray]:
        """End any running move and bring the joints to rest within the limits.

        Unlike :meth:`hold`, the stream keeps sending, so the joints decelerate
        instead of stopping at whatever setpoint was sent last.

        Returns:
            Where the joints come to rest, or ``None`` when the stream is idle.
        """
        with self._lock:
            if not self._active or self._streamer is None:
                return None
            return self._streamer.brake()

    def hold(self) -> None:
        """Stop sending; the next command starts again from the measured joints."""
        with self._lock:
            self._active = False
            if self._streamer is not None:
                self._streamer.abort_move()

    def _ready_locked(self) -> SetpointStreamer:
        """Return the streamer, starting it at the measured joints when idle."""
        speed, acceleration = self._limits()
        extrapolation = self._profile().max_extrapolation_s
        if (
            not self._active
            or self._streamer is None
            or self._streamer.dim != np.asarray(speed).size
        ):
            if self._streamer is not None:
                self._streamer.abort_move()
            self._streamer = SetpointStreamer(
                self._measure(),
                max_velocity=speed,
                max_acceleration=acceleration,
                max_extrapolation_s=extrapolation,
            )
            self._active = True
        else:
            self._streamer.set_limits(
                max_velocity=speed,
                max_acceleration=acceleration,
                max_extrapolation_s=extrapolation,
            )
        return self._streamer

    def _run(self) -> None:
        last = next_tick = time.monotonic()
        while not self._stop.is_set():
            period = self._profile().dt
            now = time.monotonic()
            # A late tick advances by the time that passed, capped so a stall
            # does not become one large step.
            dt = min(max(now - last, 0.0), 2.0 * period)
            last = now
            setpoint: Optional[np.ndarray] = None
            if self._active:
                faulted = self._faulted()
                with self._lock:
                    if self._active and self._streamer is not None:
                        if faulted:
                            self._active = False
                            self._streamer.abort_move()
                        else:
                            setpoint = self._streamer.step(now, dt)
            if setpoint is not None:
                try:
                    self._send(setpoint)
                except Exception as exc:
                    self.hold()
                    if self._stopping is not None and self._stopping():
                        self._logger.info(f"{self._name} stopped: shutting down.")
                    else:
                        self._logger.error(
                            f"{self._name} stopped streaming after a send failed: "
                            f"{exc!r}"
                        )
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_tick = time.monotonic()
