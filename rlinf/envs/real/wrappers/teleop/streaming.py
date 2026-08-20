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

"""Commanding the robot faster than the environment steps.

A leader arm tracks well only if the follower receives targets continuously, and
``env.step`` runs at the policy's rate -- often 10 Hz against the ~1 kHz the
controller can accept. A streamer owns a thread that pushes targets straight to
the controller, while ``env.step`` keeps reading state and stops forwarding
motion, so the two never race for the same motion queue.

This is not an alternative to composition; it sits beside it. What the operator
is asking for still comes from a group of devices and their bindings. The thread
is the whole reason the class exists and the whole risk: it outlives a single
step, it must pause while the env drives the robot home, and it must be joined on
shutdown. :class:`TeleopStreamer` owns that lifecycle so a subclass only writes
what one tick sends.
"""

from __future__ import annotations

import threading
import time
from abc import abstractmethod
from typing import Any, Optional

import gymnasium as gym


class TeleopStreamer:
    """A command loop that runs beside the action, not instead of it.

    What the operator is asking for comes from composition, the same as any
    other rig. This delivers it a second way, straight to the controllers, for
    hardware that tracks badly at the policy's step rate. Subclasses implement
    :meth:`stream_once` -- one iteration of the loop -- and
    :meth:`ready_to_stream`, which decides whether the hardware is in a state
    where streaming is safe to begin.

    Args:
        period: Seconds between ticks. The loop targets this rate and skips the
            sleep when a tick already took longer.
        enabled: Whether to stream at all. When ``False`` nothing runs and the
            action reaches the robot through ``env.step`` like any other rig.
    """

    #: The action parts this loop delivers itself. While it streams, ``step``
    #: does not dispatch them, so the composed vector still carries them but
    #: the robot has already been told. Naming them is what keeps that visible
    #: rather than leaving the two paths to be inferred from a config flag.
    DELIVERS: tuple[str, ...] = ()

    def __init__(self, period: float = 0.001, enabled: bool = False) -> None:
        self._period = period
        self._enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._aligned = False
        # Open means a tick may run. It is closed while the env resets so the
        # env's own motion is the only thing commanding the robot.
        self._gate = threading.Event()
        self._gate.set()

    @property
    def streaming(self) -> bool:
        """Whether the command loop is currently running."""
        return self._thread is not None and self._thread.is_alive()

    @abstractmethod
    def stream_once(self, env: gym.Env) -> None:
        """Send one set of targets. Called at roughly :attr:`period`."""

    def ready_to_stream(self, env: gym.Env) -> bool:
        """Whether streaming can safely start."""
        return True

    def align(self, env: gym.Env) -> bool:
        """Bring the robot to the device's current pose before streaming.

        Without this the first tick would push a far target straight into the
        controller, which reads as a step change in the reference.
        """
        return True

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Pause the loop for the duration of the reset."""
        self._gate.clear()
        return kwargs

    def reset(self, env: gym.Env) -> None:
        """Align to the device while the robot is still at its reset pose."""
        self._aligned = False
        if self._enabled:
            self._aligned = self.align(env)

    def after_reset(self, env: gym.Env) -> None:
        """Let the loop run again, and start it once alignment has happened."""
        self._gate.set()
        self._maybe_start(env)

    def before_step(self, env: gym.Env) -> None:
        """Start the loop if it is due and not yet running."""
        self._maybe_start(env)

    def _maybe_start(self, env: gym.Env) -> None:
        if not self._enabled or not self._aligned or self._thread is not None:
            return
        if not self.ready_to_stream(env):
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(env,),
            name=f"{type(self).__name__}Stream",
            daemon=True,
        )
        self._thread.start()

    def _loop(self, env: gym.Env) -> None:
        while self._running:
            self._gate.wait()
            if not self._running:
                break
            started = time.monotonic()
            self.stream_once(env)
            remaining = self._period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    def close(self) -> None:
        """Stop the loop and wait for the thread to finish its tick."""
        self._running = False
        self._gate.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
