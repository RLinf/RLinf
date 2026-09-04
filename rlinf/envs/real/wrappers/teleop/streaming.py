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

"""High-rate teleoperation command streaming outside the environment step loop."""

from __future__ import annotations

import threading
import time
from abc import abstractmethod
from typing import Any, Optional

import gymnasium as gym


class TeleopStreamer:
    """Run a high-rate command loop alongside normal environment steps.

    Subclasses implement :meth:`stream_once` and may gate startup with
    :meth:`ready_to_stream`.

    Args:
        period: Target interval between command ticks, in seconds.
        enabled: Whether to start the command loop.
    """

    #: Action parts delivered directly by the streaming loop.
    DELIVERS: tuple[str, ...] = ()

    def __init__(self, period: float = 0.001, enabled: bool = False) -> None:
        self._period = period
        self._enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._aligned = False
        # Pause streaming while the environment resets the robot.
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
        """Align the robot with the device before streaming starts."""
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
