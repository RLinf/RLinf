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

"""Driving an episode from a keyboard or a foot pedal.

Marking a success, aborting a take, advancing a stage, switching policies: the
operator is the only one who knows, so each of these wrappers listens for a
keypress and turns it into something the rollout can see.

They differ in what a key means and agree on everything around it -- who owns the
listener, how a press is debounced, and that presses queued between episodes must
be dropped rather than delivered to the next one. That agreement lives here, so a
new mode is a ``step`` that reads :meth:`KeyboardSession.presses`.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterator, Optional

import gymnasium as gym

from rlinf.robotics.parts.teleop.readers.keyboard import KeyboardListener


class KeyboardSession(gym.Wrapper):
    """A wrapper the operator steers with keys, holding the listener.

    Subclasses read :meth:`presses` for debounced keys, or ``self.listener``
    directly when they want the raw stream.
    """

    #: Presses of the same key closer together than this are dropped. Foot
    #: pedals bounce, and a USB key-down burst arrives as several presses.
    DEBOUNCE_S: float = 0.2

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.listener = KeyboardListener()
        self._last_press: dict[str, float] = {}

    def presses(self) -> Iterator[str]:
        """Yield each key pressed since the last call, debounced."""
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press.get(key, -math.inf) < self.DEBOUNCE_S:
                continue
            self._last_press[key] = now
            yield key

    def drain(self) -> None:
        """Discard queued presses so they do not leak into the next episode."""
        self._last_press.clear()
        self.listener.pop_pressed_keys()

    def begin_episode(self) -> None:
        """Clear whatever the previous episode left behind."""

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Drop stale presses, reset the mode's state, then reset the env."""
        self.drain()
        self.begin_episode()
        return self.env.reset(seed=seed, options=options)

    def base_env(self) -> Any:
        """The innermost env, which owns the logger and the task config."""
        return getattr(self.env, "unwrapped", self.env)

    def log(self, message: str, *args: Any) -> None:
        """Report through the env's logger; these wrappers never print."""
        logger = getattr(self.base_env(), "_logger", None)
        if logger is not None:
            logger.info(message, *args)
