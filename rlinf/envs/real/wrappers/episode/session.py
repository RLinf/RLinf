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

"""Shared keyboard and foot-pedal session handling for episode wrappers."""

from __future__ import annotations

import math
import time
from typing import Any, Iterator, Optional

import gymnasium as gym

from .keyboard import KeyboardListener


class KeyboardSession(gym.Wrapper):
    """Base wrapper for debounced operator key input.

    Subclasses read :meth:`presses` for debounced keys, or ``self.listener``
    directly when they want the raw stream.
    """

    #: Minimum interval between accepted presses of the same key.
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
        """Discard key presses queued by the previous episode."""
        self._last_press.clear()
        self.listener.pop_pressed_keys()

    def begin_episode(self) -> None:
        """Reset subclass state for a new episode."""

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """Clear session state before resetting the environment."""
        self.drain()
        self.begin_episode()
        return self.env.reset(seed=seed, options=options)

    def base_env(self) -> gym.Env:
        """Return the unwrapped environment."""
        return getattr(self.env, "unwrapped", self.env)

    def log(self, message: str, *args: Any) -> None:
        """Write an informational message through the environment logger."""
        logger = getattr(self.base_env(), "_logger", None)
        if logger is not None:
            logger.info(message, *args)
