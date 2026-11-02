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

"""Expose a symmetric [-1, 1] action space over an env's native bounds."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class NormalizeAction(gym.ActionWrapper):
    """Map a policy's ``[-1, 1]`` action onto the wrapped env's own bounds.

    A tanh-squashed policy (e.g. ``MLPPolicy`` with ``add_q_head``) always
    outputs actions in ``(-1, 1)``, regardless of the wrapped env's native
    action range. Without this wrapper, that output reaches the env unscaled
    and clipped to whatever of its range falls inside ``[-1, 1]``, so the
    policy can never command the rest of it. This wrapper exposes the
    normalized range to the policy and rescales to the inner env's bounds
    before stepping it, so replay-buffer actions stay normalized while the
    inner env keeps its own native action contract.
    """

    #: Environment configuration flag that enables this wrapper.
    CONFIG_FLAG = "normalize_actions"
    CONFIG_DEFAULT = False

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        inner = self.env.action_space
        if not isinstance(inner, Box) or not (
            np.all(np.isfinite(inner.low)) and np.all(np.isfinite(inner.high))
        ):
            raise ValueError(
                f"NormalizeAction needs a finite Box action space, got {inner!r}."
            )
        self._low = inner.low.astype(np.float64)
        self._high = inner.high.astype(np.float64)
        self.action_space = Box(-1.0, 1.0, shape=inner.shape, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Rescale a ``[-1, 1]`` action to the inner env's native bounds."""
        action = np.asarray(action, dtype=np.float64)
        # Checked before clipping/broadcasting, which would otherwise turn a
        # malformed action into a plausible one instead of raising: clipping
        # alone maps +/-inf to a finite bound, and a scalar or one-element
        # action broadcasts against (_low, _high) into a full-width one.
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected an action of shape {self.action_space.shape}, got "
                f"{action.shape}."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Expected a finite action, got {action.tolist()}.")
        action = np.clip(action, -1.0, 1.0)
        return self._low + (action + 1.0) * 0.5 * (self._high - self._low)
