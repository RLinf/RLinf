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

"""Letting an operator take over from the policy.

Every teleop device answers the same question -- *what would the operator do
right now* -- and the answer is handled the same way whatever produced it: while
the operator is driving, their action replaces the policy's, and the episode
records that it was overridden. Only the reading differs.

So the reading is the only thing a device implements. :class:`TeleopDevice`
turns hardware into a :class:`TeleopSample`, and :class:`TeleopIntervention`
owns the part that used to be copied into every wrapper: the hold window that
keeps the operator in control between samples, and the ``info`` keys a dataset
collector reads back. Holding state the policy does not command is a matter for
the composed group, which starts from the policy's action and overwrites only
what its devices fill.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np


@dataclass
class TeleopSample:
    """One reading from a teleop device.

    Attributes:
        action: What the operator is asking for, in the env's action space, or
            ``None`` when the device has nothing to say -- not connected yet, or
            still waiting for its first packet. A ``None`` sample never starts
            the hold window.
        active: Whether the operator is actually driving. Devices report small
            residual motion constantly, so each one decides its own threshold;
            this flag, not the action, is what restarts the hold window.
        info: Extra keys to merge into the step ``info``, for device state a
            collector wants to record.
    """

    action: Optional[np.ndarray]
    active: bool
    info: dict[str, Any] = field(default_factory=dict)


class TeleopDevice(ABC):
    """Hardware an operator drives, expressed as actions for one env.

    Implement :meth:`read`. The rest has defaults that suit most devices.
    """

    #: How long the operator keeps control after their last active sample.
    #: Devices sample faster than a person moves, so without this the action
    #: would flicker between operator and policy inside a single motion.
    #:
    #: Set it to ``0`` for a device the operator holds down to take over -- a
    #: trigger or a grip button says exactly when they are driving, and
    #: stretching that by half a second would keep commanding the robot after
    #: they let go.
    timeout: float = 0.5

    @abstractmethod
    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Return what the operator is asking for, in ``env``'s action space."""

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adjust the reset arguments, and quiet the device while it runs.

        A device that is already commanding the robot has to stop before the env
        drives it home, or the two fight over the same controller.
        """
        return kwargs

    def reset(self, env: gym.Env) -> None:
        """Re-sync with the robot after an episode reset."""

    def after_reset(self, env: gym.Env) -> None:
        """Run once the reset is over, whether or not it succeeded."""

    def before_step(self, env: gym.Env) -> None:
        """Hook that runs before the wrapped env steps."""

    def on_action_chunk_begin(self) -> None:
        """Let go of anything held only until the next chunk of actions."""

    def get_hold_action(
        self, env: gym.Env, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The action that keeps the robot where it is, for a skipped chunk.

        Raises:
            AttributeError: If this device commands deltas, where a zero motion
                is already the action that holds a robot still.
        """
        raise AttributeError(
            f"{type(self).__name__} commands deltas, so it has no pose to hold."
        )

    def close(self) -> None:
        """Release the device."""


class TeleopIntervention(gym.Wrapper):
    """Replace the policy's action while the operator is driving.

    This is a :class:`gymnasium.Wrapper` rather than an ``ActionWrapper``
    deliberately: deciding whether an action was overridden also decides what
    goes into ``info``, and ``ActionWrapper.action`` has nowhere to report that.

    Args:
        env: The environment to wrap.
        device: The teleop device to read.
        mark_flag: Also write ``info["intervene_flag"]`` when overriding. Some
            dataset formats key on the flag rather than on the action.
    """

    def __init__(
        self,
        env: gym.Env,
        device: TeleopDevice,
        mark_flag: bool = False,
    ) -> None:
        super().__init__(env)
        self.device = device
        self.mark_flag = mark_flag
        self._last_active: float = -float("inf")

    @property
    def intervening(self) -> bool:
        """Whether the operator currently holds control."""
        return time.monotonic() - self._last_active < self.device.timeout

    def reset(self, **kwargs: Any):
        """Reset the env, then let the device re-sync with it.

        ``after_reset`` runs even when the reset fails, because a device that
        paused itself in ``before_reset`` would otherwise stay paused for the
        rest of the session.
        """
        kwargs = self.device.before_reset(self, kwargs)
        try:
            result = self.env.reset(**kwargs)
            self._last_active = -float("inf")
            self.device.reset(self)
            return result
        finally:
            self.device.after_reset(self)

    def step(self, action: np.ndarray):
        """Step with the operator's action when they are driving."""
        self.device.before_step(self)
        sample = self.device.read(self, action)

        if sample.action is None:
            applied, overridden = action, False
        elif sample.active:
            self._last_active = time.monotonic()
            applied, overridden = sample.action, True
        elif self.intervening:
            # Quiet sample, but still inside the hold window.
            applied, overridden = sample.action, True
        else:
            applied, overridden = action, False

        obs, reward, terminated, truncated, info = self.env.step(applied)

        if overridden:
            info["intervene_action"] = applied
            if self.mark_flag:
                info["intervene_flag"] = np.ones(1)
        info.update(sample.info)

        return obs, reward, terminated, truncated, info

    def on_action_chunk_begin(self) -> None:
        """Tell the device a fresh chunk of policy actions starts here.

        On the wrapper because :mod:`rlinf.envs.real.env` finds it by name
        through ``get_wrapper_attr``: the env runs the chunk, and the device
        holds what it means.
        """
        self.device.on_action_chunk_begin()

    def get_hold_action(
        self, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The action that keeps the robot where it is, for a skipped chunk."""
        return self.device.get_hold_action(self, fallback_action)

    def close(self):
        """Release the device, then the wrapped env."""
        self.device.close()
        return super().close()
