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

"""Arbitrate between policy actions and operator teleoperation input."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np


@dataclass
class TeleopSample:
    """Action sample produced by a teleoperation device.

    Attributes:
        action: Operator command in the environment action space, or ``None``
            when no usable reading is available.
        active: Whether the operator currently holds control.
        apply_when_inactive: Whether ``action`` contains passive state, such as
            a held dexterous-hand pose, that must still reach the environment.
        info: Device state to merge into the step information.
    """

    action: Optional[np.ndarray]
    active: bool
    apply_when_inactive: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class TeleopDevice(ABC):
    """Base interface for operator input expressed as environment actions."""

    #: Duration for which control remains with the operator after an active sample.
    #: Use zero for devices that report an explicit held state.
    timeout: float = 0.5

    @abstractmethod
    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Return what the operator is asking for, in ``env``'s action space."""

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare the device and reset arguments before environment reset."""
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
        """Return an action that holds the robot during a skipped chunk.

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

    A regular :class:`gymnasium.Wrapper` is required because intervention
    metadata is written to ``info`` alongside the selected action.

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

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and synchronize the device afterward."""
        kwargs = self.device.before_reset(self, kwargs)
        try:
            result = self.env.reset(**kwargs)
            self._last_active = -float("inf")
            self.device.reset(self)
            return result
        finally:
            self.device.after_reset(self)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step with the operator's action when they are driving."""
        self.device.before_step(self)
        sample = self.device.read(self, action)

        if sample.action is None:
            applied, overridden = action, False
        elif sample.active:
            self._last_active = time.monotonic()
            applied, overridden = sample.action, True
        elif self.intervening:
            # Retain operator control for the configured hold window.
            applied, overridden = sample.action, True
        elif sample.apply_when_inactive:
            # Some devices keep only their own stateful action parts applied.
            applied, overridden = sample.action, False
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
        """Notify the device that a new policy action chunk has started."""
        self.device.on_action_chunk_begin()

    def get_hold_action(
        self, fallback_action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return an action that holds the robot during a skipped chunk."""
        return self.device.get_hold_action(self, fallback_action)

    def close(self) -> None:
        """Release the device, then the wrapped env."""
        self.device.close()
        return super().close()
