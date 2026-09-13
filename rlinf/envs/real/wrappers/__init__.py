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

"""Build the wrapper stack declared by a real-world environment."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

from rlinf.envs.real.wrappers.episode import (
    KeyboardEvalControlWrapper,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    KeyboardRLTPolicySwitchWrapper,
    KeyboardStartEndWrapper,
)
from rlinf.envs.real.wrappers.teleop.builder import build_teleop
from rlinf.envs.real.wrappers.teleop.config import NO_DEVICE, resolve_teleop_devices
from rlinf.envs.real.wrappers.teleop.intervention import TeleopIntervention
from rlinf.envs.real.wrappers.transforms import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    RelativeFrame,
)

#: Wrappers available to ``ACTION_WRAPPERS`` and ``TRANSFORMS`` declarations.
WRAPPERS: dict[str, type] = {
    "GripperCloseEnv": GripperCloseEnv,
    "Quat2EulerWrapper": Quat2EulerWrapper,
    "RelativeFrame": RelativeFrame,
}

#: Keyboard modes, by the ``keyboard_reward_wrapper`` value that selects them.
KEYBOARD_MODES: dict[str, type] = {
    "multi_stage": KeyboardRewardDoneMultiStageWrapper,
    "single_stage": KeyboardRewardDoneWrapper,
    "start_end": KeyboardStartEndWrapper,
    "eval_control": KeyboardEvalControlWrapper,
    "rlt_policy_switch": KeyboardRLTPolicySwitchWrapper,
}


class WrapperStack:
    """Apply declared wrappers in the required execution order.

    Action wrappers run first, followed by teleoperation, episode controls, and
    observation or action transforms.
    """

    def __init__(self, env: gym.Env, cfg: Mapping[str, Any]) -> None:
        self.env = env
        self.cfg = cfg
        self.inner = env.unwrapped

    def build(self) -> gym.Env:
        """Build and return the configured wrapper stack."""
        self._refuse_unsupported()
        self._apply(getattr(self.inner, "ACTION_WRAPPERS", ()))
        self._apply_teleop()
        self._apply_keyboard_reward()
        self._apply_episode()
        self._apply(getattr(self.inner, "TRANSFORMS", ()))
        return self.env

    def _refuse_unsupported(self) -> None:
        """Reject enabled flags that the environment does not support."""
        defaults = getattr(self.inner, "REFUSE_DEFAULTS", {})
        for flag in getattr(self.inner, "REFUSE_FLAGS", ()):
            if self.cfg.get(flag, defaults.get(flag, False)):
                raise NotImplementedError(
                    f"{type(self.inner).__name__} does not support {flag!r}."
                )

    def _wanted(self, wrapper: type) -> bool:
        """Return whether the configuration enables a wrapper."""
        flag = getattr(wrapper, "CONFIG_FLAG", None)
        if flag is None:
            return True
        return bool(self.cfg.get(flag, getattr(wrapper, "CONFIG_DEFAULT", True)))

    def _apply(self, names: Any) -> None:
        """Apply enabled wrappers from a sequence of registered names."""
        for name in names:
            wrapper = WRAPPERS[name]
            applies_to = getattr(wrapper, "applies_to", lambda env: True)
            if self._wanted(wrapper) and applies_to(self.inner):
                self.env = wrapper(self.env)

    def _apply_teleop(self) -> None:
        """Apply the configured teleoperation devices when supported."""
        devices = resolve_teleop_devices(
            self.cfg,
            supported=getattr(self.inner, "TELEOP", ()),
            default=getattr(self.inner, "TELEOP_DEFAULT", NO_DEVICE),
        )
        if not devices or getattr(self.inner.config, "is_dummy", False):
            return
        self.env = TeleopIntervention(
            self.env,
            build_teleop(self.env, self.cfg, devices),
            mark_flag=bool(getattr(self.inner, "TELEOP_MARK_FLAG", False)),
        )

    def _apply_keyboard_reward(self) -> None:
        """Let an operator score the episode from the keyboard."""
        mode = self.cfg.get("keyboard_reward_wrapper", None)
        if mode and not getattr(self.inner.config, "is_dummy", False):
            self.env = KEYBOARD_MODES[mode](self.env)

    def _apply_episode(self) -> None:
        """Apply environment-specific episode-control wrappers."""
        for extra in getattr(self.inner, "episode_wrappers", lambda cfg: ())(self.cfg):
            self.env = extra(self.env)


def build_stack(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Apply the wrappers declared by the environment and configuration."""
    return WrapperStack(env, cfg).build()
