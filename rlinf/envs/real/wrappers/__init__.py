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

"""Building the wrapper stack an env asks for.

There used to be one builder per arm count, each with an if-chain over device
names inside it. What varies between robots is not the procedure -- narrow the
action, hand it to an operator, let someone mark the episode, change the
representation -- but which pieces take part. So the env says which, and there
is one builder.
"""

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

#: Wrappers an env may name in ``ACTION_WRAPPERS`` or ``TRANSFORMS``. Named
#: rather than imported there so an env module does not depend on the wrappers
#: it happens to use.
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
    """The wrappers one env asks for, in the order they have to go on.

    Order matters and is the same for every robot: narrow the action first so
    the operator drives what the policy drives, then teleop, then whoever marks
    the episode, then the representation the policy expects.

    A class rather than a procedure because the three steps share the env being
    wrapped, the config asking for it, and the env's own declarations -- which
    were otherwise passed between them by hand.
    """

    def __init__(self, env: gym.Env, cfg: Mapping[str, Any]) -> None:
        self.env = env
        self.cfg = cfg
        self.inner = env.unwrapped

    def build(self) -> gym.Env:
        """Return the wrapped env, or refuse a flag this env cannot honour."""
        self._refuse_unsupported()
        self._apply(getattr(self.inner, "ACTION_WRAPPERS", ()))
        self._apply_teleop()
        self._apply_keyboard_reward()
        self._apply_episode()
        self._apply(getattr(self.inner, "TRANSFORMS", ()))
        return self.env

    def _refuse_unsupported(self) -> None:
        """Say so rather than wrapping an env in something it cannot support."""
        for flag in getattr(self.inner, "REFUSE_FLAGS", ()):
            if self.cfg.get(flag, False):
                raise NotImplementedError(
                    f"{type(self.inner).__name__} does not support {flag!r}."
                )

    def _wanted(self, wrapper: type) -> bool:
        """Whether the config switches this wrapper on.

        A wrapper with no flag is always applied; one with a flag says its own
        name and default, so this does not grow a branch per wrapper.
        """
        flag = getattr(wrapper, "CONFIG_FLAG", None)
        if flag is None:
            return True
        return bool(self.cfg.get(flag, getattr(wrapper, "CONFIG_DEFAULT", True)))

    def _apply(self, names: Any) -> None:
        """Put on each named wrapper the config asks for."""
        for name in names:
            wrapper = WRAPPERS[name]
            if self._wanted(wrapper):
                self.env = wrapper(self.env)

    def _apply_teleop(self) -> None:
        """Hand the action to an operator, if this env config asks for one.

        Which devices those are, and what each one needs, is settled by the
        config and the device registry. No device is named here.
        """
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
        """Whatever decides when this task's rollout starts and ends."""
        for extra in getattr(self.inner, "episode_wrappers", lambda cfg: ())(self.cfg):
            self.env = extra(self.env)


def build_stack(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrap ``env`` in what it declares and this config asks for."""
    return WrapperStack(env, cfg).build()
