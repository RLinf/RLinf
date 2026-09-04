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

"""Robot-part views backed by methods on a shared connection.

These adapters expose arms, end effectors, and cameras from vendor sessions
that provide method-based APIs. Each view borrows the host connection's
lifecycle and presents the standard part interface.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union, cast

import numpy as np

from .arms.base import Arm
from .base import Action, Connection, Features, Observation
from .cameras.base import Camera
from .end_effectors.base import EndEffector


def host_state(host: Any) -> dict[str, Any]:
    """Read a host's state, sharing the snapshot when it keeps one.

    A view's host is normally a :class:`Connection`, which serves one read to
    a part and its riders. Hosts are duck-typed though, so a plain object with
    ``get_state`` is still allowed.
    """
    read = getattr(host, "read_state", None)
    return state_to_dict(read() if callable(read) else host.get_state())


def state_to_dict(state: Any) -> dict[str, Any]:
    """Normalize a host part's state object into a plain dictionary."""
    if isinstance(state, dict):
        return state
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    if is_dataclass(state) and not isinstance(state, type):
        return cast(dict[str, Any], asdict(state))
    raise TypeError(f"Host state {type(state).__name__} is not dictionary-like.")


class MethodArm(Arm):
    """Expose one arm through methods on a shared host connection.

    Args:
        host: The part owning the connection.
        commands: Map from canonical action field to host method name, e.g.
            ``{"tcp_pose": "move_left_arm"}``.
        state_fields: Canonical observation names, either a tuple selecting
            host state fields verbatim or a map from canonical name to the
            host's own field name.
    """

    def __init__(
        self,
        host: "Connection",
        commands: dict[str, str],
        state_fields: Optional[Union[tuple[str, ...], dict[str, str]]] = None,
    ) -> None:
        self._host = self._owner = host
        self.commands = dict(commands)
        self.state_fields = (
            dict(state_fields)
            if isinstance(state_fields, dict)
            else {name: name for name in state_fields or ()}
        )

    @property
    def observation_features(self) -> Features:
        """Describe the state fields this view exposes."""
        return {name: {} for name in self.state_fields}

    @property
    def action_features(self) -> Features:
        """Describe the canonical command names this view accepts."""
        return {name: {} for name in self.commands}

    def reset(self) -> None:
        """Leave task-specific reset motion to the task environment."""

    def get_observation(self) -> Observation:
        """Select this view's fields out of the shared host state."""
        state = host_state(self._host)
        if not self.state_fields:
            return state
        return {name: state[source] for name, source in self.state_fields.items()}

    def send_action(self, action: Action) -> Observation:
        """Dispatch canonical command fields to the host's methods."""
        unknown = set(action) - set(self.commands)
        if unknown:
            raise KeyError(f"Unknown arm actions: {sorted(unknown)}")
        applied: dict[str, Any] = {}
        for name, value in action.items():
            getattr(self._host, self.commands[name])(value)
            applied[name] = value
        return applied


class MethodEndEffector(EndEffector):
    """Expose an end effector through methods on a shared host connection.

    Args:
        host: The part owning the connection.
        state_field: Host state field holding the end-effector value.
        dims: Width of the state and target vectors.
        command: Host method taking a continuous target. When ``None`` the
            view falls back to binary open/close on the sign of ``target[0]``.
        open_method: Host method that opens, in binary mode.
        close_method: Host method that closes, in binary mode.
        state_index: Optional index or slice selecting the end-effector value
            out of a wider state field.
    """

    def __init__(
        self,
        host: "Connection",
        state_field: str,
        dims: int = 1,
        command: Optional[str] = None,
        open_method: str = "open_gripper",
        close_method: str = "close_gripper",
        state_index: Optional[Union[int, slice]] = None,
    ) -> None:
        self._host = self._owner = host
        self.state_field = state_field
        self.dims = dims
        self.method = command
        self.open_method = open_method
        self.close_method = close_method
        self.state_index = state_index

    @property
    def action_dim(self) -> int:
        """Return the target vector width."""
        return self.dims

    @property
    def state_dim(self) -> int:
        """Return the state vector width."""
        return self.dims

    @property
    def control_mode(self) -> str:
        """Return continuous or binary control according to the host API."""
        return "continuous" if self.method is not None else "binary"

    def reset(self) -> None:
        """Leave task-specific end-effector reset to the task environment."""

    def get_state(self) -> np.ndarray:
        """Read the end-effector field out of the shared host state."""
        state = host_state(self._host)
        value = np.asarray(state[self.state_field])
        if self.state_index is not None:
            value = value[self.state_index]
        return np.asarray(value).reshape(-1)

    def command(self, action: np.ndarray) -> bool:
        """Apply a continuous target, or open and close on its sign."""
        target = np.asarray(action).reshape(-1)
        if target.shape != (self.dims,):
            raise ValueError(
                f"Expected end-effector target shape {(self.dims,)}, "
                f"got {target.shape}."
            )
        if self.method is not None:
            getattr(self._host, self.method)(target)
            return True
        opening = bool(target[0] >= 0)
        getattr(self._host, self.open_method if opening else self.close_method)()
        return True


class MethodCamera(Camera):
    """Expose camera frames returned by a method on a shared host.

    Args:
        host: The part owning the connection.
        method: Host method returning a frame.
        method_args: Fixed arguments identifying the camera, e.g. its id.
        check_method: Host method taking a timeout and returning one health
            flag per camera, indexed by the first of ``method_args``.
    """

    def __init__(
        self,
        host: "Connection",
        method: str,
        *method_args: Any,
        check_method: Optional[str] = None,
    ) -> None:
        self._host = self._owner = host
        self.method = method
        self.method_args = method_args
        self.check_method = check_method

    @property
    def observation_features(self) -> Features:
        """Describe the raw frame returned by the host."""
        return {"frame": {}}

    def reset(self) -> None:
        """Camera views have no resettable state."""

    def is_ready(self, timeout: float = 0.5) -> bool:
        """Ask the host whether this camera is delivering frames.

        Uses the host's ``check_method`` when it was given one, and otherwise
        falls back to fetching a frame, which is the only check a host that
        exposes no health method can offer.
        """
        if self.check_method is not None:
            checks = getattr(self._host, self.check_method)(timeout)
            return bool(
                checks[self.method_args[0]] if self.method_args else all(checks)
            )
        try:
            self.get_observation()
        except Exception:
            return False
        return True

    def get_observation(self) -> Observation:
        """Fetch one frame through the configured host method."""
        return {"frame": getattr(self._host, self.method)(*self.method_args)}
