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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

FieldPath = tuple[str, ...]


def _get_path(value: Mapping[str, Any], path: FieldPath) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Canonical robotics field {'.'.join(path)!r} is missing.")
        current = current[key]
    return current


def _set_path(value: dict[str, Any], path: FieldPath, field_value: Any) -> None:
    if not path:
        raise ValueError("A robotics field path cannot be empty.")
    current = value
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = field_value


class LegacyObservationAdapter:
    """Map canonical robot observations to legacy ``state``/``frames`` keys."""

    def __init__(
        self,
        state_fields: Mapping[str, FieldPath],
        frame_fields: Mapping[str, FieldPath],
    ) -> None:
        self.state_fields = dict(state_fields)
        self.frame_fields = dict(frame_fields)

    def adapt(self, observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Return the policy-facing legacy observation dictionary."""
        return {
            "state": {
                name: _get_path(observation, path)
                for name, path in self.state_fields.items()
            },
            "frames": {
                name: _get_path(observation, path)
                for name, path in self.frame_fields.items()
            },
        }


@dataclass(frozen=True)
class VectorActionBinding:
    """Bind one slice of a legacy action vector to a canonical field path."""

    path: FieldPath
    start: int
    stop: int


class VectorActionAdapter:
    """Translate flat policy actions into canonical namespaced robot actions."""

    def __init__(
        self,
        action_dim: int,
        bindings: list[VectorActionBinding],
    ) -> None:
        self.action_dim = action_dim
        self.bindings = list(bindings)
        for binding in self.bindings:
            if not 0 <= binding.start < binding.stop <= action_dim:
                raise ValueError(
                    f"Invalid action slice [{binding.start}:{binding.stop}] "
                    f"for action_dim={action_dim}."
                )

    def adapt(self, action: np.ndarray) -> dict[str, Any]:
        """Return one canonical action from a flat vector."""
        action = np.asarray(action)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, got {action.shape}."
            )
        canonical: dict[str, Any] = {}
        for binding in self.bindings:
            _set_path(
                canonical,
                binding.path,
                action[binding.start : binding.stop].copy(),
            )
        return canonical
