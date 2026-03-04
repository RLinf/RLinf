# Copyright 2025 The RLinf Authors.
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

"""Genesis task registry.

Tasks define the scene layout (robot, objects, cameras), reset logic,
reward computation, and observation extraction for a specific manipulation
or locomotion scenario running inside the Genesis simulator.

To register a new task, add it to ``_TASK_REGISTRY`` below and implement
the :class:`GenesisTaskBase` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlinf.envs.genesis.tasks.base import GenesisTaskBase


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type[GenesisTaskBase]] = {}


def register_task(name: str, cls: type[GenesisTaskBase]) -> None:
    """Register a Genesis task class under the given name."""
    _TASK_REGISTRY[name.lower()] = cls


def get_task_cls(name: str) -> type[GenesisTaskBase]:
    """Look up a registered Genesis task class by name.

    Args:
        name: Case-insensitive task name (e.g. ``"cube_pick"``).

    Raises:
        KeyError: If no task with the given name has been registered.
    """
    key = name.lower()
    if key not in _TASK_REGISTRY:
        raise KeyError(
            f"Genesis task '{name}' not registered. "
            f"Available tasks: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[key]


# ---------------------------------------------------------------------------
# Auto-import built-in tasks so they self-register on first access.
# ---------------------------------------------------------------------------

def _import_builtin_tasks() -> None:
    """Import all built-in task modules so their ``register_task`` calls run."""
    # Each task module calls ``register_task(...)`` at import time.
    from rlinf.envs.genesis.tasks import cube_pick as _cube_pick  # noqa: F401


_import_builtin_tasks()
