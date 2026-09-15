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

"""Real-world robot environments, tasks, and wrapper integration.

Reaching :class:`RealWorldEnv` imports every robot package, which is what
registers their Gymnasium tasks. That is deferred until it is asked for,
because those packages pull in optional robotics and vision dependencies.

Environment classes live in the package for the robot they drive --
``rlinf.envs.real.so101.SO101ReachEnv`` and its siblings -- and importing one
registers that robot's tasks on its own.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import RealWorldEnv

__all__ = ["RealWorldEnv", "load_tasks"]

#: Robot packages whose import registers their Gymnasium tasks.
_ROBOT_PACKAGES = (
    ".dosw1",
    ".franka",
    ".gim_arm",
    ".piper",
    ".so101",
    ".xsquare",
    ".task_env",
)

_loaded = False


def load_tasks() -> None:
    """Import every robot package and register its Gymnasium tasks.

    Idempotent, and safe to call from a process that reached a submodule
    directly rather than through this package.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in _ROBOT_PACKAGES:
        importlib.import_module(module, __name__)
    env_module = importlib.import_module(".env", __name__)
    env_module.RealWorldEnv.realworld_setup()


def __getattr__(name: str) -> Any:
    """Load the real-world envs the first time ``RealWorldEnv`` is used."""
    if name != "RealWorldEnv":
        raise AttributeError(name)
    load_tasks()
    value = importlib.import_module(".env", __name__).RealWorldEnv
    globals()[name] = value
    return value
