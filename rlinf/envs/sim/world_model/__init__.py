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

"""World-model environments.

:class:`WorldModelEnv` is the public env workers construct. Each world model
registers a backend; construction selects it from ``cfg.backend``. Built-in
backends are imported only when that ``backend`` is used, because those modules
pull in optional generation stacks.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import WorldModelEnv
    from .registry import register_backend

__all__ = ["WorldModelEnv", "register_backend"]


def __getattr__(name: str) -> Any:
    """Load the public env class the first time ``WorldModelEnv`` is used."""
    if name == "WorldModelEnv":
        value = importlib.import_module(".env", __name__).WorldModelEnv
    elif name == "register_backend":
        value = importlib.import_module(".registry", __name__).register_backend
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
