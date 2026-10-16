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

"""Registration for world-model backends.

Workers construct :class:`~rlinf.envs.sim.world_model.env.WorldModelEnv` when
``env_type`` is ``world_model``. That class looks up the backend registered for
``cfg.backend`` (for example ``wan``). Built-in backends are imported only
when that name is looked up, so Wan and OpenSora keep separate optional
dependencies.

To add a world model, implement a :class:`~rlinf.envs.sim.world_model.backend.WorldModelBackend`,
call :func:`register_backend`, and either add the module to
``_BUILTIN_MODULES`` or import it before the env is constructed. YAML
``backend`` must match the registered name.
"""

from __future__ import annotations

import importlib
from typing import Optional, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, type] = {}

#: Canonical ``backend`` name to the module that calls :func:`register_backend`
#: on import. Looked up one name at a time so unused backends stay unloaded.
_BUILTIN_MODULES: dict[str, str] = {
    "wan": "rlinf.envs.sim.world_model.backend.wan",
    "opensora": "rlinf.envs.sim.world_model.backend.opensora",
    "remote": "rlinf.envs.sim.world_model.backend.remote",
}

#: Retired ``env_type`` spellings mapped to the backend they now select.
_BACKEND_ALIASES: dict[str, str] = {
    "wan_wm": "wan",
    "opensora_wm": "opensora",
}


def _normalize_backend(name: object) -> str:
    value = getattr(name, "value", name)
    key = str(value).lower()
    return _BACKEND_ALIASES.get(key, key)


def get_backend_name(name: object) -> Optional[str]:
    """Return the canonical backend name if ``name`` is known, else ``None``.

    Accepts a registered name, a built-in name, or a retired alias. Does not
    import the backend module, so unused generation stacks stay unloaded.
    """
    key = _normalize_backend(name)
    if key in _REGISTRY or key in _BUILTIN_MODULES:
        return key
    return None


def register_backend(name: str):
    """Register a world-model backend under a ``backend`` name."""

    def decorator(cls: T) -> T:
        key = name.lower()
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"world model backend {name!r} is already registered as {existing.__name__}"
            )
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_backend(name: object) -> type:
    """Return the backend class registered for ``name``, loading a built-in if needed.

    Args:
        name: Case-insensitive ``backend`` (for example ``"wan"``).

    Raises:
        KeyError: If no backend is registered for ``name`` and it is not a
            built-in module.
    """
    key = _normalize_backend(name)
    if key not in _REGISTRY:
        module_name = _BUILTIN_MODULES.get(key)
        if module_name is not None:
            importlib.import_module(module_name)
    if key not in _REGISTRY:
        available = sorted(set(_REGISTRY) | set(_BUILTIN_MODULES))
        raise KeyError(
            f"world model backend {name!r} is not registered. Available: {available}"
        )
    return _REGISTRY[key]
