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

"""Gymnasium registration helpers for real-world tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from .wrappers import build_stack


def task_factory(env_cls: type) -> Callable[..., gym.Env]:
    """Create a Gymnasium entry point for a real-world environment class."""

    def create(
        override_cfg: dict[str, Any],
        worker_info: Any,
        robot_info: Any,
        env_idx: int,
        env_cfg: Mapping[str, Any],
    ) -> gym.Env:
        env = env_cls(
            override_cfg=override_cfg,
            worker_info=worker_info,
            robot_info=robot_info,
            env_idx=env_idx,
        )
        return build_stack(env, env_cfg)

    create.__name__ = f"create_{env_cls.__name__}"
    create.__qualname__ = create.__name__
    return create


def register_tasks(
    module: str,
    namespace: dict[str, Any],
    tasks: Mapping[str, type],
) -> list[str]:
    """Register real-world tasks and publish their generated entry points.

    Gymnasium resolves an entry point by importing ``module`` and reading the
    named attribute, so each generated factory is bound into ``namespace``
    before registration.

    Args:
        module: Dotted path of the calling package, i.e. ``__name__``.
        namespace: The caller's ``globals()``, where entry points are bound.
        tasks: Mapping from Gymnasium ID to environment class.

    Returns:
        The entry point names bound into ``namespace``, for ``__all__``.
    """
    names = []
    for env_id, env_cls in tasks.items():
        entry_point = task_factory(env_cls)
        namespace[entry_point.__name__] = entry_point
        register(id=env_id, entry_point=f"{module}:{entry_point.__name__}")
        names.append(entry_point.__name__)
    return sorted(names)
