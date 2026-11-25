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

from rlinf.utils.logging import get_logger

from .wrappers import WrapperStack

ConfigValidator = Callable[[Mapping[str, Any], Mapping[str, Any]], list[Any] | None]


def task_factory(
    env_cls: type, *, validate_config: ConfigValidator | None = None
) -> Callable[..., gym.Env]:
    """Create a Gymnasium entry point with optional pre-connection validation."""

    def create(
        override_cfg: dict[str, Any],
        worker_info: Any,
        robot_info: Any,
        env_idx: int,
        env_cfg: Mapping[str, Any],
    ) -> gym.Env:
        devices = None
        if validate_config is not None:
            devices = validate_config(override_cfg, env_cfg)
        env = env_cls(
            override_cfg=override_cfg,
            worker_info=worker_info,
            robot_info=robot_info,
            env_idx=env_idx,
        )
        stack = WrapperStack(env, env_cfg)
        try:
            return stack.build(teleop_devices=devices)
        except BaseException:
            try:
                stack.env.close()
            except BaseException:
                get_logger().exception(
                    "Failed to close %s after wrapper construction failed",
                    env_cls.__name__,
                )
            raise

    create.__name__ = f"create_{env_cls.__name__}"
    create.__qualname__ = create.__name__
    return create


def register_tasks(
    module: str,
    namespace: dict[str, Any],
    tasks: Mapping[str, type],
    *,
    validate_config: ConfigValidator | None = None,
) -> list[str]:
    """Register real-world tasks and publish their generated entry points.

    Gymnasium resolves an entry point by importing ``module`` and reading the
    named attribute, so each generated factory is bound into ``namespace``
    before registration.

    Args:
        module: Dotted path of the calling package, i.e. ``__name__``.
        namespace: The caller's ``globals()``, where entry points are bound.
        tasks: Mapping from Gymnasium ID to environment class.
        validate_config: Optional check of the task overrides and outer config
            before constructing each environment in this registration. Return
            resolved teleop devices to use them when building the stack, or
            ``None`` to resolve them from the constructed environment.

    Returns:
        The entry point names bound into ``namespace``, for ``__all__``.
    """
    names = []
    for env_id, env_cls in tasks.items():
        entry_point = task_factory(env_cls, validate_config=validate_config)
        namespace[entry_point.__name__] = entry_point
        register(id=env_id, entry_point=f"{module}:{entry_point.__name__}")
        names.append(entry_point.__name__)
    return sorted(names)
