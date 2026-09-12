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

from typing import Any, Callable, Iterable, Mapping

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


def task_on_robot(task: type, presets: Mapping[str, tuple[str, type]]) -> Callable:
    """Create an entry point that runs ``task`` on whichever robot a run has.

    Args:
        task: The task class the id names.
        presets: For each robot type, the robot-bound id and env class that
            run ``task`` on it.
    """
    known = sorted(env_id for env_id, _ in presets.values())

    def create(
        override_cfg: dict[str, Any],
        worker_info: Any,
        robot_info: Any,
        env_idx: int,
        env_cfg: Mapping[str, Any],
    ) -> gym.Env:
        if robot_info is None:
            raise ValueError(
                f"{task.__name__}-v1 runs on the robot a run is given, and this "
                f"run was given none. Supply robot_info, or name the robot "
                f"with one of {known}."
            )
        chosen = presets.get(robot_info.type)
        if chosen is None:
            raise ValueError(
                f"{task.__name__} has no preset for {robot_info.type!r}; it "
                f"runs on {sorted(presets)}."
            )
        return task_factory(chosen[1])(
            override_cfg, worker_info, robot_info, env_idx, env_cfg
        )

    create.__name__ = f"create_{task.__name__}"
    create.__qualname__ = create.__name__
    return create


def register_task_ids(
    module: str,
    namespace: dict[str, Any],
    tables: Iterable[Mapping[str, type]],
) -> list[str]:
    """Register one robot-free id per task, beside the robot-bound ones.

    A task registered on robots as ``PegInsertionEnv-v1`` and
    ``GimArmPegInsertionEnv-v1`` also gets ``PegInsertion-v1``, which picks
    the robot's preset from the ``robot_info`` a run is given. Only envs
    built on a task, as ``RegisteredTaskEnv`` subclasses are, take part.

    Args:
        module: Dotted path of the calling module, i.e. ``__name__``.
        namespace: The caller's ``globals()``, where entry points are bound.
        tables: Every robot package's ``TASKS`` mapping.

    Returns:
        The entry point names bound into ``namespace``, for ``__all__``.

    Raises:
        ValueError: If one robot registers one task under two ids without
            either of them opting out with ``GENERIC_ID = False``.
    """
    from .task_env import RegisteredTaskEnv

    by_task: dict[type, dict[str, tuple[str, type]]] = {}
    opted_out: set[type] = set()
    for table in tables:
        for env_id, env_cls in table.items():
            if not issubclass(env_cls, RegisteredTaskEnv):
                continue
            if not env_cls.GENERIC_ID:
                # The id says how the robot is driven, which a robot-free id
                # cannot pick for a run.
                opted_out.add(env_cls.TASK)
                continue
            presets = by_task.setdefault(env_cls.TASK, {})
            robot_type = env_cls.ROBOT.ROBOT_TYPE
            if robot_type in presets:
                raise ValueError(
                    f"{robot_type} registers {env_cls.TASK.__name__} as both "
                    f"{presets[robot_type][0]} and {env_id}. Set "
                    "'GENERIC_ID = False' on the ids that differ only in how "
                    "the robot is driven."
                )
            presets[robot_type] = (env_id, env_cls)
    names = []
    for task, presets in by_task.items():
        if task in opted_out:
            continue
        entry_point = task_on_robot(task, presets)
        namespace[entry_point.__name__] = entry_point
        register(
            id=f"{task.__name__}-v1", entry_point=f"{module}:{entry_point.__name__}"
        )
        names.append(entry_point.__name__)
    return sorted(names)
