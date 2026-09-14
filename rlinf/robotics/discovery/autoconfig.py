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

from __future__ import annotations

import dataclasses
import os
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from .registry import RobotConfig

RobotConfigType = TypeVar("RobotConfigType", bound=RobotConfig)


class RobotAutoConfig:
    """Resolve unset robot configuration fields from environment variables.

    Environment variable names are the uppercase field names. Values for
    multiple robots are comma-separated and assigned in configuration order.
    If no configurations are provided, scalar fields determine how many
    configurations to create.
    """

    #: Fields that are never auto-resolved (placement / structural fields).
    SKIP_FIELDS: frozenset[str] = frozenset({"node_rank"})

    @classmethod
    def resolve(
        cls,
        configs: list[RobotConfigType],
        config_cls: type[RobotConfigType] | None = None,
        node_rank: int | None = None,
    ) -> list[RobotConfigType]:
        """Resolve unset fields in place and return the configurations.

        Existing values take precedence. If ``configs`` is empty,
        ``config_cls`` and ``node_rank`` are required to create configurations.
        """
        created = not configs
        if created:
            configs = cls._create_from_env(config_cls, node_rank)
        if not configs:
            return configs

        type_hints = get_type_hints(type(configs[0]))
        n = len(configs)

        for field in dataclasses.fields(configs[0]):
            name = field.name
            if name in cls.SKIP_FIELDS or name.endswith("_node_rank"):
                continue

            raw = os.environ.get(name.upper())
            if raw is None:
                continue

            # Split values only when the node declares multiple robots.
            if n == 1:
                chunks = [raw]
            else:
                chunks = [chunk.strip() for chunk in raw.split(",")]
                if len(chunks) != n:
                    raise ValueError(
                        f"Environment variable {name.upper()} must provide {n} "
                        f"comma-separated values (one per robot config on this "
                        f"node), but got {len(chunks)}: {raw!r}."
                    )

            hint = type_hints.get(name)
            for config, chunk in zip(configs, chunks):
                # Environment values override defaults on generated configs.
                if created or getattr(config, name) is None:
                    setattr(config, name, cls._parse(chunk, hint))

        return configs

    @classmethod
    def _create_from_env(
        cls,
        config_cls: type[RobotConfigType] | None,
        node_rank: int | None,
    ) -> list[RobotConfigType]:
        """Create configurations using the largest scalar value count.

        List-valued fields describe one robot and therefore do not determine
        the number of configurations.
        """
        if config_cls is None or node_rank is None:
            return []

        type_hints = get_type_hints(config_cls)
        n = 0
        for field in dataclasses.fields(config_cls):
            name = field.name
            if name in cls.SKIP_FIELDS or name.endswith("_node_rank"):
                continue
            base = cls._unwrap_optional(type_hints.get(name))
            if get_origin(base) in (list, set, tuple):
                continue
            raw = os.environ.get(name.upper())
            if raw is None:
                continue
            # Defer count-mismatch reporting to resolve().
            n = max(n, len(raw.split(",")))
        if not n:
            return []
        return [config_cls(node_rank=node_rank) for _ in range(n)]

    @classmethod
    def _parse(cls, raw: str, type_hint: Any) -> Any:
        """Parse ``raw`` per ``type_hint`` (``str`` when unknown)."""
        raw = raw.strip()
        base = cls._unwrap_optional(type_hint)
        if get_origin(base) in (list, set, tuple):
            item_args = get_args(base)
            item_type = item_args[0] if item_args else str
            items = [cls._cast(part.strip(), item_type) for part in raw.split(",")]
            items = [item for item in items if item != ""]
            return list(items)
        return cls._cast(raw, base)

    @staticmethod
    def _unwrap_optional(type_hint: Any) -> Any:
        """Strip ``Optional[...]`` down to the inner type."""
        if get_origin(type_hint) in (Union, UnionType):
            non_none = [a for a in get_args(type_hint) if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return type_hint

    @staticmethod
    def _cast(raw: str, target_type: Any) -> Any:
        """Cast a single string to ``target_type`` (best effort)."""
        if target_type is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if target_type is int:
            return int(raw)
        if target_type is float:
            return float(raw)
        return raw
