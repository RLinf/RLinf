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
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints

from ..hardware import HardwareConfig


class RobotAutoConfig:
    """Fill unset robot config fields from same-named env vars.

    Each ``None`` field is read from the uppercased field name on the
    enumerating node (``robot_ip`` -> ``ROBOT_IP``), parsed by its type
    annotation: int/float/bool cast, lists comma-split. Since enumeration runs
    per node, each node resolves the hardware attached to it.

    When a node hosts several robots, one env var carries a comma-separated
    value per robot, assigned to the configs in order (so a list field gets one
    item per robot); with a single robot the whole value is used.

    When no configs are given for a node, configs are created from the env vars:
    the number of robots is the comma-separated count of an identifier env var
    (``count_fields``), and every field is taken from its env var.
    """

    #: Fields that are never auto-resolved (placement / structural fields).
    SKIP_FIELDS: frozenset[str] = frozenset({"node_rank"})

    @classmethod
    def resolve(
        cls,
        configs: list[HardwareConfig],
        config_cls: type[HardwareConfig] | None = None,
        node_rank: int | None = None,
        count_fields: Sequence[str] = (),
    ) -> list[HardwareConfig]:
        """Fill the configs' unset fields from env vars in place and return them.

        Fields already set in the YAML are left untouched. With multiple
        configs, each env var must hold one comma-separated value per config.

        When ``configs`` is empty, configs are created from the env vars (see
        the class docstring); ``config_cls``, ``node_rank`` and ``count_fields``
        must be given to enable this.
        """
        created = not configs
        if created:
            configs = cls._create_from_env(config_cls, node_rank, count_fields)
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

            # One robot uses the whole value; several robots split it by comma,
            # one value each, in order.
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
                # Created configs have no YAML to preserve, so env wins over the
                # dataclass defaults; otherwise only fill the unset fields.
                if created or getattr(config, name) is None:
                    setattr(config, name, cls._parse(chunk, hint))

        return configs

    @staticmethod
    def _create_from_env(
        config_cls: type[HardwareConfig] | None,
        node_rank: int | None,
        count_fields: Sequence[str],
    ) -> list[HardwareConfig]:
        """Create configs from env vars, sized by an identifier env var.

        Returns an empty list unless ``config_cls``/``node_rank`` are given and
        one of ``count_fields`` is set in the environment.
        """
        if config_cls is None or node_rank is None:
            return []

        n: int | None = None
        for name in count_fields:
            raw = os.environ.get(name.upper())
            if raw is None:
                continue
            count = len(raw.split(","))
            if n is None:
                n = count
            elif n != count:
                raise ValueError(
                    f"Identifier env vars {[f.upper() for f in count_fields]} "
                    f"disagree on the robot count ({n} vs {count}) on this node."
                )
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
        if get_origin(type_hint) is Union:
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
