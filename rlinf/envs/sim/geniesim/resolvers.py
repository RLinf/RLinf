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

"""Resolve fields of a GenieSim ``override_cfg``.

Covers the endpoint and client arguments, task names, render width,
randomization, per-slot overrides and the work-item shard.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf

from rlinf.utils.nested_dict_process import update_nested_cfg


def _override_dict(cfg: Any) -> dict[str, Any]:
    if "override_cfg" not in cfg or cfg.override_cfg is None:
        return {}
    return dict(OmegaConf.to_container(cfg.override_cfg, resolve=True) or {})


def _client_kwargs(
    override: dict[str, Any], *, address: str | None = None
) -> tuple[str, dict[str, Any]]:
    backend = str(override.get("comm_backend", "zmq"))
    if backend != "zmq":
        raise ValueError("GenieSim task_metadata/init requires comm_backend=zmq")
    timeout_s = float(override.get("client_timeout_s", 60.0))
    if timeout_s <= 0:
        raise ValueError(f"GenieSim client_timeout_s must be positive, got {timeout_s}")
    return backend, {
        "address": str(address or override.get("address", "tcp://127.0.0.1:5601")),
        "timeout_s": timeout_s,
    }


def resolve_geniesim_address(
    override: dict[str, Any], *, rank: int, world_size: int
) -> str:
    raw_addresses = override.get("addresses")
    if raw_addresses is None:
        if world_size > 1:
            raise ValueError(
                "GenieSim multi-rank execution requires override_cfg.addresses"
            )
        return str(override.get("address", "tcp://127.0.0.1:5601")).strip()

    if isinstance(raw_addresses, str | bytes) or not isinstance(
        raw_addresses, Sequence
    ):
        raise TypeError("GenieSim override_cfg.addresses must be a sequence")
    addresses = [str(value).strip() for value in raw_addresses]
    if len(addresses) != world_size:
        raise ValueError(
            "GenieSim override_cfg.addresses must contain exactly one endpoint "
            f"per env rank: expected {world_size}, got {len(addresses)}"
        )
    if len(set(addresses)) != len(addresses):
        raise ValueError(
            "GenieSim override_cfg.addresses must contain unique endpoints"
        )
    return addresses[rank]


def resolve_geniesim_task_names(override: dict[str, Any]) -> list[str]:
    """Resolve allowed backgrounds in preference order.

    Each sub-task selects the first compatible background from server metadata.
    All configured backgrounds must share the policy's robot and camera layout.
    """
    raw = override.get("task_name")
    if raw is None:
        return []
    if isinstance(raw, str | bytes):
        scalar = str(raw).strip()
        return [scalar] if scalar else []
    if not isinstance(raw, Sequence):
        raise TypeError(
            f"GenieSim override_cfg.task_name must be a string or a sequence of "
            f"strings, got {type(raw).__name__}"
        )
    names = [str(value).strip() for value in raw]
    if not names or not all(names):
        raise ValueError(
            "GenieSim override_cfg.task_name must be a non-empty list of "
            f"non-empty names, got {list(raw)!r}"
        )
    # This list is a priority order consumed by ``next()``, so a repeat is
    # unreachable and can only mean the author mistyped.
    if len(set(names)) != len(names):
        raise ValueError(
            f"GenieSim override_cfg.task_name must not repeat a background, got {names!r}"
        )
    return names


def resolve_geniesim_render_width(override: dict[str, Any]) -> int | None:
    """Resolve a positive render width; None or zero keeps the native resolution."""
    raw = override.get("render_width")
    if raw is None:
        return None
    width = int(raw)
    if width == 0:
        return None
    if width < 0:
        raise ValueError(
            f"GenieSim render_width must be positive or 0 (=native), got {width}"
        )
    return width


_RANDOMIZATION_KEYS = frozenset(
    {
        "object_xy_radius",
        "exclude",
        "min_gap",
        "object_constraints",
        "init_joint_thresh",
        "init_base_xy_thresh",
        "joint_pd",
        "render",
        "use_task_generalization",
        "use_benchmark_generalization",
    }
)
_RANDOMIZATION_SUBKEYS = {
    "render": frozenset({"lights", "material", "background", "camera_pose"}),
}


def resolve_geniesim_randomization(override: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the randomization section passed to each GenieSim task init."""
    raw = override.get("randomization")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"GenieSim randomization must be a mapping of the sim's randomization keys, got {raw!r}"
        )
    if "object_xy_radius" in raw and raw["object_xy_radius"] is None:
        raise ValueError(
            "GenieSim randomization.object_xy_radius: null does not turn object "
            "randomization off (the engine falls back to 0.03 m); write object_xy_radius: 0"
        )
    unknown = [key for key in raw if key not in _RANDOMIZATION_KEYS]
    for section, allowed in _RANDOMIZATION_SUBKEYS.items():
        sub = raw.get(section)
        if sub is None:
            continue
        if not isinstance(sub, dict):
            raise ValueError(
                f"GenieSim randomization.{section} must be a mapping, got {sub!r}"
            )
        unknown += [f"{section}.{key}" for key in sub if key not in allowed]
    if unknown:
        raise ValueError(
            f"GenieSim randomization has unsupported keys {unknown}. "
            f"Supported keys: {sorted(_RANDOMIZATION_KEYS)}, "
            f"render.{sorted(_RANDOMIZATION_SUBKEYS['render'])}"
        )
    return raw


# Endpoints stay in the single logical-rank-indexed ``addresses`` list so that
# uniqueness is validated in exactly one place (``resolve_geniesim_address``).
# A slot that could also name its own ``address`` would let two slots point at
# the same sim server with no check at all -- the failure is two clients
# fighting over one server's task, which looks like a hang, not a config error.
_SLOT_FORBIDDEN_KEYS = ("address", "addresses", "slots")

# Keys that decide *which* work items an env walks. Two slots agreeing on all
# of them share one catalogue and therefore must share one shard group; slots
# differing in any of them own independent catalogues.
_WORK_ITEM_IDENTITY_KEYS = (
    "task_name",
    "sub_task_name",
    "instance_ids",
    "instruction_ids_by_instance",
    "num_episodes",
    "work_item_budget",
)


def _work_item_identity(override: Mapping[str, Any]) -> str:
    """Canonical identity of one env's work-item catalogue.

    JSON with sorted keys rather than ``repr``: two slots may spell the same
    ``instruction_ids_by_instance`` in different key order and must still be
    recognised as the same catalogue.
    """
    return json.dumps(
        {key: override.get(key) for key in _WORK_ITEM_IDENTITY_KEYS},
        sort_keys=True,
        default=str,
    )


def resolve_geniesim_slot_overrides(
    override: dict[str, Any], *, world_size: int
) -> list[dict[str, Any]] | None:
    """Merge overrides for each logical slot (worker rank × stages + stage ID).

    Return None when no slot-specific overrides are configured."""
    raw_slots = override.get("slots")
    if raw_slots is None:
        return None
    if isinstance(raw_slots, str | bytes) or not isinstance(raw_slots, Sequence):
        raise TypeError("GenieSim override_cfg.slots must be a sequence")
    if len(raw_slots) != world_size:
        raise ValueError(
            "GenieSim override_cfg.slots must contain exactly one entry per "
            f"logical env slot: expected {world_size}, got {len(raw_slots)}"
        )
    base = {key: value for key, value in override.items() if key != "slots"}
    merged: list[dict[str, Any]] = []
    for index, slot in enumerate(raw_slots):
        if not isinstance(slot, Mapping):
            raise TypeError(
                f"GenieSim override_cfg.slots[{index}] must be a mapping, "
                f"got {type(slot).__name__}"
            )
        forbidden = [key for key in _SLOT_FORBIDDEN_KEYS if key in slot]
        if forbidden:
            raise ValueError(
                f"GenieSim override_cfg.slots[{index}] must not set {forbidden}: "
                "endpoints come from the logical-rank-indexed override_cfg.addresses, "
                "which is where uniqueness across slots is validated"
            )
        merged.append(update_nested_cfg(copy.deepcopy(base), dict(slot)))
    return merged


def resolve_geniesim_work_item_shard(
    override: Mapping[str, Any],
    slot_overrides: Sequence[Mapping[str, Any]] | None,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """``(shard_rank, shard_world_size)`` for this env's work-item cursor.

    The cursor deals items round-robin across every env that shares the same
    catalogue: base ``shard_rank * num_envs``, stride ``shard_world * num_envs``.
    Getting the group wrong is silent -- too large a ``shard_world`` makes the
    stride skip whole congruence classes, e.g. a 100-item catalogue walked with
    stride 32 only ever reaches 25 items.

    Three sources, in precedence order:

    1. explicit ``work_item_shard_rank`` + ``work_item_shard_world_size``;
    2. derived from ``slots`` -- position within, and size of, the set of slots
       whose catalogue keys match this one. Slots with distinct tasks each get
       ``(0, 1)`` and therefore walk their own catalogue from item 0; R replicas
       of one task get ``(0..R-1, R)``;
    3. no ``slots`` -> ``(rank, world_size)``, i.e. the whole job is one shard
       group, which is the historical single-task behaviour.
    """
    explicit_rank = override.get("work_item_shard_rank")
    explicit_world = override.get("work_item_shard_world_size")
    if (explicit_rank is None) != (explicit_world is None):
        raise ValueError(
            "GenieSim work_item_shard_rank and work_item_shard_world_size must be "
            "set together (a half-specified shard silently mixes two conventions)"
        )
    if explicit_rank is not None:
        shard_rank, shard_world = int(explicit_rank), int(explicit_world)
    elif slot_overrides is None:
        shard_rank, shard_world = int(rank), int(world_size)
    else:
        identity = _work_item_identity(override)
        peers = [
            index
            for index, slot_override in enumerate(slot_overrides)
            if _work_item_identity(slot_override) == identity
        ]
        shard_rank, shard_world = peers.index(rank), len(peers)
    if shard_world < 1:
        raise ValueError(
            f"GenieSim work_item_shard_world_size must be >= 1, got {shard_world}"
        )
    if not 0 <= shard_rank < shard_world:
        raise ValueError(
            f"GenieSim work_item_shard_rank must be in [0, {shard_world}), "
            f"got {shard_rank}"
        )
    return shard_rank, shard_world
