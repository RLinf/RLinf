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

"""Official RLinf adapter for an externally launched GenieSim server."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.geniesim.adapter import _CAM_MAP, GenieSimAdapter
from rlinf.envs.utils import to_tensor
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import update_nested_cfg


def _nonfinite_state_error_types() -> tuple[type[BaseException], ...]:
    """Sim-side non-finite error classes, imported lazily.

    Returns an empty tuple when ``rlinf_geniesim`` is unavailable, which
    disables reset retries instead of breaking this module's import.
    """
    try:
        from rlinf_geniesim.envs.vec_env import SimStateNonFiniteError
    except Exception:  # pragma: no cover - depends on the deployment
        return ()
    return (SimStateNonFiniteError,)


# Camera keys in the raw simulator observation.
_IMAGE_KEYS = frozenset(_CAM_MAP)


@dataclasses.dataclass(frozen=True)
class GenieSimTaskMetadata:
    task_name: str
    sub_task_name: str
    switchable_instance_ids: list[int]
    instructions_by_instance: dict[int, list[int]]


@dataclasses.dataclass(frozen=True)
class GenieSimWorkItem:
    task_name: str
    sub_task_name: str
    instance_id: int
    instruction_id: int
    episode_idx: int


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _override_dict(cfg: Any) -> dict[str, Any]:
    if "override_cfg" not in cfg or cfg.override_cfg is None:
        return {}
    return dict(OmegaConf.to_container(cfg.override_cfg, resolve=True) or {})


def _make_geniesim_client(backend: str, **kwargs: Any) -> Any:
    from rlinf_geniesim.comm import make_client

    return make_client(backend, **kwargs)


def _make_geniesim_venv(client: Any, **kwargs: Any) -> Any:
    from rlinf_geniesim.envs.vec_env import GenieSimVecEnv

    return GenieSimVecEnv(client, **kwargs)


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


def resolve_geniesim_sub_tasks(
    override: dict[str, Any], *, rank: int, world_size: int
) -> tuple[list[str], int]:
    """Resolve the global task list and this logical slot's initial task.

    Duplicate names weight task frequency; the list need not match the slot count."""
    raw_names = override.get("sub_task_names")
    if raw_names is None:
        scalar = str(override.get("sub_task_name", "")).strip()
        return ([scalar] if scalar else []), 0

    if isinstance(raw_names, str | bytes) or not isinstance(raw_names, Sequence):
        raise TypeError("GenieSim override_cfg.sub_task_names must be a sequence")
    names = [str(value).strip() for value in raw_names]
    if not names or not all(names):
        raise ValueError(
            "GenieSim override_cfg.sub_task_names must be a non-empty list of "
            f"non-empty names, got {list(raw_names)!r}"
        )
    # Unlike ``addresses``, no length check and no uniqueness check -- see the
    # docstring: this list is global, and duplicates are the weighting knob.
    return names, rank % len(names)


def resolve_geniesim_task_names(override: dict[str, Any]) -> list[str]:
    """Resolve allowed backgrounds in preference order.

    Each sub-task selects the first compatible background from server metadata.
    All configured backgrounds must share the policy's robot and camera layout."""
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
    # Duplicates are rejected here, unlike in ``sub_task_names`` where they are
    # the weighting knob: this list is a priority order consumed by ``next()``,
    # so a repeat is unreachable and can only mean the author mistyped.
    if len(set(names)) != len(names):
        raise ValueError(
            f"GenieSim override_cfg.task_name must not repeat a background, got {names!r}"
        )
    return names


# Mixed into the per-block seed below. Any odd constant works; it only has to be
# fixed forever, because changing it re-deals every task schedule.
_SUB_TASK_STREAM_SALT = 1000003


def geniesim_sub_task_stream(
    num_tasks: int,
    length: int,
    *,
    base_seed: int,
    window: int = 1,
    max_retries: int = 32,
) -> list[int]:
    """A deterministic WITHOUT-REPLACEMENT stream of sub-task indices.

    The stream is the concatenation of independent random permutations of
    ``range(num_tasks)`` ("shuffled blocks"). Consuming it ``window`` indices at
    a time -- one per env rank, one draw per rotation round -- is what decouples
    the task count ``N`` from the server count ``M`` entirely: nothing here
    depends on how ``M`` and ``N`` relate.

    Why blocks and not i.i.d. sampling. Drawing each round independently gives
    every task the right *marginal* probability but only converges to equal
    frequency in expectation, leaving O(sqrt(k)) imbalance in any finite rollout
    buffer -- and, worse, a nonzero chance that a task is absent from a buffer
    altogether. A shuffled block gives every task **exactly** the same count
    over any whole number of blocks, so the buffer is exactly balanced whenever
    ``rollout_epoch * M`` is a multiple of ``N`` (minimal such value:
    ``N // gcd(M, N)``). Same wall-clock cost, strictly lower variance, and the
    order is still random -- which is the point of not using the deterministic
    ``+M`` stride: that one couples the task-pair sequence to the iteration
    index with period ``N``, and the advantage-normalisation baseline rides
    along with it.

    ``window`` (the number of ranks drawn per round) additionally forbids each
    block's first ``window - 1`` entries from repeating the previous block's
    last ``window - 1``. Within one block the entries are distinct by
    construction; only a window straddling a block seam could put two ranks on
    the SAME sub-task, wasting one server's worth of task diversity for that
    episode. The constraint is satisfiable iff ``num_tasks >= 2 * (window - 1)``;
    past ``max_retries`` the last shuffle is accepted rather than looping
    forever, so a pathological ``M`` near ``N`` degrades diversity instead of
    hanging.

    Determinism ACROSS PROCESSES is load-bearing, not a convenience: every env
    rank calls this independently and reads only its own slot, which is what
    removes the need for any cross-rank coordination channel (the ranks are
    separate Ray workers with no barrier between them at reset time). Hence the
    per-block seed is an explicit integer mix and NEVER ``hash()`` of a tuple
    containing a ``str`` -- ``PYTHONHASHSEED`` randomises that per process, and
    the failure would be silent: ranks would disagree on the schedule, landing
    two servers on one task while another task is never drawn at all.

    The stream is also PREFIX-STABLE: block ``b`` depends only on blocks
    ``< b``, so asking for a longer stream reproduces the shorter one exactly.
    Callers can therefore regrow it on demand instead of preallocating.
    """
    if num_tasks <= 0:
        raise ValueError(
            f"geniesim_sub_task_stream needs num_tasks >= 1, got {num_tasks}"
        )
    if length <= 0:
        return []
    overlap = min(max(int(window) - 1, 0), num_tasks)
    stream: list[int] = []
    block = 0
    while len(stream) < length:
        rng = random.Random(
            (int(base_seed) * _SUB_TASK_STREAM_SALT + block) & 0xFFFF_FFFF
        )
        tail = set(stream[len(stream) - overlap :]) if overlap else set()
        perm = list(range(num_tasks))
        for _ in range(max(int(max_retries), 1)):
            rng.shuffle(perm)
            if not tail or not tail.intersection(perm[:overlap]):
                break
        stream.extend(perm)
        block += 1
    return stream[:length]


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


def parse_geniesim_task_metadata(
    response: dict[str, Any], *, task_names: Sequence[str], sub_task_name: str
) -> GenieSimTaskMetadata:
    """Validate the server's catalog entry for ``sub_task_name`` and pick its
    background.

    ``task_names`` is a preference list (see ``resolve_geniesim_task_names``);
    the returned ``metadata.task_name`` is the first entry the server reports as
    valid for this sub-task. A sub-task none of them covers is a hard error --
    never a silent substitution, because the failure would be a run on a
    different scene that looks perfectly healthy in the logs.
    """
    if not isinstance(response, dict) or not isinstance(
        response.get("sub_tasks"), dict
    ):
        raise ValueError("GenieSim task_metadata response must contain sub_tasks")
    catalog = response["sub_tasks"]
    if sub_task_name not in catalog:
        raise ValueError(
            f"Unknown GenieSim sub_task_name={sub_task_name!r}; "
            f"available={sorted(catalog)}"
        )
    entry = catalog[sub_task_name]

    valid_task_names: list[str] = []
    for raw in entry.get("task_names") or []:
        name = raw.get("task_name") if isinstance(raw, dict) else raw
        if name:
            valid_task_names.append(str(name))
    chosen = next((name for name in task_names if name in valid_task_names), None)
    if chosen is None:
        raise ValueError(
            f"none of task_name={list(task_names)!r} is valid for sub task "
            f"{sub_task_name!r}; available={valid_task_names}"
        )

    switchable = [int(value) for value in entry.get("switchable_instance_ids") or []]
    if not switchable:
        raise ValueError(
            f"GenieSim sub task {sub_task_name!r} has no switchable instances"
        )

    raw_by_instance = entry.get("instructions_by_instance") or {}
    instructions_by_instance: dict[int, list[int]] = {}
    for instance_id in switchable:
        raw_entries = raw_by_instance.get(
            instance_id, raw_by_instance.get(str(instance_id))
        )
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"GenieSim instance {instance_id} has no instructions")
        instructions_by_instance[instance_id] = [int(raw["id"]) for raw in raw_entries]

    return GenieSimTaskMetadata(
        task_name=chosen,
        sub_task_name=sub_task_name,
        switchable_instance_ids=switchable,
        instructions_by_instance=instructions_by_instance,
    )


def build_geniesim_work_items(
    metadata: GenieSimTaskMetadata,
    *,
    instance_ids: Sequence[int] | None,
    instruction_ids_by_instance: dict[int, Sequence[int]] | None,
    num_episodes: int,
) -> list[GenieSimWorkItem]:
    selected_instances = (
        list(metadata.switchable_instance_ids)
        if instance_ids is None
        else [int(value) for value in instance_ids]
    )
    requested = {
        int(instance_id): [int(value) for value in values]
        for instance_id, values in (instruction_ids_by_instance or {}).items()
    }
    items: list[GenieSimWorkItem] = []
    for episode_idx in range(int(num_episodes)):
        for instance_id in selected_instances:
            available = metadata.instructions_by_instance[instance_id]
            instruction_ids = requested.get(instance_id)
            instructions = available if instruction_ids is None else instruction_ids
            unknown = set(instructions) - set(available)
            if unknown:
                raise ValueError(
                    f"GenieSim instance {instance_id} has no instruction IDs {sorted(unknown)}"
                )
            for instruction in instructions:
                items.append(
                    GenieSimWorkItem(
                        task_name=metadata.task_name,
                        sub_task_name=metadata.sub_task_name,
                        instance_id=instance_id,
                        instruction_id=instruction,
                        episode_idx=episode_idx,
                    )
                )
    if not items:
        raise ValueError("GenieSim work-item selection is empty")
    return items


def derive_geniesim_seed(base_seed: int, item: GenieSimWorkItem) -> int:
    seed = int(base_seed)
    if seed == -1:
        return -1
    payload = "\0".join(
        (
            str(seed),
            item.task_name,
            item.sub_task_name,
            str(item.instance_id),
            str(item.instruction_id),
            str(item.episode_idx),
        )
    ).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def build_geniesim_reset_vectors(
    num_envs: int,
    env_ids: Sequence[int],
    work_items_by_env: Sequence[GenieSimWorkItem],
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeds = np.full(num_envs, -1, dtype=np.int64)
    instance_ids = np.full(num_envs, -1, dtype=np.int64)
    instruction_ids = np.full(num_envs, -1, dtype=np.int64)
    for env_id in np.asarray(env_ids, dtype=np.int64).reshape(-1).tolist():
        item = work_items_by_env[env_id]
        seeds[env_id] = derive_geniesim_seed(base_seed, item)
        instance_ids[env_id] = int(item.instance_id)
        instruction_ids[env_id] = int(item.instruction_id)
    return seeds, instance_ids, instruction_ids


class GenieSimEnv:
    """Thin env wrapper matching official EnvWorker's reset/chunk_step contract."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info=None,
    ) -> None:
        self.cfg = cfg
        self._num_envs = int(num_envs)
        # Logical slot identity includes the worker rank and pipeline stage.
        self._seed_offset = int(seed_offset)
        self._total_num_processes = int(total_num_processes)
        # The physical worker rank is deliberately not kept: it aliases every
        # pipeline stage of a process onto one slot.
        del worker_info
        self._auto_reset = bool(cfg.get("auto_reset", False))
        self._ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self._max_episode_steps = int(cfg.get("max_episode_steps", 200))

        override = _override_dict(cfg)
        # Per-logical-slot config, if any. Applied before every other key is
        # read, so a slot can override anything the shared block sets (task,
        # instances, episode budget, physics backend, ...).
        self._slot_overrides = resolve_geniesim_slot_overrides(
            override, world_size=self._total_num_processes
        )
        if self._slot_overrides is not None:
            override = self._slot_overrides[self._seed_offset]
        # Retry only reset failures; a lost step has no valid transition to return.
        self._nonfinite_state_recovery = bool(
            override.get("nonfinite_state_recovery", False)
        )
        self._nonfinite_state_reset_retries = int(
            override.get("nonfinite_state_reset_retries", 3)
        )
        if self._nonfinite_state_reset_retries < 1:
            raise ValueError(
                f"GenieSim nonfinite_state_reset_retries must be >= 1, "
                f"got {self._nonfinite_state_reset_retries}"
            )
        self._address = resolve_geniesim_address(
            override,
            rank=self._seed_offset,
            world_size=self._total_num_processes,
        )
        # Resolve backgrounds before selecting a sub-task.
        self._task_names = resolve_geniesim_task_names(override)
        # Select this worker's initial sub-task.
        self._sub_task_names, self._sub_task_idx = resolve_geniesim_sub_tasks(
            override,
            rank=self._seed_offset,
            world_size=self._total_num_processes,
        )
        if not self._task_names or not self._sub_task_names:
            raise ValueError("GenieSim task_name and sub_task_name are required")
        # Binding may select a different background for this sub-task.
        self._task_name = self._task_names[0]
        self._sub_task_name = self._sub_task_names[self._sub_task_idx]
        # Rotate after this many rollout epochs; zero disables rotation.
        self._sub_task_rotate_every = int(override.get("sub_task_rotate_every", 0) or 0)
        if self._sub_task_rotate_every < 0:
            raise ValueError(
                f"GenieSim sub_task_rotate_every must be >= 0, "
                f"got {self._sub_task_rotate_every}"
            )
        self._rotation_counter = 0
        # Shuffle draws without replacement; stride advances deterministically.
        self._sub_task_sampling = (
            str(override.get("sub_task_sampling", "stride")).strip().lower()
        )
        if self._sub_task_sampling not in ("stride", "shuffle"):
            raise ValueError(
                f"GenieSim sub_task_sampling must be 'stride' or 'shuffle', "
                f"got {self._sub_task_sampling!r}"
            )
        # Count rotation rounds separately from rollout epochs.
        self._sub_task_round = 0
        self._sub_task_slot_stream: list[int] = []
        if str(override.get("instruction_mode", "full")) != "full":
            raise NotImplementedError("GenieSim only supports instruction_mode=full")
        self._reset_blocking = bool(override.get("reset_blocking", True))
        if not self._reset_blocking:
            raise NotImplementedError("GenieSim only supports reset_blocking=true")

        backend, client_kwargs = _client_kwargs(override, address=self._address)
        self._client = _make_geniesim_client(backend, **client_kwargs)
        backend_cfg = dict(override.get("backend_cfg") or {})
        if backend_cfg:
            self._client.set_backends(backend_cfg)
        self._venv = _make_geniesim_venv(
            self._client,
            to_torch=bool(override.get("venv_to_torch", True)),
            device=str(override.get("venv_device", "cuda:0")),
            # Keep per-step boundaries; this adapter owns auto-reset.
            auto_reset=False,
            # The client truncates on its own counter, so it needs the same
            # limit; its default is 200 and would cut a longer episode short.
            max_episode_steps=self._max_episode_steps,
        )
        # Retain settings needed when the sub-task changes.
        self._render_width = resolve_geniesim_render_width(override)
        self._randomization = resolve_geniesim_randomization(override)
        self._eval_interval = int(override.get("eval_interval", 1))
        self._work_item_spec = {
            "instance_ids": override.get("instance_ids"),
            "instruction_ids_by_instance": override.get("instruction_ids_by_instance"),
            "num_episodes": int(override.get("num_episodes", 1)),
            "budget": int(override.get("work_item_budget", 0) or 0),
        }
        self._need_images = bool(override.get("need_images", True))
        # Render only the last action substep when enabled. Accept the legacy key.
        self._images_at_chunk_end_only = bool(
            override.get(
                "images_at_chunk_end_only",
                override.get("render_last_frame_only", False),
            )
        )
        # Video recording requires a frame for every substep.
        self._save_video = bool((cfg.get("video_cfg") or {}).get("save_video", False))
        self._base_seed = int(override.get("geniesim_env_seed", cfg.get("seed", 0)))
        self._cycle_work_items = bool(override.get("cycle_work_items", True))

        self._adapter = GenieSimAdapter(
            main_image_key=str(cfg.get("main_image_key", "main")),
            extra_image_keys=(
                list(cfg.extra_image_keys) if "extra_image_keys" in cfg else None
            ),
        )
        self._elapsed_steps = np.zeros(self._num_envs, dtype=np.int32)
        # Shard work by task catalogue and GRPO group; group members share an item and seed.
        (
            self._work_item_shard_rank,
            self._work_item_shard_world_size,
        ) = resolve_geniesim_work_item_shard(
            override,
            self._slot_overrides,
            rank=self._seed_offset,
            world_size=self._total_num_processes,
        )
        self._group_size = int(cfg.get("group_size", 1) or 1)
        if self._group_size < 1:
            raise ValueError(
                f"GenieSim group_size must be >= 1, got {self._group_size}"
            )
        if self._num_envs % self._group_size != 0:
            # ``config.py`` does not check this pairing, so without the raise a
            # trailing partial group would quietly get its own work item and the
            # GRPO baseline would be a mean over *different* initial states --
            # which looks exactly like "the run just does not learn".
            raise ValueError(
                f"GenieSim group_size {self._group_size} must divide the per-process "
                f"env count {self._num_envs}"
            )
        self._num_group = self._num_envs // self._group_size
        self._work_stride = self._work_item_shard_world_size * self._num_group
        # Keep task cursors by reference so revisiting a task resumes its work sequence.
        self._work_state: dict[str, tuple[list[GenieSimWorkItem], list[int]]] = {}
        # Sub-tasks whose cursor has already been consumed at least once. A
        # freshly-bound task's cursor sits at the rank's base offset and has not
        # been used yet, so the first reset on it must NOT advance. This is the
        # per-sub-task generalization of ``_first_reset``.
        self._work_started: set[str] = set()
        self._action_dim = -1

        # All ranks share a fixed scheduling seed, independent of later reset seeds.
        self._sub_task_stream_seed = int(self._base_seed)
        # Resolve the curriculum before building the first scene.
        self._task_metadata = self._validate_sub_task_curriculum()
        if self._sub_task_sampling == "shuffle" and len(self._sub_task_names) > 1:
            # Draw the initial assignment from the shuffled stream.
            self._sub_task_idx = self._draw_sub_task_slot(0)
            self._sub_task_name = self._sub_task_names[self._sub_task_idx]

        self._bind_sub_task(self._sub_task_name)
        self.is_start = True
        # Track the first reset independently of the worker's is_start flag.
        self._first_reset = True
        # One-shot flag for the group init-state agreement check below.
        self._group_state_checked = False
        # Reuse camera frames when intermediate substeps skip rendering.
        self._last_raw_obs: dict[str, Any] | None = None
        self._last_is_success = np.zeros(self._num_envs, dtype=bool)
        # PPO uses the change in episode progress as reward.
        self._prev_progress = np.zeros(self._num_envs, dtype=np.float32)
        self._episode_return = np.zeros(self._num_envs, dtype=np.float32)
        self._success_once = np.zeros(self._num_envs, dtype=bool)
        self._terminated = np.zeros(self._num_envs, dtype=bool)
        self._truncated = np.zeros(self._num_envs, dtype=bool)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def seed(self) -> int:
        """Effective seed of this env instance.

        ``rlinf.envs.wrappers.RecordVideo`` requires this attribute (it refuses
        to wrap an env without one) and names its output directory after it.
        """
        # -1 disables reset randomization on every server. Do not turn it into
        # a nonnegative seed on slots with a positive offset.
        if self._base_seed == -1:
            return -1
        return self._base_seed + self._seed_offset

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _init_config(self) -> dict[str, Any] | None:
        """Collect optional task-init sections without changing server defaults."""
        config: dict[str, Any] = {}
        if self._render_width is not None:
            config["observation"] = {"render_width": self._render_width}
        if self._randomization is not None:
            config["randomization"] = copy.deepcopy(self._randomization)
        return config or None

    def _bind_sub_task(self, sub_task_name: str) -> None:
        """Switch the server task and restore its catalogue, cursor, and action layout."""
        self._sub_task_name = sub_task_name
        self._metadata = self._task_metadata[sub_task_name]
        # Each sub-task can select a different background from the configured list.
        self._task_name = self._metadata.task_name
        self._venv.init(
            {
                "task_name": self._task_name,
                "sub_task_name": sub_task_name,
                # The server needs num_envs on every task switch to retain its layout.
                "num_envs": self._num_envs,
                "eval_interval": self._eval_interval,
            },
            config=self._init_config(),
        )
        new_action_dim = int(self._venv.action_dim)
        if self._action_dim >= 0 and new_action_dim != self._action_dim:
            # A task switch cannot change the policy action layout.
            raise RuntimeError(
                f"GenieSim action_dim changed across a sub-task switch "
                f"({self._action_dim} -> {new_action_dim}, now {sub_task_name!r}). "
                f"The policy's action head cannot follow; every background in "
                f"task_name={self._task_names!r} must share one robot."
            )
        self._action_dim = new_action_dim

        cached = self._work_state.get(sub_task_name)
        if cached is None:
            work_items = build_geniesim_work_items(
                self._metadata,
                instance_ids=self._work_item_spec["instance_ids"],
                instruction_ids_by_instance=self._work_item_spec[
                    "instruction_ids_by_instance"
                ],
                num_episodes=self._work_item_spec["num_episodes"],
            )
            budget = self._work_item_spec["budget"]
            if budget > 0:
                work_items = work_items[:budget]
            # Slots with distinct catalogues start at their own catalogue offset.
            base = self._work_item_shard_rank * self._num_group
            cursor = [
                (base + env_id // self._group_size) % len(work_items)
                for env_id in range(self._num_envs)
            ]
            cached = (work_items, cursor)
            self._work_state[sub_task_name] = cached
        self._work_items, self._work_cursor = cached

        # Log the slot mapping again whenever the task changes.
        get_logger().info(
            f"[geniesim] slot={self._seed_offset}/{self._total_num_processes} "
            f"address={self._address} "
            f"task={self._task_name} sub_task={self._sub_task_name} "
            f"({self._sub_task_idx + 1}/{len(self._sub_task_names)}) "
            f"num_envs={self._num_envs} server action_dim={self._action_dim} "
            f"work_items={len(self._work_items)} "
            f"shard={self._work_item_shard_rank}/{self._work_item_shard_world_size} "
            f"group_size={self._group_size} num_group={self._num_group} "
            f"cursor={self._work_cursor} stride={self._work_stride}"
        )

    def _validate_sub_task_curriculum(self) -> dict[str, GenieSimTaskMetadata]:
        """Validate each distinct task once and retain its resolved catalogue."""
        resolved: dict[str, GenieSimTaskMetadata] = {}
        for name in dict.fromkeys(self._sub_task_names):  # de-dup, keep order
            metadata = parse_geniesim_task_metadata(
                self._client.task_metadata(sub_task_name=name),
                task_names=self._task_names,
                sub_task_name=name,
            )
            resolved[name] = metadata
        by_background: dict[str, list[str]] = {}
        for sub, metadata in resolved.items():
            by_background.setdefault(metadata.task_name, []).append(sub)
        get_logger().info(
            f"[geniesim] curriculum N={len(self._sub_task_names)} over "
            f"{len(by_background)} background(s): "
            + "; ".join(
                f"{background}={subs}"
                for background, subs in sorted(by_background.items())
            )
        )
        return resolved

    def _draw_sub_task_slot(self, round_index: int) -> int:
        """Select this rank's slot from a shared, prefix-stable shuffled schedule."""
        num_ranks = max(self._total_num_processes, 1)
        needed = (round_index + 1) * num_ranks
        if len(self._sub_task_slot_stream) < needed:
            self._sub_task_slot_stream = geniesim_sub_task_stream(
                len(self._sub_task_names),
                needed,
                base_seed=self._sub_task_stream_seed,
                window=num_ranks,
            )
        return self._sub_task_slot_stream[round_index * num_ranks + self._seed_offset]

    def _maybe_rotate_sub_task(self) -> None:
        """Advance the task schedule at a whole-env reset.

        Stride mode advances by the logical slot count; shuffle mode draws from
        the shared stream. Repeated task assignments do not rebuild the scene."""

        if self._sub_task_rotate_every <= 0 or len(self._sub_task_names) <= 1:
            return
        self._rotation_counter += 1
        if self._rotation_counter < self._sub_task_rotate_every:
            return
        self._rotation_counter = 0
        self._sub_task_round += 1
        if self._sub_task_sampling == "shuffle":
            next_idx = self._draw_sub_task_slot(self._sub_task_round)
        else:
            next_idx = (self._sub_task_idx + self._total_num_processes) % len(
                self._sub_task_names
            )
        if next_idx == self._sub_task_idx:
            # stride mode: N divides the stride, nothing to switch to.
            # shuffle mode: this rank happened to redraw its current task
            # (probability ~1/N) -- a free skipped rebuild, not an error.
            return
        self._sub_task_idx = next_idx
        next_name = self._sub_task_names[next_idx]
        get_logger().info(
            f"[geniesim] rotating sub_task {self._sub_task_name!r} -> "
            f"{next_name!r} (rank {self._seed_offset}/{self._total_num_processes}, "
            f"mode={self._sub_task_sampling} round={self._sub_task_round})"
        )
        self._bind_sub_task(next_name)

    @property
    def _success_metric_key(self) -> str:
        """Task-specific metric key for the currently bound task."""
        return f"success/{self._sub_task_name}"

    def _current_work_items(self) -> list[GenieSimWorkItem]:
        return [self._work_items[cursor] for cursor in self._work_cursor]

    def _advance_work(self, group_id: int) -> None:
        """Move one whole group to its next work item, in lockstep.

        The cursor stays a per-env list so that every reader keeps indexing by
        env id (``_current_work_items``, the reset log); it is
        the *value* that a group shares, so it is read from the group's first env
        and written back to all of them. At ``group_size == 1`` that is the old
        single-element update, unchanged.
        """
        first = group_id * self._group_size
        next_cursor = self._work_cursor[first] + self._work_stride
        if next_cursor >= len(self._work_items):
            next_cursor = (
                next_cursor % len(self._work_items)
                if self._cycle_work_items
                else self._work_cursor[first]
            )
        for env_id in range(first, first + self._group_size):
            self._work_cursor[env_id] = next_cursor

    def _venv_reset_with_retry(
        self,
        ids: np.ndarray,
        seeds: np.ndarray,
        instance_ids: np.ndarray,
        instruction_ids: np.ndarray,
    ) -> tuple[dict, dict]:
        """Reset, retrying when the reset itself returns non-finite state.

        Retries only when recovery is enabled. A bad frame here means the
        fresh episode diverged before its first step, so there is no last-good
        obs to fall back on -- retrying the reset is the only recovery
        available.
        """
        attempts = (
            self._nonfinite_state_reset_retries if self._nonfinite_state_recovery else 1
        )
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return self._venv.reset(
                    env_ids=ids.tolist(),
                    seeds=seeds,
                    instance_ids=instance_ids,
                    instruction_ids=instruction_ids,
                    block=self._reset_blocking,
                )
            except _nonfinite_state_error_types() as exc:
                if not self._nonfinite_state_recovery:
                    raise
                last_exc = exc
                get_logger().error(
                    "[geniesim] reset returned non-finite state (attempt %d/%d), "
                    "envs=%s: %s",
                    attempt + 1,
                    attempts,
                    ids.tolist(),
                    exc,
                )
        raise RuntimeError(
            f"GenieSim reset returned non-finite state on all {attempts} attempts, "
            f"envs={ids.tolist()}"
        ) from last_exc

    def reset(
        self, *, seed: int | None = None, options=None, env_idx=None
    ) -> tuple[dict, dict]:
        del options
        if seed is not None:
            self._base_seed = int(seed)
        ids = (
            np.arange(self._num_envs, dtype=np.int64)
            if env_idx is None
            else np.asarray(env_idx, dtype=np.int64).reshape(-1)
        )
        # Rotate only at whole-env resets; partial resets must not change other episodes.
        if env_idx is None and not self._first_reset:
            self._maybe_rotate_sub_task()

        if self._sub_task_name in self._work_started:
            # Advance each group once to keep its reset work item and seed aligned.
            groups = sorted({env_id // self._group_size for env_id in ids.tolist()})
            if self._group_size > 1 and int(ids.size) != self._group_size * len(groups):
                # Partial group resets break GRPO initial-state alignment.
                get_logger().warning(
                    "[geniesim] partial group reset with group_size=%d: "
                    "env_idx=%s covers %d group(s) but only %d env(s); group "
                    "members will stop sharing a work item",
                    self._group_size,
                    ids.tolist(),
                    len(groups),
                    int(ids.size),
                )
            for group_id in groups:
                self._advance_work(group_id)
        else:
            # First visit to this sub-task: its cursor was just laid down at the
            # rank's base offset and has never been consumed, so advancing now
            # would skip the first work item of every env. (The single-task case
            # is the ``_first_reset`` guard this generalizes.)
            self._work_started.add(self._sub_task_name)

        seeds, instance_ids, instruction_ids = build_geniesim_reset_vectors(
            self._num_envs,
            ids,
            self._current_work_items(),
            self.seed,
        )
        raw_obs, infos = self._venv_reset_with_retry(
            ids, seeds, instance_ids, instruction_ids
        )
        self._last_raw_obs = raw_obs
        self._prev_progress[ids] = 0.0
        self._last_is_success[ids] = False
        self._elapsed_steps[ids] = 0
        self._episode_return[ids] = 0.0
        self._success_once[ids] = False
        self._terminated[ids] = False
        self._truncated[ids] = False
        self.is_start = False
        self._first_reset = False
        obs = self._adapter.wrap_batch(raw_obs)
        # Log the resolved prompt and work item to diagnose server-side instruction fallback.
        get_logger().info(
            f"[geniesim] instruction[0]={obs['task_descriptions'][0]!r} "
            f"(instance={instance_ids[0]} instruction_id={instruction_ids[0]} "
            f"work_item={self._work_cursor[int(ids[0])] + 1}/{len(self._work_items)} "
            f"group_size={self._group_size} "
            f"groups={sorted({c + 1 for c in self._work_cursor})})"
        )
        self._log_group_state_agreement(obs)
        return obs, dict(infos)

    def _log_group_state_agreement(self, obs: dict) -> None:
        """Once per run, measure whether a group really shares its initial state.

        Sharing a work item gives the group identical reset *arguments*; that the
        resulting worlds are identical is the server's side of the contract, and
        nothing else in the run would ever check it. A GRPO baseline built over
        group members that actually started from different states is not an
        error anywhere -- it just makes the advantage noise, i.e. a run that
        trains and does not improve. So compare the group members' proprio state
        vectors on the first reset and print the worst disagreement.

        No-op at ``group_size == 1`` (nothing to compare) and after the first
        reset, so PPO configs never reach past the first line.
        """
        if self._group_size < 2 or self._group_state_checked:
            return
        self._group_state_checked = True
        raw_states = obs.get("states")
        if raw_states is None:
            return
        states = _as_numpy(raw_states)
        if states.ndim != 2 or states.shape[0] != self._num_envs:
            return
        worst_group, worst_dev = -1, 0.0
        for group_id in range(self._num_group):
            first = group_id * self._group_size
            block = states[first : first + self._group_size]
            dev = float(np.max(np.abs(block - block[0]))) if block.size else 0.0
            if dev >= worst_dev:
                worst_group, worst_dev = group_id, dev
        log = get_logger().info if worst_dev == 0.0 else get_logger().warning
        log(
            "[geniesim] group init-state check: group_size=%d num_group=%d "
            "max |state - state_of_group_member_0| = %.3e (worst group %d); "
            "0 means the group really does share its initial state, which is "
            "what a group-relative (GRPO) baseline needs",
            self._group_size,
            self._num_group,
            worst_dev,
            worst_group,
        )

    @property
    def supports_chunk_render_skip(self) -> bool:
        """Whether configured chunk rendering can skip intermediate frames."""
        return self._need_images and self._images_at_chunk_end_only

    def _fill_unrendered_frames(self, raw_obs: dict[str, Any]) -> bool:
        """Fill missing camera frames from the last rendered observation."""
        missing = [
            key
            for key, value in raw_obs.items()
            if key in _IMAGE_KEYS and value is None
        ]
        if not missing:
            return False
        if not self.supports_chunk_render_skip:
            # Never train on a frozen reset frame when images are disabled.
            raise RuntimeError(
                f"need_images=false leaves the camera keys {missing} empty on every "
                "step, but the observation adapter requires camera frames. Backfilling "
                "them would freeze the image at the reset frame and train on it. Set "
                "need_images=true, and use images_at_chunk_end_only to cut render cost."
            )
        cached = self._last_raw_obs
        if cached is None or any(cached.get(key) is None for key in missing):
            raise RuntimeError(
                "chunk-end-only rendering has no cached frame to reuse for "
                f"{missing}: the first observation of an episode must be a real "
                "frame. This means reset returned no image, which should not happen."
            )
        for key in missing:
            raw_obs[key] = cached[key]
        return True

    def step(
        self,
        action: np.ndarray,
        *,
        need_images: bool | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Render every substep while recording video.
        if self._save_video and need_images is False:
            need_images = True
        actions = np.ascontiguousarray(_as_numpy(action), dtype=np.float32)
        if actions.ndim == 1 and self._num_envs == 1:
            actions = actions[None, :]
        expected = (self._num_envs, self._action_dim)
        if actions.shape != expected:
            raise ValueError(
                f"GenieSim actions must have shape {expected}, got {actions.shape}"
            )
        # The environment-level flag controls all image rendering.
        step_images = self._need_images and (
            True if need_images is None else need_images
        )
        raw_obs, rewards, terminations, truncations, infos = self._venv.step(
            actions[:, None, :],
            auto_reset=False,
            need_images=step_images,
            return_per_step_obs=False,
        )
        # Fill skipped frames before replacing the cached observation.
        self._fill_unrendered_frames(raw_obs)
        self._last_raw_obs = raw_obs
        obs = self._adapter.wrap_batch(raw_obs)
        # Use progress at the end of the action chunk.
        rewards = _as_numpy(rewards).astype(np.float32)[:, -1]
        terminations = _as_numpy(terminations).astype(bool)[:, 0]
        truncations = _as_numpy(truncations).astype(bool)[:, 0]
        success = _as_numpy(infos.get("is_success", np.zeros(self._num_envs))).astype(
            bool
        )
        return self._record_step(
            obs, rewards, success, terminations, truncations, infos
        )

    def _episode_info(self, info: dict) -> dict:
        """Attach adapter-owned metrics, measured until the first episode boundary."""
        info = dict(info)
        success = to_tensor(self._last_is_success.astype(np.float32))
        info["episode"] = {
            "success": success,
            self._success_metric_key: success.clone(),
            "success_once": to_tensor(self._success_once.astype(np.float32)),
            "success_at_end": to_tensor(self._last_is_success.copy()),
            "return": to_tensor(self._episode_return.copy()),
            "length": to_tensor(self._elapsed_steps.astype(np.float32)),
        }
        return info

    def _record_step(self, obs, progress, success, terminated, truncated, info):
        """Credit active episodes once; latch boundaries until an explicit reset."""
        active = ~(self._terminated | self._truncated)
        delta = np.zeros(self._num_envs, dtype=np.float32)
        delta[active] = progress[active] - self._prev_progress[active]
        self._prev_progress[active] = progress[active]
        self._last_is_success[active] = success[active]
        self._success_once |= success & active
        self._episode_return += delta
        self._elapsed_steps[active] += 1
        if not self._ignore_terminations:
            self._terminated |= (terminated | success) & active
        self._truncated |= (
            truncated | (self._elapsed_steps >= self._max_episode_steps)
        ) & active
        info = self._episode_info(info)
        info["success"] = self._last_is_success.copy()
        info["geniesim_progress_delta"] = delta.copy()
        if (self._terminated | self._truncated).any():
            info["final_observation"] = obs
        return (
            obs,
            to_tensor(delta),
            to_tensor(self._terminated.copy()),
            to_tensor(self._truncated.copy()),
            info,
        )

    def _finish_chunk(self, observations, rewards, terminated, truncated, infos):
        """Reset once at the chunk boundary and preserve terminal metrics."""
        done = _as_numpy(terminated[-1] | truncated[-1]).astype(bool)
        if done.any() and self._auto_reset:
            final_info = infos[-1]
            final_observation = observations[-1]
            observations[-1], infos[-1] = self.reset(env_idx=np.flatnonzero(done))
            infos[-1]["final_info"] = final_info
            infos[-1]["final_observation"] = final_observation
            infos[-1]["_final_info"] = done.copy()
            infos[-1]["_final_observation"] = done.copy()
        return (
            observations,
            torch.stack(rewards, dim=1),
            torch.stack(terminated, dim=1),
            torch.stack(truncated, dim=1),
            infos,
        )

    def chunk_step(
        self, actions: np.ndarray
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        action_chunk = np.ascontiguousarray(_as_numpy(actions), dtype=np.float32)
        if action_chunk.ndim == 2 and self._num_envs == 1:
            action_chunk = action_chunk[None, :, :]
        expected_prefix = (self._num_envs,)
        if (
            action_chunk.ndim != 3
            or action_chunk.shape[:1] != expected_prefix
            or action_chunk.shape[2] != self._action_dim
        ):
            raise ValueError(
                "GenieSim action chunks must have shape "
                f"({self._num_envs}, chunk, {self._action_dim}), got {action_chunk.shape}"
            )

        if self._can_batch_chunk(action_chunk.shape[1]):
            return self._chunk_step_batched(action_chunk)

        obs_list = []
        rewards = []
        terminations = []
        truncations = []
        infos_list = []
        # Intermediate substeps reuse the last rendered frame when enabled.
        num_substeps = action_chunk.shape[1]
        render_every_substep = not self.supports_chunk_render_skip
        for chunk_idx in range(num_substeps):
            obs, reward, terminated, truncated, info = self.step(
                action_chunk[:, chunk_idx],
                need_images=(
                    True if render_every_substep else chunk_idx == num_substeps - 1
                ),
            )
            obs_list.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos_list.append(info)
        return self._finish_chunk(
            obs_list, rewards, terminations, truncations, infos_list
        )

    def _can_batch_chunk(self, num_substeps: int) -> bool:
        """Whether the whole chunk can go to the simulator in one call.

        ``GenieSimVecEnv.step`` takes an ``(N, chunk, A)`` chunk and decides by
        itself whether the server can run it in one round trip. It only returns
        the chunk's last frame, so a profile that needs every frame keeps
        stepping substep by substep.
        """
        return (
            num_substeps > 1
            and self.supports_chunk_render_skip
            and not self._save_video
        )

    def _chunk_step_batched(self, action_chunk: np.ndarray):
        """Send one action chunk and reconstruct RLinf's per-step tensors.

        The simulator returns per-substep progress, termination, truncation and
        success arrays, plus the chunk's last frame. Episode resets stay here
        because only this adapter owns the work-item cursor and the reset seed.
        Intermediate frames do not exist, so the last real frame stands in for
        them.
        """
        n, horizon, _ = action_chunk.shape
        raw_obs, c_rew, c_term, c_trunc, venv_info = self._venv.step(
            action_chunk,
            auto_reset=False,
            need_images=self._need_images,
            return_per_step_obs=False,
        )
        if "chunk_success" not in venv_info:
            raise RuntimeError(
                "GenieSim step() returned no 'chunk_success'; per-substep "
                "success is required to credit a chunk. Expected the key from "
                "a chunk-aware client, got keys: "
                f"{sorted(venv_info)[:8]}"
            )
        progress = _as_numpy(c_rew).astype(np.float32).reshape(n, horizon)
        terms = _as_numpy(c_term).astype(bool).reshape(n, horizon)
        truncs = _as_numpy(c_trunc).astype(bool).reshape(n, horizon)
        successes = (
            _as_numpy(venv_info["chunk_success"]).astype(bool).reshape(n, horizon)
        )
        obs = self._adapter.wrap_batch(raw_obs)
        self._last_raw_obs = raw_obs

        rewards_out = []
        terms_out = []
        truncs_out = []
        infos_out = []
        for index in range(horizon):
            info = dict(venv_info) if index == horizon - 1 else {}
            info.pop("chunk_success", None)
            _, reward, term, trunc, info = self._record_step(
                obs,
                progress[:, index],
                successes[:, index],
                terms[:, index],
                truncs[:, index],
                info,
            )
            rewards_out.append(reward)
            terms_out.append(term)
            truncs_out.append(trunc)
            infos_out.append(info)
        return self._finish_chunk(
            [obs] * horizon, rewards_out, terms_out, truncs_out, infos_out
        )

    def update_reset_state_ids(self, *args, **kwargs) -> None:
        del args, kwargs

    def flush_video(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        for resource in (getattr(self, "_venv", None), getattr(self, "_client", None)):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
