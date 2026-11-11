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

"""Build GenieSim work items from server task metadata and the reset vectors they produce."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np


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
