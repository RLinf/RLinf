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

import copy

import torch


def clone_obs(obs):
    cloned = {}
    for key, value in obs.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def merge_obs_rows(base_obs, update_obs, env_indices):
    merged_obs = clone_obs(base_obs)
    for key, update_value in update_obs.items():
        if key not in merged_obs:
            merged_obs[key] = update_value
            continue
        base_value = merged_obs[key]
        if torch.is_tensor(base_value):
            index = torch.as_tensor(
                env_indices, device=base_value.device, dtype=torch.long
            )
            merged_obs[key][index] = update_value.to(base_value.device)
        elif isinstance(base_value, list):
            for local_idx, env_idx in enumerate(env_indices):
                base_value[env_idx] = update_value[local_idx]
        else:
            merged_obs[key] = update_value
    return merged_obs


def merge_info_rows(base_infos, update_infos, env_indices, num_envs: int):
    merged_infos = copy.deepcopy(base_infos)

    def merge_value(base_value, update_value):
        if isinstance(update_value, dict):
            if not isinstance(base_value, dict):
                base_value = {}
            for key, child_update in update_value.items():
                base_value[key] = merge_value(base_value.get(key), child_update)
            return base_value

        if torch.is_tensor(update_value):
            if not torch.is_tensor(base_value):
                base_shape = (num_envs, *update_value.shape[1:])
                base_value = torch.zeros(
                    base_shape, dtype=update_value.dtype, device=update_value.device
                )
            index = torch.as_tensor(
                env_indices, device=base_value.device, dtype=torch.long
            )
            base_value[index] = update_value.to(base_value.device)
            return base_value

        return update_value

    for key, value in update_infos.items():
        merged_infos[key] = merge_value(merged_infos.get(key), value)
    return merged_infos


def reset_payload_with_instance_ids(
    reset_mask: list[bool],
    instance_ids: list[int] | None,
    full_reset: bool = False,
):
    if instance_ids is None:
        return reset_mask

    instance_iter = iter(instance_ids)
    payload = []
    for should_reset in reset_mask:
        item = {"reset": bool(should_reset), "full_reset": full_reset}
        if should_reset:
            item["instance_id"] = next(instance_iter)
        payload.append(item)
    return payload


def parse_reset_payload(payload):
    """Parse a reset payload into (reset_indices, instance_ids, is_full_reset).

    Supports three formats:
    - ``None``: reset all envs.
    - ``list[bool]``: each element indicates whether to reset that env.
    - ``list[dict]``: each dict may contain ``reset`` (bool), ``full_reset``
      (bool), and ``instance_id`` (int) keys.

    Args:
        payload: Reset specification in one of the supported formats.

    Returns:
        Tuple of (reset_indices, instance_ids, is_full_reset). ``instance_ids``
        is None when not provided.

    Raises:
        ValueError: If instance_ids are provided for some but not all reset envs.
    """
    if payload is None:
        return None, None, False

    if payload and all(isinstance(item, dict) for item in payload):
        reset_indices = [
            idx for idx, item in enumerate(payload) if bool(item.get("reset", True))
        ]
        is_full_reset = all(bool(item.get("full_reset", False)) for item in payload)
        instance_ids = [
            int(payload[idx]["instance_id"])
            for idx in reset_indices
            if payload[idx].get("instance_id") is not None
        ]
        if instance_ids and len(instance_ids) != len(reset_indices):
            raise ValueError(
                "Reset payload must provide instance_id for every reset env "
                "or for none of them."
            )
        return reset_indices, instance_ids or None, is_full_reset

    reset_indices = [idx for idx, flag in enumerate(payload) if bool(flag)]
    return reset_indices, None, False


def reset_payload_item_enabled(item) -> bool:
    if isinstance(item, dict):
        return bool(item.get("reset", True))
    return bool(item)


def build_reset_payload_shards(
    payload: list,
    plan: list[tuple[int, list[int], list[int]]],
    num_env_shard: int,
    num_env_subprocess: int,
) -> tuple[list[list], list[list[int]]]:
    """Expand slice-order reset payload into local shard-order payloads."""
    payload_is_dict = bool(payload) and all(isinstance(item, dict) for item in payload)
    filler = {"reset": False} if payload_is_dict else False
    payload_shards = [
        [copy.deepcopy(filler) for _ in range(num_env_shard)]
        for _ in range(num_env_subprocess)
    ]
    reset_positions_by_proc = [[] for _ in range(num_env_subprocess)]

    for sp, positions, local_rows in plan:
        for pos, local_row in zip(positions, local_rows):
            item = payload[pos]
            payload_shards[sp][local_row] = item
            if reset_payload_item_enabled(item):
                reset_positions_by_proc[sp].append(pos)

    return payload_shards, reset_positions_by_proc
