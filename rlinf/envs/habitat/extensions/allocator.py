# Copyright 2025 The RLinf Authors.
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

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

EpisodeId = int | str
EpisodeSequences = list[list[list[EpisodeId]]]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: EpisodeId
    scene_id: str
    weight: int


def load_scene_vram_profile(profile_path: str | Path | None = None) -> dict[str, int]:
    if profile_path is None:
        profile_path = (
            Path(__file__).resolve().parent / "config" / "scene_vram_profile.json"
        )
    with Path(profile_path).open("r", encoding="utf-8") as file_obj:
        scene_profile = json.load(file_obj)
    return {
        scene_key: int(scene_info["vram_cost_mb"])
        for scene_key, scene_info in scene_profile["scenes"].items()
    }


def get_assignment_mode(auto_reset: bool) -> str:
    """Map Habitat's auto_reset behavior onto eval/train assignment mode."""
    if auto_reset:
        return "eval"
    return "train"


def vram_balance_episode_ids(
    episodes,
    *,
    auto_reset: bool,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
    seed_offset: int,
    scene_weights: dict[str, int] | None = None,
) -> list[list[EpisodeId]]:
    if scene_weights is None:
        scene_weights = load_scene_vram_profile()
    episode_records = build_episode_records(episodes, scene_weights)
    episode_records = trim_episode_records(
        episode_records,
        mode=get_assignment_mode(auto_reset),
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )
    episode_sequences = assign_episode_sequences(
        episode_records,
        total_num_processes=total_num_processes,
        num_group=num_group,
    )
    return episode_sequences[seed_offset]


def build_episode_records(
    episodes, scene_weights: dict[str, int]
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for episode in episodes:
        scene_key = normalize_scene_key(episode.scene_id)
        records.append(
            EpisodeRecord(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                weight=int(scene_weights[scene_key]),
            )
        )
    return records


def normalize_scene_key(scene_id: str) -> str:
    scene_path = Path(scene_id)
    if scene_path.suffix:
        return scene_path.stem
    return scene_id


def trim_episode_records(
    records: list[EpisodeRecord],
    *,
    mode: str,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
) -> list[EpisodeRecord]:
    total_count = len(records)
    if mode == "eval":
        slot_count = compute_slot_count(
            max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
            max_episode_steps=max_episode_steps,
        )
        target_count = total_num_envs * slot_count
    elif mode == "train":
        granularity = total_num_processes * num_group
        target_count = (len(records) // granularity) * granularity
    else:
        raise ValueError(f"Unknown mode: {mode}")
    dropped_count = total_count - target_count
    logger.info(
        "[Allocator] episode trim mode=%s total=%d kept=%d dropped=%d",
        mode,
        total_count,
        target_count,
        dropped_count,
    )
    return records[:target_count]


def compute_slot_count(
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
) -> int:
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if max_steps_per_rollout_epoch % max_episode_steps != 0:
        raise ValueError(
            "max_steps_per_rollout_epoch must be divisible by max_episode_steps"
        )
    return max_steps_per_rollout_epoch // max_episode_steps


def assign_episode_sequences(
    records: list[EpisodeRecord],
    *,
    total_num_processes: int,
    num_group: int,
) -> EpisodeSequences:
    """Return episode ids as [process_idx][group_idx][slot_idx]."""
    if not records:
        return [[[] for _ in range(num_group)] for _ in range(total_num_processes)]

    total_streams = total_num_processes * num_group
    if len(records) % total_streams != 0:
        raise ValueError(
            "record count must be divisible by total_num_processes * num_group"
        )

    slot_count = len(records) // total_streams
    sequences = [[[] for _ in range(num_group)] for _ in range(total_num_processes)]
    unit_slot_loads = [
        [0 for _ in range(slot_count)] for _ in range(total_num_processes)
    ]
    unit_histories = [0 for _ in range(total_num_processes)]
    ordered_records = sorted(
        records,
        key=lambda record: (
            -record.weight,
            isinstance(record.episode_id, str),
            record.episode_id,
        ),
    )

    for slot_idx in range(slot_count):
        groups_for_slot = []
        for process_idx in range(total_num_processes):
            for group_idx in range(num_group):
                groups_for_slot.append((process_idx, group_idx))

        for record in ordered_records[:]:
            best_choice = None
            best_score = None
            for process_idx, group_idx in groups_for_slot:
                score = (
                    unit_slot_loads[process_idx][slot_idx],
                    unit_histories[process_idx],
                    process_idx,
                    group_idx,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = (process_idx, group_idx)

            process_idx, group_idx = best_choice
            sequences[process_idx][group_idx].append(record.episode_id)
            unit_slot_loads[process_idx][slot_idx] += record.weight
            unit_histories[process_idx] += record.weight
            groups_for_slot.remove((process_idx, group_idx))
            ordered_records.remove(record)
            if not groups_for_slot:
                break

    return sequences
