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
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_MB_PER_GB = 1024.0

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
    total_episode_count = len(episode_records)
    episode_records = trim_episode_records(
        episode_records,
        mode=get_assignment_mode(auto_reset),
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )
    dropped_episode_count = total_episode_count - len(episode_records)
    episode_sequences = assign_episode_sequences(
        episode_records,
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        dropped_episode_count=dropped_episode_count,
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
    if mode == "eval":
        assert max_steps_per_rollout_epoch % max_episode_steps == 0, (
            "max_steps_per_rollout_epoch must be divisible by max_episode_steps"
        )
        slot_count = int(max_steps_per_rollout_epoch / max_episode_steps)
        target_count = total_num_envs * slot_count
    elif mode == "train":
        granularity = total_num_processes * num_group
        target_count = (len(records) // granularity) * granularity
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return records[:target_count]


def _build_peak_first_score(
    process_slot_loads: list[list[int]],
    process_total_loads: list[int],
    process_peak_loads: list[int],
    *,
    process_idx: int,
    slot_idx: int,
    record_weight: int,
) -> tuple[int, int, int, int, int]:
    """Score order before cell tie-break append: peak, slot gap, row range, process total, process tie-break; caller then appends group and slot."""
    projected_slot_load = process_slot_loads[process_idx][slot_idx] + record_weight
    projected_process_peak = max(process_peak_loads[process_idx], projected_slot_load)

    peak_after = projected_process_peak
    for other_process_idx, other_peak in enumerate(process_peak_loads):
        if other_process_idx == process_idx:
            continue
        peak_after = max(peak_after, other_peak)

    slot_min_after = None
    slot_max_after = None
    for other_process_idx, slot_loads in enumerate(process_slot_loads):
        slot_value = slot_loads[slot_idx]
        if other_process_idx == process_idx:
            slot_value = projected_slot_load
        if slot_min_after is None or slot_value < slot_min_after:
            slot_min_after = slot_value
        if slot_max_after is None or slot_value > slot_max_after:
            slot_max_after = slot_value

    process_min_after = None
    process_max_after = None
    for other_slot_idx, slot_value in enumerate(process_slot_loads[process_idx]):
        if other_slot_idx == slot_idx:
            slot_value = projected_slot_load
        if process_min_after is None or slot_value < process_min_after:
            process_min_after = slot_value
        if process_max_after is None or slot_value > process_max_after:
            process_max_after = slot_value

    return (
        peak_after,
        slot_max_after - slot_min_after,
        process_max_after - process_min_after,
        process_total_loads[process_idx] + record_weight,
        process_idx,
    )


def _assign_peak_first_greedy(
    records: list[EpisodeRecord],
    *,
    total_num_processes: int,
    num_group: int,
    slot_count: int,
) -> EpisodeSequences:
    placements = [
        [[None for _ in range(slot_count)] for _ in range(num_group)]
        for _ in range(total_num_processes)
    ]
    open_cells = [
        (process_idx, group_idx, slot_idx)
        for slot_idx in range(slot_count)
        for process_idx in range(total_num_processes)
        for group_idx in range(num_group)
    ]
    process_slot_loads = [
        [0 for _ in range(slot_count)] for _ in range(total_num_processes)
    ]
    process_total_loads = [0 for _ in range(total_num_processes)]
    process_peak_loads = [0 for _ in range(total_num_processes)]
    ordered_records = sorted(
        records,
        key=lambda record: (
            -record.weight,
            isinstance(record.episode_id, str),
            record.episode_id,
        ),
    )

    for record in ordered_records:
        best_choice = None
        best_score = None
        for process_idx, group_idx, slot_idx in open_cells:
            score = _build_peak_first_score(
                process_slot_loads,
                process_total_loads,
                process_peak_loads,
                process_idx=process_idx,
                slot_idx=slot_idx,
                record_weight=record.weight,
            ) + (group_idx, slot_idx)
            if best_score is None or score < best_score:
                best_score = score
                best_choice = (process_idx, group_idx, slot_idx)

        process_idx, group_idx, slot_idx = best_choice
        placements[process_idx][group_idx][slot_idx] = record.episode_id
        process_slot_loads[process_idx][slot_idx] += record.weight
        process_total_loads[process_idx] += record.weight
        process_peak_loads[process_idx] = max(
            process_peak_loads[process_idx],
            process_slot_loads[process_idx][slot_idx],
        )
        open_cells.remove(best_choice)

    return [
        [list(group_slots) for group_slots in process_slots]
        for process_slots in placements
    ]


def _compute_process_slot_loads(
    sequences: EpisodeSequences,
    episode_weight_map: dict[EpisodeId, int],
) -> list[list[int]]:
    return [
        _compute_slot_vram_series(process_group_episode_ids, episode_weight_map)
        for process_group_episode_ids in sequences
    ]


def _peak_for_process_slot_loads(process_slot_loads: list[list[int]]) -> int:
    peak_load = 0
    for process_loads in process_slot_loads:
        if process_loads:
            peak_load = max(peak_load, max(process_loads))
    return peak_load


def _swap_process_slot_loads(
    process_slot_loads: list[list[int]],
    *,
    left_cell: tuple[int, int, int],
    right_cell: tuple[int, int, int],
    left_weight: int,
    right_weight: int,
) -> list[list[int]]:
    swapped_loads = [list(process_loads) for process_loads in process_slot_loads]
    left_process_idx, _, left_slot_idx = left_cell
    right_process_idx, _, right_slot_idx = right_cell
    swapped_loads[left_process_idx][left_slot_idx] += right_weight - left_weight
    swapped_loads[right_process_idx][right_slot_idx] += left_weight - right_weight
    return swapped_loads


def _swap_sequence_cells(
    sequences: EpisodeSequences,
    *,
    left_cell: tuple[int, int, int],
    right_cell: tuple[int, int, int],
) -> EpisodeSequences:
    swapped_sequences = [
        [list(group_slots) for group_slots in process_slots]
        for process_slots in sequences
    ]
    left_process_idx, left_group_idx, left_slot_idx = left_cell
    right_process_idx, right_group_idx, right_slot_idx = right_cell
    left_episode_id = swapped_sequences[left_process_idx][left_group_idx][left_slot_idx]
    swapped_sequences[left_process_idx][left_group_idx][left_slot_idx] = (
        swapped_sequences[right_process_idx][right_group_idx][right_slot_idx]
    )
    swapped_sequences[right_process_idx][right_group_idx][right_slot_idx] = (
        left_episode_id
    )
    return swapped_sequences


def _find_best_bottleneck_swap(
    sequences: EpisodeSequences,
    process_slot_loads: list[list[int]],
    episode_weight_map: dict[EpisodeId, int],
    *,
    source_process_idx: int,
    source_slot_idx: int,
    current_peak: int,
    bottleneck_positions: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int], int, int] | None:
    best_swap = None
    best_score = None

    for source_group_idx, source_group_slots in enumerate(sequences[source_process_idx]):
        source_episode_id = source_group_slots[source_slot_idx]
        source_weight = episode_weight_map[source_episode_id]

        for target_slot_idx in range(len(process_slot_loads[source_process_idx])):
            if target_slot_idx == source_slot_idx:
                continue
            if (source_process_idx, target_slot_idx) in bottleneck_positions:
                continue
            if process_slot_loads[source_process_idx][target_slot_idx] >= current_peak:
                continue

            for target_group_idx, target_group_slot_ids in enumerate(
                sequences[source_process_idx]
            ):
                target_cell = (source_process_idx, target_group_idx, target_slot_idx)
                if target_cell in used_target_cells:
                    continue

                target_episode_id = target_group_slot_ids[target_slot_idx]
                target_weight = episode_weight_map[target_episode_id]
                if source_weight <= target_weight:
                    continue

                source_load_after = (
                    process_slot_loads[source_process_idx][source_slot_idx]
                    - source_weight
                    + target_weight
                )
                target_load_after = (
                    process_slot_loads[source_process_idx][target_slot_idx]
                    - target_weight
                    + source_weight
                )
                if source_load_after >= current_peak or target_load_after >= current_peak:
                    continue

                candidate_score = (
                    source_load_after,
                    target_load_after,
                    target_slot_idx,
                    target_group_idx,
                    source_group_idx,
                )
                if best_score is None or candidate_score < best_score:
                    best_score = candidate_score
                    best_swap = (
                        (source_process_idx, source_group_idx, source_slot_idx),
                        target_cell,
                        source_weight,
                        target_weight,
                    )

    return best_swap


def _assign_bottleneck_local_search(
    records: list[EpisodeRecord],
    *,
    total_num_processes: int,
    num_group: int,
    slot_count: int,
) -> EpisodeSequences:
    episode_weight_map = {record.episode_id: record.weight for record in records}
    sequences = _assign_peak_first_greedy(
        records,
        total_num_processes=total_num_processes,
        num_group=num_group,
        slot_count=slot_count,
    )
    while True:
        process_slot_loads = _compute_process_slot_loads(sequences, episode_weight_map)
        current_peak = _peak_for_process_slot_loads(process_slot_loads)
        if current_peak == 0:
            break

        bottleneck_positions = {
            (process_idx, slot_idx)
            for process_idx, process_loads in enumerate(process_slot_loads)
            for slot_idx, slot_load in enumerate(process_loads)
            if slot_load == current_peak
        }
        candidate_sequences = [
            [list(group_slots) for group_slots in process_slots]
            for process_slots in sequences
        ]
        candidate_loads = [list(process_loads) for process_loads in process_slot_loads]
        used_target_cells: set[tuple[int, int, int]] = set()
        repair_succeeds = True

        for source_process_idx, source_slot_idx in sorted(bottleneck_positions):
            best_swap = _find_best_bottleneck_swap(
                candidate_sequences,
                candidate_loads,
                episode_weight_map,
                source_process_idx=source_process_idx,
                source_slot_idx=source_slot_idx,
                current_peak=current_peak,
                bottleneck_positions=bottleneck_positions,
                used_target_cells=used_target_cells,
            )
            if best_swap is None:
                repair_succeeds = False
                break

            source_cell, target_cell, source_weight, target_weight = best_swap
            candidate_sequences = _swap_sequence_cells(
                candidate_sequences,
                left_cell=source_cell,
                right_cell=target_cell,
            )
            candidate_loads = _swap_process_slot_loads(
                candidate_loads,
                left_cell=source_cell,
                right_cell=target_cell,
                left_weight=source_weight,
                right_weight=target_weight,
            )
            used_target_cells.add(target_cell)

        if not repair_succeeds or _peak_for_process_slot_loads(candidate_loads) >= current_peak:
            break

        sequences = candidate_sequences

    return sequences


def assign_episode_sequences(
    records: list[EpisodeRecord],
    *,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int | None = None,
    dropped_episode_count: int = 0,
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
    sequences = _assign_bottleneck_local_search(
        records,
        total_num_processes=total_num_processes,
        num_group=num_group,
        slot_count=slot_count,
    )

    logger.info(
        "%s",
        render_assignment_vram_summary(
            records=records,
            sequences=sequences,
            total_num_processes=total_num_processes,
            num_group=num_group,
            total_num_envs=total_num_envs,
            dropped_episode_count=dropped_episode_count,
        ),
    )
    return sequences


def _compute_slot_vram_series(
    process_group_episode_ids: list[list[EpisodeId]],
    episode_weight_map: dict[EpisodeId, int],
) -> list[int]:
    if not process_group_episode_ids:
        return []
    slot_count = len(process_group_episode_ids[0])
    slot_series: list[int] = []
    for slot_idx in range(slot_count):
        slot_vram = 0
        for group_episode_ids in process_group_episode_ids:
            slot_vram += episode_weight_map[group_episode_ids[slot_idx]]
        slot_series.append(slot_vram)
    return slot_series


def _summarize_slot_series(slot_vram_mb: list[int]) -> dict[str, float]:
    if not slot_vram_mb:
        return {"max_gb": 0.0, "min_gb": 0.0, "mean_gb": 0.0, "std_gb": 0.0}
    max_gb = float(max(slot_vram_mb) / _MB_PER_GB)
    min_gb = float(min(slot_vram_mb) / _MB_PER_GB)
    mean_gb = float(sum(slot_vram_mb) / len(slot_vram_mb) / _MB_PER_GB)
    variance = sum((value / _MB_PER_GB - mean_gb) ** 2 for value in slot_vram_mb) / len(
        slot_vram_mb
    )
    return {
        "max_gb": max_gb,
        "min_gb": min_gb,
        "mean_gb": mean_gb,
        "std_gb": float(math.sqrt(variance)),
    }


def render_assignment_vram_summary(
    *,
    records: list[EpisodeRecord],
    sequences: EpisodeSequences,
    total_num_processes: int,
    num_group: int | None = None,
    total_num_envs: int | None = None,
    dropped_episode_count: int = 0,
) -> str:

    denom = total_num_processes * num_group
    assert total_num_envs % denom == 0, (
        "total_num_envs must be divisible by total_num_processes * num_group"
    )
    group_size = total_num_envs // denom

    episode_weight_map = {record.episode_id: int(record.weight) for record in records}

    per_gpu: dict[str, dict] = {}
    for process_idx in range(total_num_processes):
        slot_series = _compute_slot_vram_series(
            sequences[process_idx], episode_weight_map
        )
        slot_series = [value * group_size for value in slot_series]
        per_gpu[f"gpu_{process_idx}"] = {
            "slot_vram_gb": [value / _MB_PER_GB for value in slot_series],
            **_summarize_slot_series(slot_series),
        }

    total_episode_count = len(records) + dropped_episode_count
    lines = [
        (
            "Episodes: total={total}, kept={kept}, dropped={dropped}; per-gpu stats:".format(
                total=total_episode_count,
                kept=len(records),
                dropped=dropped_episode_count,
            )
        )
    ]
    for gpu_name, gpu_data in per_gpu.items():
        lines.append(
            (
                "  {gpu}: max={max_gb:.2f} GB, min={min_gb:.2f} GB, "
                "mean={mean_gb:.2f} GB, std={std_gb:.2f} GB, slots={slot_len}"
            ).format(
                gpu=gpu_name,
                max_gb=gpu_data["max_gb"],
                min_gb=gpu_data["min_gb"],
                mean_gb=gpu_data["mean_gb"],
                std_gb=gpu_data["std_gb"],
                slot_len=len(gpu_data["slot_vram_gb"]),
            )
        )
    return "\n".join(lines)
