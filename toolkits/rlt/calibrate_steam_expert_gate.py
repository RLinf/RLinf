#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay compact RLT traces to calibrate the STEAM actor-to-expert gate."""

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class EpisodeTrace:
    """One environment episode extracted from a batched rollout trace."""

    episode_id: str
    chunk_size: int
    score: torch.Tensor
    score_ready: torch.Tensor
    actor_active: torch.Tensor
    oracle_active: torch.Tensor
    reward_sum: torch.Tensor
    versions: torch.Tensor | None


def _parse_number_list(value: str, cast) -> list:
    result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return result


def _as_time_batch(trace: dict[str, Any], key: str, *, dtype) -> torch.Tensor:
    value = trace.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Trace key {key!r} must be a rank-2 tensor")
    return value.to(dtype=dtype)


def _optional_time_batch(
    trace: dict[str, Any],
    key: str,
    *,
    like: torch.Tensor,
    dtype,
) -> torch.Tensor:
    value = trace.get(key)
    if value is None:
        return torch.zeros_like(like, dtype=dtype)
    if not isinstance(value, torch.Tensor) or value.shape != like.shape:
        raise ValueError(
            f"Trace key {key!r} must have shape {tuple(like.shape)}, "
            f"got {getattr(value, 'shape', None)}"
        )
    return value.to(dtype=dtype)


def _episode_ranges(dones: torch.Tensor) -> Iterable[tuple[int, int]]:
    previous_done = False
    start = 0
    for idx, done in enumerate(dones.tolist()):
        current_done = bool(done)
        if current_done and not previous_done:
            yield start, idx + 1
            start = idx + 1
        previous_done = current_done
    if start < int(dones.shape[0]) and not previous_done:
        yield start, int(dones.shape[0])


def load_episodes(trace_dir: Path, *, min_version: int | None) -> list[EpisodeTrace]:
    """Load all trace files and split batched tensors at episode boundaries."""
    episodes: list[EpisodeTrace] = []
    trace_paths = sorted(trace_dir.rglob("trace_*.pt"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace_*.pt files found under {trace_dir}")

    for trace_path in trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        if int(trace.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported trace format in {trace_path}")
        score = _as_time_batch(trace, "rlt_gate_score_min", dtype=torch.float32)
        score_ready = _optional_time_batch(
            trace,
            "rlt_gate_score_ready",
            like=score,
            dtype=torch.bool,
        )
        actor_active = _as_time_batch(
            trace,
            "rlt_gate_actor_active",
            dtype=torch.bool,
        )
        oracle_active = _optional_time_batch(
            trace,
            "rlt_oracle_expert_active",
            like=score,
            dtype=torch.bool,
        )
        oracle_active = oracle_active & actor_active
        rewards = _optional_time_batch(
            trace,
            "reward_sum",
            like=score,
            dtype=torch.float32,
        )
        dones = _optional_time_batch(
            trace,
            "dones",
            like=score,
            dtype=torch.bool,
        )
        versions = trace.get("versions")
        if versions is not None:
            if not isinstance(versions, torch.Tensor) or versions.shape != score.shape:
                raise ValueError(f"Invalid versions tensor in {trace_path}")
            versions = versions.to(torch.long)

        for env_idx in range(score.shape[1]):
            for episode_idx, (start, end) in enumerate(
                _episode_ranges(dones[:, env_idx])
            ):
                episode_versions = (
                    None if versions is None else versions[start:end, env_idx]
                )
                if min_version is not None and episode_versions is None:
                    raise ValueError(
                        f"--min-version was set, but {trace_path} has no versions"
                    )
                if (
                    min_version is not None
                    and episode_versions is not None
                    and int(episode_versions.max().item()) < min_version
                ):
                    continue
                episodes.append(
                    EpisodeTrace(
                        episode_id=(
                            f"{trace_path.parent.name}/{trace_path.stem}:"
                            f"env{env_idx}:episode{episode_idx}"
                        ),
                        chunk_size=int(trace.get("chunk_size", 10)),
                        score=score[start:end, env_idx],
                        score_ready=score_ready[start:end, env_idx],
                        actor_active=actor_active[start:end, env_idx],
                        oracle_active=oracle_active[start:end, env_idx],
                        reward_sum=rewards[start:end, env_idx],
                        versions=episode_versions,
                    )
                )
    if not episodes:
        raise RuntimeError("No episodes remained after trace filtering")
    return episodes


def replay_expert_gate(
    episode: EpisodeTrace,
    *,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
) -> tuple[int | None, torch.Tensor]:
    """Replay the deployed consecutive-low-score gate on one episode."""
    active = torch.zeros_like(episode.actor_active)
    low_progress_count = 0
    critical_chunk_count = -1
    entered_at = None
    latched = False

    for idx in range(int(episode.score.shape[0])):
        if not bool(episode.actor_active[idx]):
            critical_chunk_count = -1
            low_progress_count = 0
            continue
        critical_chunk_count += 1
        eligible = (
            bool(episode.score_ready[idx])
            and critical_chunk_count >= warmup_chunks
        )
        low_progress = eligible and float(episode.score[idx]) <= threshold
        low_progress_count = low_progress_count + 1 if low_progress else 0
        if not latched and low_progress_count >= patience_chunks:
            latched = True
            entered_at = idx
        active[idx] = latched
    return entered_at, active


def _first_true(value: torch.Tensor) -> int | None:
    indices = torch.nonzero(value, as_tuple=False).reshape(-1)
    return None if indices.numel() == 0 else int(indices[0].item())


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def evaluate_parameters(
    episodes: list[EpisodeTrace],
    *,
    split: str,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
) -> dict[str, Any]:
    oracle_positive = 0
    predicted_on_oracle = 0
    oracle_negative = 0
    false_takeovers = 0
    predicted_episodes = 0
    successful_episodes = 0
    critical_chunks = 0
    expert_chunks = 0
    entry_delays: list[float] = []
    actor_opportunities: list[float] = []

    for episode in episodes:
        predicted_entry, predicted_active = replay_expert_gate(
            episode,
            threshold=threshold,
            patience_chunks=patience_chunks,
            warmup_chunks=warmup_chunks,
        )
        oracle_entry = _first_true(episode.oracle_active)
        critical_entry = _first_true(episode.actor_active)
        has_oracle = oracle_entry is not None
        has_prediction = predicted_entry is not None

        oracle_positive += int(has_oracle)
        oracle_negative += int(not has_oracle)
        predicted_on_oracle += int(has_oracle and has_prediction)
        false_takeovers += int((not has_oracle) and has_prediction)
        predicted_episodes += int(has_prediction)
        successful_episodes += int(bool((episode.reward_sum > 0).any()))
        critical_chunks += int(episode.actor_active.sum().item())
        expert_chunks += int((predicted_active & episode.actor_active).sum().item())
        if has_oracle and has_prediction:
            entry_delays.append(
                float(predicted_entry - oracle_entry) * episode.chunk_size
            )
        if has_prediction and critical_entry is not None:
            actor_opportunities.append(
                float(predicted_entry - critical_entry) * episode.chunk_size
            )

    median_delay = (
        statistics.median(entry_delays) if entry_delays else math.nan
    )
    median_opportunity = (
        statistics.median(actor_opportunities)
        if actor_opportunities
        else math.nan
    )
    return {
        "split": split,
        "threshold": threshold,
        "patience_chunks": patience_chunks,
        "warmup_chunks": warmup_chunks,
        "num_episodes": len(episodes),
        "oracle_positive_episodes": oracle_positive,
        "oracle_stall_recall": _safe_rate(predicted_on_oracle, oracle_positive),
        "false_takeover_episode_rate": _safe_rate(
            false_takeovers,
            oracle_negative,
        ),
        "predicted_takeover_episode_rate": _safe_rate(
            predicted_episodes,
            len(episodes),
        ),
        "expert_fraction_given_critical": _safe_rate(
            expert_chunks,
            critical_chunks,
        ),
        "median_entry_delay_steps": median_delay,
        "median_actor_opportunity_steps": median_opportunity,
        "success_episode_rate": _safe_rate(successful_episodes, len(episodes)),
    }


def _split_episodes(
    episodes: list[EpisodeTrace],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[EpisodeTrace], list[EpisodeTrace]]:
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2 or validation_fraction <= 0:
        return shuffled, []
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def _rank_key(row: dict[str, Any]) -> tuple:
    recall = row["oracle_stall_recall"]
    false_rate = row["false_takeover_episode_rate"]
    delay = row["median_entry_delay_steps"]
    expert_fraction = row["expert_fraction_given_critical"]
    feasible = (
        not math.isnan(recall)
        and not math.isnan(false_rate)
        and recall >= 0.85
        and false_rate <= 0.05
        and (math.isnan(delay) or abs(delay) <= 50.0)
        and (math.isnan(expert_fraction) or expert_fraction <= 0.30)
    )
    return (
        not feasible,
        false_rate if not math.isnan(false_rate) else math.inf,
        -recall if not math.isnan(recall) else math.inf,
        abs(delay) if not math.isnan(delay) else math.inf,
        expert_fraction if not math.isnan(expert_fraction) else math.inf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the RLT STEAM actor-to-expert gate from shadow traces."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--thresholds",
        default="0,-0.05,-0.10,-0.15,-0.20,-0.25,-0.30,-0.35,-0.40,-0.45,-0.50",
    )
    parser.add_argument("--patience-chunks", default="3,4,5,6")
    parser.add_argument("--warmup-chunks", default="4,6,8")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-version", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    thresholds = _parse_number_list(args.thresholds, float)
    patience_values = _parse_number_list(args.patience_chunks, int)
    warmup_values = _parse_number_list(args.warmup_chunks, int)
    if min(patience_values) < 1 or min(warmup_values) < 0:
        parser.error("patience must be positive and warmup must be non-negative")

    episodes = load_episodes(args.trace_dir, min_version=args.min_version)
    calibration, validation = _split_episodes(
        episodes,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    rows = []
    for threshold, patience, warmup in product(
        thresholds,
        patience_values,
        warmup_values,
    ):
        rows.append(
            evaluate_parameters(
                calibration,
                split="calibration",
                threshold=threshold,
                patience_chunks=patience,
                warmup_chunks=warmup,
            )
        )
        if validation:
            rows.append(
                evaluate_parameters(
                    validation,
                    split="validation",
                    threshold=threshold,
                    patience_chunks=patience,
                    warmup_chunks=warmup,
                )
            )

    output = args.output or (args.trace_dir / "steam_expert_gate_grid.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranking_split = "validation" if validation else "calibration"
    ranked = sorted(
        (row for row in rows if row["split"] == ranking_split),
        key=_rank_key,
    )
    print(
        f"Loaded {len(episodes)} episodes: {len(calibration)} calibration, "
        f"{len(validation)} validation"
    )
    print(f"Wrote {len(rows)} grid rows to {output}")
    print(f"Top {min(10, len(ranked))} settings on {ranking_split}:")
    for row in ranked[:10]:
        print(
            "  threshold={threshold:.3f} patience={patience_chunks} "
            "warmup={warmup_chunks} recall={oracle_stall_recall:.3f} "
            "false={false_takeover_episode_rate:.3f} "
            "delay={median_entry_delay_steps:.1f} "
            "expert|critical={expert_fraction_given_critical:.3f}".format(**row)
        )


if __name__ == "__main__":
    main()
