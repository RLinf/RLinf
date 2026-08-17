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

"""Replay shadow traces to calibrate the RLT STEAM expert-entry gate."""

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
    success_signal: torch.Tensor
    success_label_source: str
    versions: torch.Tensor | None
    complete: bool


@dataclass(frozen=True)
class EpisodeOutcome:
    """Gate outcome and post-signal behavior for one actor-active episode."""

    has_oracle: bool
    has_prediction: bool
    success: bool
    critical_chunks: int
    expert_chunks: int
    entry_delay_steps: float | None
    actor_opportunity_steps: float | None
    prediction_to_success_steps: float | None


_BOOTSTRAP_METRICS = (
    "oracle_stall_recall",
    "geometry_disagreement_episode_rate",
    "predicted_takeover_episode_rate",
    "takeover_rate_on_success",
    "takeover_rate_on_failure",
    "autonomous_success_after_prediction_rate",
    "autonomous_success_within_horizon_rate",
    "expert_fraction_given_critical",
)


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


def _success_time_batch(
    trace: dict[str, Any],
    *,
    like: torch.Tensor,
    trace_path: Path,
) -> tuple[torch.Tensor, str]:
    # ManiSkill RLT peg insertion maps task success to termination. Reward is a
    # compatibility fallback for older or externally generated trace formats.
    if trace.get("terminations") is not None:
        terminations = _as_time_batch(trace, "terminations", dtype=torch.bool)
        if terminations.shape != like.shape:
            raise ValueError(
                f"Trace key 'terminations' in {trace_path} must have shape "
                f"{tuple(like.shape)}, got {tuple(terminations.shape)}"
            )
        return terminations, "terminations"

    if trace.get("reward_sum") is not None:
        rewards = _as_time_batch(trace, "reward_sum", dtype=torch.float32)
        if rewards.shape != like.shape:
            raise ValueError(
                f"Trace key 'reward_sum' in {trace_path} must have shape "
                f"{tuple(like.shape)}, got {tuple(rewards.shape)}"
            )
        return rewards > 0, "reward_sum"

    raise ValueError(
        f"Trace {trace_path} has neither 'terminations' nor 'reward_sum'; "
        "success labels cannot be reconstructed"
    )


def _episode_ranges(dones: torch.Tensor) -> Iterable[tuple[int, int, bool]]:
    previous_done = False
    start = 0
    for idx, done in enumerate(dones.tolist()):
        current_done = bool(done)
        if current_done and not previous_done:
            yield start, idx + 1, True
            start = idx + 1
        previous_done = current_done
    if start < int(dones.shape[0]) and not previous_done:
        yield start, int(dones.shape[0]), False


def _episode_version(episode: EpisodeTrace) -> int | None:
    if episode.versions is None or episode.versions.numel() == 0:
        return None
    return int(episode.versions.max().item())


def load_episodes(
    trace_dir: Path,
    *,
    min_version: int | None,
    max_version: int | None,
) -> list[EpisodeTrace]:
    """Load trace files and split their batched tensors into episodes."""
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
        success_signal, success_label_source = _success_time_batch(
            trace,
            like=score,
            trace_path=trace_path,
        )
        dones = _as_time_batch(trace, "dones", dtype=torch.bool)
        if dones.shape != score.shape:
            raise ValueError(
                f"Trace key 'dones' in {trace_path} must have shape "
                f"{tuple(score.shape)}, got {tuple(dones.shape)}"
            )
        versions = trace.get("versions")
        if versions is not None:
            if not isinstance(versions, torch.Tensor) or versions.shape != score.shape:
                raise ValueError(f"Invalid versions tensor in {trace_path}")
            versions = versions.to(torch.long)

        for env_idx in range(score.shape[1]):
            ranges = _episode_ranges(dones[:, env_idx])
            for episode_idx, (start, end, complete) in enumerate(ranges):
                episode_versions = (
                    None if versions is None else versions[start:end, env_idx]
                )
                if (
                    min_version is not None or max_version is not None
                ) and episode_versions is None:
                    raise ValueError(
                        f"A version filter was set, but {trace_path} has no versions"
                    )
                if episode_versions is not None:
                    episode_min = int(episode_versions.min().item())
                    episode_max = int(episode_versions.max().item())
                    if min_version is not None and episode_max < min_version:
                        continue
                    if max_version is not None and episode_min > max_version:
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
                        success_signal=success_signal[start:end, env_idx],
                        success_label_source=success_label_source,
                        versions=episode_versions,
                        complete=complete,
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
            bool(episode.score_ready[idx]) and critical_chunk_count >= warmup_chunks
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


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def _evaluate_episode(
    episode: EpisodeTrace,
    *,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
) -> EpisodeOutcome:
    predicted_entry, predicted_active = replay_expert_gate(
        episode,
        threshold=threshold,
        patience_chunks=patience_chunks,
        warmup_chunks=warmup_chunks,
    )
    oracle_entry = _first_true(episode.oracle_active)
    critical_entry = _first_true(episode.actor_active)
    success_entry = _first_true(episode.success_signal)

    entry_delay = None
    if oracle_entry is not None and predicted_entry is not None:
        entry_delay = float(predicted_entry - oracle_entry) * episode.chunk_size

    actor_opportunity = None
    if predicted_entry is not None and critical_entry is not None:
        actor_opportunity = float(predicted_entry - critical_entry) * episode.chunk_size

    prediction_to_success = None
    if (
        predicted_entry is not None
        and success_entry is not None
        and success_entry >= predicted_entry
    ):
        prediction_to_success = (
            float(success_entry - predicted_entry) * episode.chunk_size
        )

    return EpisodeOutcome(
        has_oracle=oracle_entry is not None,
        has_prediction=predicted_entry is not None,
        success=success_entry is not None,
        critical_chunks=int(episode.actor_active.sum().item()),
        expert_chunks=int((predicted_active & episode.actor_active).sum().item()),
        entry_delay_steps=entry_delay,
        actor_opportunity_steps=actor_opportunity,
        prediction_to_success_steps=prediction_to_success,
    )


def _aggregate_outcomes(
    outcomes: list[EpisodeOutcome],
    *,
    success_horizon_steps: int,
) -> dict[str, Any]:
    num_episodes = len(outcomes)
    oracle_positive = sum(outcome.has_oracle for outcome in outcomes)
    oracle_negative = num_episodes - oracle_positive
    predicted = sum(outcome.has_prediction for outcome in outcomes)
    predicted_on_oracle = sum(
        outcome.has_prediction and outcome.has_oracle for outcome in outcomes
    )
    geometry_disagreements = sum(
        outcome.has_prediction and not outcome.has_oracle for outcome in outcomes
    )
    successful = sum(outcome.success for outcome in outcomes)
    failed = num_episodes - successful
    predicted_on_success = sum(
        outcome.has_prediction and outcome.success for outcome in outcomes
    )
    predicted_on_failure = sum(
        outcome.has_prediction and not outcome.success for outcome in outcomes
    )
    success_after_prediction = sum(
        outcome.prediction_to_success_steps is not None for outcome in outcomes
    )
    success_within_horizon = sum(
        outcome.prediction_to_success_steps is not None
        and outcome.prediction_to_success_steps <= success_horizon_steps
        for outcome in outcomes
    )
    critical_chunks = sum(outcome.critical_chunks for outcome in outcomes)
    expert_chunks = sum(outcome.expert_chunks for outcome in outcomes)
    entry_delays = [
        outcome.entry_delay_steps
        for outcome in outcomes
        if outcome.entry_delay_steps is not None
    ]
    actor_opportunities = [
        outcome.actor_opportunity_steps
        for outcome in outcomes
        if outcome.actor_opportunity_steps is not None
    ]
    prediction_to_success = [
        outcome.prediction_to_success_steps
        for outcome in outcomes
        if outcome.prediction_to_success_steps is not None
    ]

    return {
        "num_episodes": num_episodes,
        "successful_episodes": successful,
        "failed_episodes": failed,
        "oracle_positive_episodes": oracle_positive,
        "oracle_negative_episodes": oracle_negative,
        "predicted_takeover_episodes": predicted,
        "oracle_stall_recall": _safe_rate(
            predicted_on_oracle,
            oracle_positive,
        ),
        "geometry_disagreement_episode_rate": _safe_rate(
            geometry_disagreements,
            oracle_negative,
        ),
        "predicted_takeover_episode_rate": _safe_rate(
            predicted,
            num_episodes,
        ),
        "takeover_rate_on_success": _safe_rate(
            predicted_on_success,
            successful,
        ),
        "takeover_rate_on_failure": _safe_rate(
            predicted_on_failure,
            failed,
        ),
        "autonomous_success_after_prediction_rate": _safe_rate(
            success_after_prediction,
            predicted,
        ),
        "autonomous_success_within_horizon_rate": _safe_rate(
            success_within_horizon,
            predicted,
        ),
        "expert_fraction_given_critical": _safe_rate(
            expert_chunks,
            critical_chunks,
        ),
        "median_entry_delay_steps": _median(entry_delays),
        "median_actor_opportunity_steps": _median(actor_opportunities),
        "median_prediction_to_success_steps": _median(prediction_to_success),
        "success_episode_rate": _safe_rate(successful, num_episodes),
        "success_horizon_steps": success_horizon_steps,
    }


def _evaluate_outcomes(
    episodes: list[EpisodeTrace],
    *,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
) -> list[EpisodeOutcome]:
    return [
        _evaluate_episode(
            episode,
            threshold=threshold,
            patience_chunks=patience_chunks,
            warmup_chunks=warmup_chunks,
        )
        for episode in episodes
    ]


def evaluate_parameters(
    episodes: list[EpisodeTrace],
    *,
    split: str,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
    success_horizon_steps: int,
) -> dict[str, Any]:
    """Evaluate one deployable STEAM gate configuration."""
    outcomes = _evaluate_outcomes(
        episodes,
        threshold=threshold,
        patience_chunks=patience_chunks,
        warmup_chunks=warmup_chunks,
    )
    return {
        "split": split,
        "threshold": threshold,
        "patience_chunks": patience_chunks,
        "warmup_chunks": warmup_chunks,
        "success_label_source": "+".join(
            sorted({episode.success_label_source for episode in episodes})
        ),
        **_aggregate_outcomes(
            outcomes,
            success_horizon_steps=success_horizon_steps,
        ),
        "pareto_front": False,
    }


def _split_episodes_chronologically(
    episodes: list[EpisodeTrace],
    *,
    validation_fraction: float,
) -> tuple[list[EpisodeTrace], list[EpisodeTrace], str]:
    if len(episodes) < 2 or validation_fraction <= 0:
        return list(episodes), [], "disabled"

    validation_size = max(1, round(len(episodes) * validation_fraction))
    validation_size = min(validation_size, len(episodes) - 1)
    versions = [_episode_version(episode) for episode in episodes]
    if all(version is not None for version in versions):
        by_version: dict[int, list[EpisodeTrace]] = {}
        for episode, version in zip(episodes, versions, strict=True):
            assert version is not None
            by_version.setdefault(version, []).append(episode)
        ordered_versions = sorted(by_version)
        if len(ordered_versions) > 1:
            validation_versions: set[int] = set()
            validation_count = 0
            for version in reversed(ordered_versions[1:]):
                validation_versions.add(version)
                validation_count += len(by_version[version])
                if validation_count >= validation_size:
                    break
            calibration = [
                episode
                for episode in episodes
                if _episode_version(episode) not in validation_versions
            ]
            validation = [
                episode
                for episode in episodes
                if _episode_version(episode) in validation_versions
            ]
            calibration.sort(
                key=lambda episode: (_episode_version(episode), episode.episode_id)
            )
            validation.sort(
                key=lambda episode: (_episode_version(episode), episode.episode_id)
            )
            return calibration, validation, "model_version"

    ordered = sorted(episodes, key=lambda episode: episode.episode_id)
    return (
        ordered[:-validation_size],
        ordered[-validation_size:],
        "episode_id_fallback",
    )


def _auto_thresholds(
    episodes: list[EpisodeTrace],
    *,
    quantiles: list[float],
) -> list[float]:
    values = []
    for episode in episodes:
        mask = episode.actor_active & episode.score_ready & episode.score.isfinite()
        if mask.any():
            values.append(episode.score[mask])
    if not values:
        raise RuntimeError(
            "No actor-active, score-ready values for automatic thresholds"
        )
    scores = torch.cat(values).to(torch.float32)
    thresholds = {0.0}
    for quantile in quantiles:
        thresholds.add(round(float(torch.quantile(scores, quantile).item()), 6))
    return sorted(thresholds, reverse=True)


def _finite_for_min(value: float) -> float:
    return value if not math.isnan(value) else math.inf


def _finite_for_max(value: float) -> float:
    return value if not math.isnan(value) else -math.inf


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_objectives = (
        _finite_for_max(left["oracle_stall_recall"]),
        _finite_for_max(left["takeover_rate_on_failure"]),
        -_finite_for_min(left["takeover_rate_on_success"]),
        -_finite_for_min(left["expert_fraction_given_critical"]),
    )
    right_objectives = (
        _finite_for_max(right["oracle_stall_recall"]),
        _finite_for_max(right["takeover_rate_on_failure"]),
        -_finite_for_min(right["takeover_rate_on_success"]),
        -_finite_for_min(right["expert_fraction_given_critical"]),
    )
    return all(
        left_value >= right_value
        for left_value, right_value in zip(left_objectives, right_objectives)
    ) and any(
        left_value > right_value
        for left_value, right_value in zip(left_objectives, right_objectives)
    )


def _mark_pareto_front(rows: list[dict[str, Any]], *, split: str) -> None:
    candidates = [row for row in rows if row["split"] == split]
    for candidate in candidates:
        candidate["pareto_front"] = not any(
            _dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )


def _base_profile_feasible(row: dict[str, Any]) -> bool:
    recall = row["oracle_stall_recall"]
    return (
        row["predicted_takeover_episodes"] > 0
        and (math.isnan(recall) or recall >= 0.70)
        and row["takeover_rate_on_success"] <= 0.30
        and row["expert_fraction_given_critical"] <= 0.30
    )


def _select_profiles(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows if row["split"] == split and bool(row["pareto_front"])
    ]
    if not candidates:
        candidates = [row for row in rows if row["split"] == split]

    def conservative_key(row: dict[str, Any]) -> tuple:
        return (
            not _base_profile_feasible(row),
            _finite_for_min(row["takeover_rate_on_success"]),
            _finite_for_min(row["predicted_takeover_episode_rate"]),
            -_finite_for_max(row["oracle_stall_recall"]),
            -_finite_for_max(row["takeover_rate_on_failure"]),
        )

    def balanced_key(row: dict[str, Any]) -> tuple:
        return (
            not _base_profile_feasible(row),
            abs(row["predicted_takeover_episode_rate"] - 0.20),
            _finite_for_min(row["takeover_rate_on_success"]),
            -_finite_for_max(row["oracle_stall_recall"]),
            _finite_for_min(row["expert_fraction_given_critical"]),
        )

    def rescue_key(row: dict[str, Any]) -> tuple:
        rescue_feasible = (
            row["predicted_takeover_episodes"] > 0
            and row["takeover_rate_on_success"] <= 0.40
            and row["expert_fraction_given_critical"] <= 0.40
        )
        return (
            not rescue_feasible,
            -_finite_for_max(row["takeover_rate_on_failure"]),
            _finite_for_min(row["takeover_rate_on_success"]),
            _finite_for_min(row["expert_fraction_given_critical"]),
        )

    profiles = []
    for name, key in (
        ("conservative", conservative_key),
        ("balanced", balanced_key),
        ("rescue_heavy", rescue_key),
    ):
        selected = dict(min(candidates, key=key))
        selected["profile"] = name
        profiles.append(selected)
    return profiles


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_confidence_intervals(
    episodes: list[EpisodeTrace],
    *,
    row: dict[str, Any],
    success_horizon_steps: int,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    if num_samples <= 0 or not episodes:
        return {}
    outcomes = _evaluate_outcomes(
        episodes,
        threshold=float(row["threshold"]),
        patience_chunks=int(row["patience_chunks"]),
        warmup_chunks=int(row["warmup_chunks"]),
    )
    rng = random.Random(seed)
    sampled_metrics = {key: [] for key in _BOOTSTRAP_METRICS}
    for _ in range(num_samples):
        sample = [outcomes[rng.randrange(len(outcomes))] for _ in outcomes]
        metrics = _aggregate_outcomes(
            sample,
            success_horizon_steps=success_horizon_steps,
        )
        for key in _BOOTSTRAP_METRICS:
            value = float(metrics[key])
            if not math.isnan(value):
                sampled_metrics[key].append(value)

    intervals = {}
    for key, values in sampled_metrics.items():
        intervals[f"{key}_ci95_low"] = _percentile(values, 0.025)
        intervals[f"{key}_ci95_high"] = _percentile(values, 0.975)
    return intervals


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_interval(row: dict[str, Any], key: str) -> str:
    value = float(row[key])
    low = float(row.get(f"{key}_ci95_low", math.nan))
    high = float(row.get(f"{key}_ci95_high", math.nan))
    if math.isnan(low) or math.isnan(high):
        return f"{value:.3f}"
    return f"{value:.3f}[{low:.3f},{high:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the RLT STEAM expert gate from shadow traces."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--thresholds",
        default="auto",
        help="Comma-separated thresholds, or 'auto' for score quantiles.",
    )
    parser.add_argument(
        "--threshold-quantiles",
        default="0.05,0.10,0.20,0.30,0.40,0.50,0.60",
    )
    parser.add_argument("--patience-chunks", default="3,4,5,6")
    parser.add_argument("--warmup-chunks", default="4,6,8")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--success-horizon-steps", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-version", type=int, default=None)
    parser.add_argument("--max-version", type=int, default=None)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include partial trailing episodes; unsafe for success/failure metrics.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--profiles-output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.success_horizon_steps < 0:
        parser.error("--success-horizon-steps must be non-negative")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    if (
        args.min_version is not None
        and args.max_version is not None
        and args.min_version > args.max_version
    ):
        parser.error("--min-version must be <= --max-version")

    patience_values = _parse_number_list(args.patience_chunks, int)
    warmup_values = _parse_number_list(args.warmup_chunks, int)
    if min(patience_values) < 1 or min(warmup_values) < 0:
        parser.error("patience must be positive and warmup must be non-negative")

    loaded_episodes = load_episodes(
        args.trace_dir,
        min_version=args.min_version,
        max_version=args.max_version,
    )
    complete_episodes = (
        loaded_episodes
        if args.include_incomplete
        else [episode for episode in loaded_episodes if episode.complete]
    )
    episodes = [
        episode for episode in complete_episodes if bool(episode.actor_active.any())
    ]
    if not episodes:
        raise RuntimeError(
            "No complete actor-active episodes remained. Check --min-version or "
            "use --include-incomplete only when partial episodes are intentional."
        )

    calibration, validation, split_strategy = _split_episodes_chronologically(
        episodes,
        validation_fraction=args.validation_fraction,
    )

    quantiles = _parse_number_list(args.threshold_quantiles, float)
    if min(quantiles) < 0.0 or max(quantiles) > 1.0:
        parser.error("--threshold-quantiles values must be in [0, 1]")
    if args.thresholds.strip().lower() == "auto":
        thresholds = _auto_thresholds(calibration, quantiles=quantiles)
    else:
        thresholds = _parse_number_list(args.thresholds, float)

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
                success_horizon_steps=args.success_horizon_steps,
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
                    success_horizon_steps=args.success_horizon_steps,
                )
            )

    for split in {row["split"] for row in rows}:
        _mark_pareto_front(rows, split=split)

    output = args.output or (args.trace_dir / "steam_expert_gate_grid.csv")
    _write_csv(output, rows)

    selected_profiles = _select_profiles(rows, split="calibration")
    rows_by_parameters = {
        (
            row["split"],
            row["threshold"],
            row["patience_chunks"],
            row["warmup_chunks"],
        ): row
        for row in rows
    }
    profiles = []
    evaluation_splits = [("calibration", calibration)]
    if validation:
        evaluation_splits.append(("validation", validation))
    for profile_idx, selected in enumerate(selected_profiles):
        parameters = (
            selected["threshold"],
            selected["patience_chunks"],
            selected["warmup_chunks"],
        )
        for split_idx, (split, split_episodes) in enumerate(evaluation_splits):
            profile = dict(rows_by_parameters[(split, *parameters)])
            profile["profile"] = selected["profile"]
            profile["selected_on"] = "calibration"
            profile.update(
                _bootstrap_confidence_intervals(
                    split_episodes,
                    row=profile,
                    success_horizon_steps=args.success_horizon_steps,
                    num_samples=args.bootstrap_samples,
                    seed=args.seed + profile_idx * len(evaluation_splits) + split_idx,
                )
            )
            profiles.append(profile)
    profiles_output = args.profiles_output or (
        args.trace_dir / "steam_expert_gate_profiles.csv"
    )
    _write_csv(profiles_output, profiles)

    incomplete_count = sum(not episode.complete for episode in loaded_episodes)
    noncritical_count = len(complete_episodes) - len(episodes)
    completeness = "episodes" if args.include_incomplete else "complete episodes"
    print(
        f"Loaded {len(loaded_episodes)} episodes; using {len(episodes)} "
        f"actor-active {completeness} ({incomplete_count} incomplete loaded, "
        f"{noncritical_count} without actor phase excluded)"
    )
    print(
        f"Chronological split ({split_strategy}): {len(calibration)} calibration, "
        f"{len(validation)} validation"
    )
    success_label_counts = {
        source: sum(episode.success_label_source == source for episode in episodes)
        for source in sorted({episode.success_label_source for episode in episodes})
    }
    print(
        "Success labels: "
        + ", ".join(
            f"{source}={count}" for source, count in success_label_counts.items()
        )
    )
    print(f"Thresholds: {','.join(f'{value:.6g}' for value in thresholds)}")
    print(f"Wrote {len(rows)} grid rows to {output}")
    print(f"Wrote profile candidates to {profiles_output}")
    print("Profiles selected on calibration and evaluated unchanged by split:")
    for row in profiles:
        print(
            "  {profile}/{split}: threshold={threshold:.4f} "
            "patience={patience_chunks} warmup={warmup_chunks} "
            "takeover={takeover} oracle_recall={recall} "
            "geometry_disagreement={disagreement} "
            "takeover|success={on_success} "
            "takeover|failure={on_failure} "
            "autonomous_success_after_signal={autonomous} "
            "expert|critical={expert_fraction}".format(
                **row,
                takeover=_format_interval(
                    row,
                    "predicted_takeover_episode_rate",
                ),
                recall=_format_interval(row, "oracle_stall_recall"),
                disagreement=_format_interval(
                    row,
                    "geometry_disagreement_episode_rate",
                ),
                on_success=_format_interval(row, "takeover_rate_on_success"),
                on_failure=_format_interval(row, "takeover_rate_on_failure"),
                autonomous=_format_interval(
                    row,
                    "autonomous_success_after_prediction_rate",
                ),
                expert_fraction=_format_interval(
                    row,
                    "expert_fraction_given_critical",
                ),
            )
        )


if __name__ == "__main__":
    main()
