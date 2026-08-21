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

"""Calibrate STEAM base-to-actor routing against the geometry shadow gate."""

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class EpisodeTrace:
    """One episode of STEAM scores and geometry critical-phase labels."""

    episode_id: str
    chunk_size: int
    score: torch.Tensor
    score_ready: torch.Tensor
    geometry_active: torch.Tensor
    versions: torch.Tensor | None
    complete: bool


@dataclass(frozen=True)
class EpisodeOutcome:
    """One episode evaluated with a candidate STEAM critical-phase gate."""

    geometry_entered: bool
    steam_entered: bool
    entry_delta_steps: float | None
    intersection_chunks: int
    union_chunks: int
    geometry_chunks: int
    steam_chunks: int
    total_chunks: int
    chunk_size: int


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
    """Load gate traces and split their batched tensors into episodes."""
    episodes: list[EpisodeTrace] = []
    trace_paths = sorted(trace_dir.rglob("trace_*.pt"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace_*.pt files found under {trace_dir}")

    for trace_path in trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        if int(trace.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported trace format in {trace_path}")
        score = _as_time_batch(trace, "rlt_gate_score_min", dtype=torch.float32)
        score_ready = _as_time_batch(
            trace,
            "rlt_gate_score_ready",
            dtype=torch.bool,
        )
        geometry_active = _as_time_batch(
            trace,
            "geometry_critical_active",
            dtype=torch.bool,
        )
        dones = _as_time_batch(trace, "dones", dtype=torch.bool)
        for key, value in (
            ("rlt_gate_score_ready", score_ready),
            ("geometry_critical_active", geometry_active),
            ("dones", dones),
        ):
            if value.shape != score.shape:
                raise ValueError(
                    f"Trace key {key!r} in {trace_path} must have shape "
                    f"{tuple(score.shape)}, got {tuple(value.shape)}"
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
                        geometry_active=geometry_active[start:end, env_idx],
                        versions=episode_versions,
                        complete=complete,
                    )
                )
    if not episodes:
        raise RuntimeError("No episodes remained after trace filtering")
    return episodes


def replay_critical_gate(
    episode: EpisodeTrace,
    *,
    threshold: float,
    patience_chunks: int,
) -> tuple[int | None, torch.Tensor]:
    """Replay the deployed latched STEAM base-to-actor gate."""
    active = torch.zeros_like(episode.geometry_active)
    low_progress_count = 0
    entered_at = None
    latched = False

    for idx in range(int(episode.score.shape[0])):
        score = float(episode.score[idx])
        low_progress = (
            bool(episode.score_ready[idx])
            and math.isfinite(score)
            and score <= threshold
        )
        low_progress_count = low_progress_count + 1 if low_progress else 0
        if not latched and low_progress_count >= patience_chunks:
            latched = True
            entered_at = idx
        active[idx] = latched
    return entered_at, active


def _first_true(value: torch.Tensor) -> int | None:
    indices = torch.nonzero(value, as_tuple=False).reshape(-1)
    return None if indices.numel() == 0 else int(indices[0].item())


def _evaluate_episode(
    episode: EpisodeTrace,
    *,
    threshold: float,
    patience_chunks: int,
) -> EpisodeOutcome:
    steam_entry, steam_active = replay_critical_gate(
        episode,
        threshold=threshold,
        patience_chunks=patience_chunks,
    )
    geometry_entry = _first_true(episode.geometry_active)
    entry_delta_steps = None
    if geometry_entry is not None and steam_entry is not None:
        entry_delta_steps = float(steam_entry - geometry_entry) * episode.chunk_size

    intersection = steam_active & episode.geometry_active
    union = steam_active | episode.geometry_active
    return EpisodeOutcome(
        geometry_entered=geometry_entry is not None,
        steam_entered=steam_entry is not None,
        entry_delta_steps=entry_delta_steps,
        intersection_chunks=int(intersection.sum().item()),
        union_chunks=int(union.sum().item()),
        geometry_chunks=int(episode.geometry_active.sum().item()),
        steam_chunks=int(steam_active.sum().item()),
        total_chunks=int(steam_active.numel()),
        chunk_size=episode.chunk_size,
    )


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


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


def _aggregate_outcomes(outcomes: list[EpisodeOutcome]) -> dict[str, Any]:
    num_episodes = len(outcomes)
    geometry_positive = sum(outcome.geometry_entered for outcome in outcomes)
    geometry_negative = num_episodes - geometry_positive
    steam_positive = sum(outcome.steam_entered for outcome in outcomes)
    matched_positive = sum(
        outcome.geometry_entered and outcome.steam_entered for outcome in outcomes
    )
    false_positive = sum(
        outcome.steam_entered and not outcome.geometry_entered for outcome in outcomes
    )
    missed = sum(
        outcome.geometry_entered and not outcome.steam_entered for outcome in outcomes
    )
    deltas = [
        outcome.entry_delta_steps
        for outcome in outcomes
        if outcome.entry_delta_steps is not None
    ]
    absolute_deltas = [abs(delta) for delta in deltas]
    within_one_chunk = sum(
        abs(outcome.entry_delta_steps) <= outcome.chunk_size
        for outcome in outcomes
        if outcome.entry_delta_steps is not None
    )
    within_two_chunks = sum(
        abs(outcome.entry_delta_steps) <= 2 * outcome.chunk_size
        for outcome in outcomes
        if outcome.entry_delta_steps is not None
    )
    intersection_chunks = sum(outcome.intersection_chunks for outcome in outcomes)
    union_chunks = sum(outcome.union_chunks for outcome in outcomes)
    geometry_chunks = sum(outcome.geometry_chunks for outcome in outcomes)
    steam_chunks = sum(outcome.steam_chunks for outcome in outcomes)
    total_chunks = sum(outcome.total_chunks for outcome in outcomes)

    return {
        "num_episodes": num_episodes,
        "geometry_positive_episodes": geometry_positive,
        "geometry_negative_episodes": geometry_negative,
        "steam_positive_episodes": steam_positive,
        "matched_entry_episodes": matched_positive,
        "missed_entry_episodes": missed,
        "false_entry_episodes": false_positive,
        "geometry_entry_recall": _safe_rate(matched_positive, geometry_positive),
        "steam_entry_precision": _safe_rate(matched_positive, steam_positive),
        "missed_entry_rate": _safe_rate(missed, geometry_positive),
        "false_entry_rate": _safe_rate(false_positive, geometry_negative),
        "within_one_chunk_rate": _safe_rate(within_one_chunk, len(deltas)),
        "within_two_chunks_rate": _safe_rate(within_two_chunks, len(deltas)),
        "early_entry_rate": _safe_rate(sum(delta < 0 for delta in deltas), len(deltas)),
        "late_entry_rate": _safe_rate(sum(delta > 0 for delta in deltas), len(deltas)),
        "median_entry_delta_steps": (statistics.median(deltas) if deltas else math.nan),
        "median_absolute_entry_delta_steps": (
            statistics.median(absolute_deltas) if absolute_deltas else math.nan
        ),
        "mean_absolute_entry_delta_steps": (
            statistics.fmean(absolute_deltas) if absolute_deltas else math.nan
        ),
        "p90_absolute_entry_delta_steps": _percentile(absolute_deltas, 0.90),
        "critical_active_iou": _safe_rate(intersection_chunks, union_chunks),
        "geometry_critical_chunk_rate": _safe_rate(geometry_chunks, total_chunks),
        "steam_critical_chunk_rate": _safe_rate(steam_chunks, total_chunks),
    }


def evaluate_parameters(
    episodes: list[EpisodeTrace],
    *,
    split: str,
    threshold: float,
    patience_chunks: int,
) -> dict[str, Any]:
    """Evaluate one deployable STEAM critical-phase gate configuration."""
    outcomes = [
        _evaluate_episode(
            episode,
            threshold=threshold,
            patience_chunks=patience_chunks,
        )
        for episode in episodes
    ]
    return {
        "split": split,
        "threshold": threshold,
        "patience_chunks": patience_chunks,
        **_aggregate_outcomes(outcomes),
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
        mask = episode.score_ready & episode.score.isfinite()
        if mask.any():
            values.append(episode.score[mask])
    if not values:
        raise RuntimeError("No score-ready values for automatic thresholds")
    scores = torch.cat(values).to(torch.float32)
    thresholds = {-0.1, 0.0}
    for quantile in quantiles:
        thresholds.add(round(float(torch.quantile(scores, quantile).item()), 6))
    return sorted(thresholds, reverse=True)


def _finite_for_min(value: float) -> float:
    return value if not math.isnan(value) else math.inf


def _finite_for_max(value: float) -> float:
    return value if not math.isnan(value) else -math.inf


def select_recommended(
    rows: list[dict[str, Any]],
    *,
    split: str,
    min_entry_recall: float,
    max_false_entry_rate: float,
) -> dict[str, Any]:
    """Select the closest timing profile subject to entry-rate constraints."""
    candidates = [row for row in rows if row["split"] == split]
    if not candidates:
        raise ValueError(f"No rows found for split {split!r}")

    def key(row: dict[str, Any]) -> tuple:
        recall = float(row["geometry_entry_recall"])
        false_rate = float(row["false_entry_rate"])
        recall_shortfall = max(0.0, min_entry_recall - recall)
        false_excess = (
            0.0
            if math.isnan(false_rate)
            else max(0.0, false_rate - max_false_entry_rate)
        )
        feasible = recall_shortfall == 0.0 and false_excess == 0.0
        return (
            not feasible,
            recall_shortfall + false_excess,
            -_finite_for_max(float(row["within_one_chunk_rate"])),
            _finite_for_min(float(row["median_absolute_entry_delta_steps"])),
            _finite_for_min(float(row["p90_absolute_entry_delta_steps"])),
            _finite_for_min(false_rate),
            _finite_for_min(float(row["early_entry_rate"])),
            -_finite_for_max(float(row["critical_active_iou"])),
            int(row["patience_chunks"]),
        )

    selected = dict(min(candidates, key=key))
    recall = float(selected["geometry_entry_recall"])
    false_rate = float(selected["false_entry_rate"])
    selected["feasible"] = recall >= min_entry_recall and (
        math.isnan(false_rate) or false_rate <= max_false_entry_rate
    )
    selected["min_entry_recall"] = min_entry_recall
    selected["max_false_entry_rate"] = max_false_entry_rate
    return selected


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


def _format_metric(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the RLT STEAM base-to-actor gate against geometry labels."
        )
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--thresholds",
        default="auto",
        help="Comma-separated thresholds, or 'auto' for score quantiles.",
    )
    parser.add_argument(
        "--threshold-quantiles",
        default="0.01,0.025,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.975,0.99",
    )
    parser.add_argument("--patience-chunks", default="1,2,3,4")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-entry-recall", type=float, default=0.95)
    parser.add_argument("--max-false-entry-rate", type=float, default=0.05)
    parser.add_argument("--min-version", type=int, default=None)
    parser.add_argument("--max-version", type=int, default=None)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include partial trailing episodes in timing calibration.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--profile-output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if not 0.0 <= args.min_entry_recall <= 1.0:
        parser.error("--min-entry-recall must be in [0, 1]")
    if not 0.0 <= args.max_false_entry_rate <= 1.0:
        parser.error("--max-false-entry-rate must be in [0, 1]")
    if (
        args.min_version is not None
        and args.max_version is not None
        and args.min_version > args.max_version
    ):
        parser.error("--min-version must be <= --max-version")

    patience_values = _parse_number_list(args.patience_chunks, int)
    if min(patience_values) < 1:
        parser.error("patience must be positive")
    quantiles = _parse_number_list(args.threshold_quantiles, float)
    if min(quantiles) < 0.0 or max(quantiles) > 1.0:
        parser.error("--threshold-quantiles values must be in [0, 1]")

    loaded_episodes = load_episodes(
        args.trace_dir,
        min_version=args.min_version,
        max_version=args.max_version,
    )
    episodes = (
        loaded_episodes
        if args.include_incomplete
        else [episode for episode in loaded_episodes if episode.complete]
    )
    if not episodes:
        raise RuntimeError(
            "No complete episodes remained. Use --include-incomplete only when "
            "partial episodes are intentional."
        )

    calibration, validation, split_strategy = _split_episodes_chronologically(
        episodes,
        validation_fraction=args.validation_fraction,
    )
    if not any(bool(episode.geometry_active.any()) for episode in calibration):
        raise RuntimeError("Calibration split has no geometry-positive episodes")

    if args.thresholds.strip().lower() == "auto":
        thresholds = _auto_thresholds(calibration, quantiles=quantiles)
    else:
        thresholds = _parse_number_list(args.thresholds, float)

    rows = []
    for threshold, patience in product(thresholds, patience_values):
        rows.append(
            evaluate_parameters(
                calibration,
                split="calibration",
                threshold=threshold,
                patience_chunks=patience,
            )
        )
        if validation:
            rows.append(
                evaluate_parameters(
                    validation,
                    split="validation",
                    threshold=threshold,
                    patience_chunks=patience,
                )
            )

    output = args.output or (args.trace_dir / "steam_critical_gate_grid.csv")
    _write_csv(output, rows)
    selected = select_recommended(
        rows,
        split="calibration",
        min_entry_recall=args.min_entry_recall,
        max_false_entry_rate=args.max_false_entry_rate,
    )
    parameters = (
        float(selected["threshold"]),
        int(selected["patience_chunks"]),
    )
    profile_rows = []
    for split, split_episodes in (
        ("calibration", calibration),
        ("validation", validation),
    ):
        if not split_episodes:
            continue
        row = evaluate_parameters(
            split_episodes,
            split=split,
            threshold=parameters[0],
            patience_chunks=parameters[1],
        )
        row["selected_on"] = "calibration"
        row["feasible_on_calibration"] = bool(selected["feasible"])
        row["min_entry_recall"] = args.min_entry_recall
        row["max_false_entry_rate"] = args.max_false_entry_rate
        profile_rows.append(row)
    profile_output = args.profile_output or (
        args.trace_dir / "steam_critical_gate_profile.csv"
    )
    _write_csv(profile_output, profile_rows)

    incomplete_count = sum(not episode.complete for episode in loaded_episodes)
    print(
        f"Loaded {len(loaded_episodes)} episodes; using {len(episodes)} "
        f"({'including' if args.include_incomplete else 'excluding'} "
        f"{incomplete_count} incomplete episodes)"
    )
    print(
        f"Chronological split ({split_strategy}): {len(calibration)} calibration, "
        f"{len(validation)} validation"
    )
    print(f"Wrote {len(rows)} grid rows to {output}")
    print(f"Wrote recommended profile to {profile_output}")
    print("Recommended rollout.rlt_critical_phase_gate settings:")
    print(f"  enter_threshold: {parameters[0]:.6g}")
    print(f"  patience_chunks: {parameters[1]}")
    print(f"  feasible: {bool(selected['feasible'])}")
    for row in profile_rows:
        print(
            "  {split}: recall={recall} false_entry={false_entry} "
            "within_1_chunk={within_one} median_abs_delta={median_abs:.1f} "
            "p90_abs_delta={p90:.1f} active_iou={iou}".format(
                split=row["split"],
                recall=_format_metric(float(row["geometry_entry_recall"])),
                false_entry=_format_metric(float(row["false_entry_rate"])),
                within_one=_format_metric(float(row["within_one_chunk_rate"])),
                median_abs=float(row["median_absolute_entry_delta_steps"]),
                p90=float(row["p90_absolute_entry_delta_steps"]),
                iou=_format_metric(float(row["critical_active_iou"])),
            )
        )


if __name__ == "__main__":
    main()
