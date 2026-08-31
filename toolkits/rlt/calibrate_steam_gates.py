#!/usr/bin/env python3
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

"""Train and calibrate both STEAM RLT gates from one trace collection.

The trace collection contains frozen STEAM features and geometry labels for the
base-to-actor phase head, plus STEAM scores and the geometry stalled-progress
oracle for the actor-to-expert gate. The phase head is trained first. Its
selected actor mask is then replayed over the traces before expert parameters
are calibrated, so expert calibration does not inherit a stale actor mask from
the collection run.
"""

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rlinf.algorithms.rlt.phase_head import RLT_PHASE_FEATURE_KEY, SteamPhaseHead


@dataclass(frozen=True)
class EpisodeRef:
    """A complete episode slice inside one batched trace."""

    episode_id: str
    trace_path: Path
    env_idx: int
    start: int
    end: int
    chunk_size: int
    version: int | None


@dataclass(frozen=True)
class EpisodeData:
    """Scalar trace data needed by both gate calibrators."""

    episode_id: str
    chunk_size: int
    score: torch.Tensor
    score_ready: torch.Tensor
    stored_actor_active: torch.Tensor
    actor_active: torch.Tensor
    geometry_active: torch.Tensor
    oracle_active: torch.Tensor
    success_signal: torch.Tensor
    success_label_source: str
    phase_probability: torch.Tensor | None = None


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


def _required_trace_tensor(
    trace: dict[str, Any],
    key: str,
    *,
    shape: tuple[int, int],
    dtype,
    trace_path: Path,
) -> torch.Tensor:
    value = trace.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(
            f"Trace key {key!r} in {trace_path} must be a rank-2 tensor"
        )
    if tuple(value.shape) != shape:
        raise ValueError(
            f"Trace key {key!r} in {trace_path} must have shape {shape}, "
            f"got {tuple(value.shape)}"
        )
    return value.to(dtype=dtype)


def _success_time_batch(
    trace: dict[str, Any],
    *,
    like: torch.Tensor,
    trace_path: Path,
) -> tuple[torch.Tensor, str]:
    """Recover success from action-aligned reward or explicit labels."""
    if trace.get("reward_sum") is not None:
        rewards = _as_time_batch(trace, "reward_sum", dtype=torch.float32)
        if rewards.shape != like.shape:
            raise ValueError(
                f"Trace key 'reward_sum' in {trace_path} must have shape "
                f"{tuple(like.shape)}, got {tuple(rewards.shape)}"
            )
        return rewards > 0, "reward_sum"
    if trace.get("success_current") is not None:
        success = _as_time_batch(trace, "success_current", dtype=torch.bool)
        if success.shape != like.shape:
            raise ValueError(
                f"Trace key 'success_current' in {trace_path} must have shape "
                f"{tuple(like.shape)}, got {tuple(success.shape)}"
            )
        return success, "success_current"
    if trace.get("terminations") is not None:
        terminations = _as_time_batch(trace, "terminations", dtype=torch.bool)
        if terminations.shape != like.shape:
            raise ValueError(
                f"Trace key 'terminations' in {trace_path} must have shape "
                f"{tuple(like.shape)}, got {tuple(terminations.shape)}"
            )
        return terminations, "terminations"
    raise ValueError(
        f"Trace {trace_path} has no 'success_current', 'terminations', or "
        "'reward_sum'; "
        "success labels cannot be reconstructed"
    )


def _episode_ranges(dones: torch.Tensor) -> Iterable[tuple[int, int, bool]]:
    previous_done = False
    start = 0
    for index, done in enumerate(dones.tolist()):
        current_done = bool(done)
        if current_done and not previous_done:
            yield start, index + 1, True
            start = index + 1
        previous_done = current_done
    if start < int(dones.shape[0]) and not previous_done:
        yield start, int(dones.shape[0]), False


def _episode_version(
    trace: dict[str, Any], start: int, end: int, env_idx: int
) -> int | None:
    versions = trace.get("versions")
    if versions is None:
        return None
    if not isinstance(versions, torch.Tensor) or versions.ndim != 2:
        raise ValueError("Trace key 'versions' must be a rank-2 tensor")
    value = versions[start:end, env_idx]
    return int(value.max().item()) if value.numel() else None


def discover_episode_refs(
    trace_dir: Path,
    *,
    min_version: int | None,
    max_version: int | None,
) -> list[EpisodeRef]:
    """Validate traces and return complete episode references."""
    trace_paths = sorted(trace_dir.rglob("trace_*.pt"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace_*.pt files found under {trace_dir}")

    refs: list[EpisodeRef] = []
    missing_features = 0
    for trace_path in trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        if int(trace.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported trace format in {trace_path}")
        features = trace.get(RLT_PHASE_FEATURE_KEY)
        if not isinstance(features, torch.Tensor):
            missing_features += 1
            continue
        if features.ndim != 4:
            raise ValueError(
                f"{RLT_PHASE_FEATURE_KEY} in {trace_path} must be [T,B,E,D]"
            )
        score = _as_time_batch(trace, "rlt_gate_score_min", dtype=torch.float32)
        expected_shape = tuple(score.shape)
        if tuple(features.shape[:2]) != expected_shape:
            raise ValueError(
                f"{RLT_PHASE_FEATURE_KEY} and score shape mismatch in {trace_path}: "
                f"{tuple(features.shape[:2])} != {expected_shape}"
            )
        for key in (
            "rlt_gate_score_ready",
            "rlt_gate_actor_active",
            "geometry_critical_active",
            "rlt_oracle_expert_active",
            "dones",
        ):
            _required_trace_tensor(
                trace,
                key,
                shape=expected_shape,
                dtype=torch.bool,
                trace_path=trace_path,
            )
        _success_time_batch(trace, like=score, trace_path=trace_path)

        dones = _as_time_batch(trace, "dones", dtype=torch.bool)
        for env_idx in range(score.shape[1]):
            for episode_idx, (start, end, complete) in enumerate(
                _episode_ranges(dones[:, env_idx])
            ):
                if not complete:
                    continue
                version = _episode_version(trace, start, end, env_idx)
                if min_version is not None and (
                    version is None or version < min_version
                ):
                    continue
                if max_version is not None and (
                    version is None or version > max_version
                ):
                    continue
                refs.append(
                    EpisodeRef(
                        episode_id=(
                            f"{trace_path.parent.name}/{trace_path.stem}:"
                            f"env{env_idx}:episode{episode_idx}"
                        ),
                        trace_path=trace_path,
                        env_idx=env_idx,
                        start=start,
                        end=end,
                        chunk_size=int(trace.get("chunk_size", 10)),
                        version=version,
                    )
                )
    if not refs:
        suffix = (
            f"; {missing_features} files did not contain {RLT_PHASE_FEATURE_KEY}"
            if missing_features
            else ""
        )
        raise RuntimeError(
            "No complete trace episodes found. Collect a run with "
            "rlt_gate_calibration enabled and actor_switch.collect_phase_features=True"
            + suffix
        )
    return refs


def _load_episode_data(ref: EpisodeRef, *, include_features: bool = False):
    trace = torch.load(ref.trace_path, map_location="cpu", weights_only=False)
    score = _as_time_batch(trace, "rlt_gate_score_min", dtype=torch.float32)
    score_ready = _as_time_batch(trace, "rlt_gate_score_ready", dtype=torch.bool)
    stored_actor_active = _as_time_batch(
        trace, "rlt_gate_actor_active", dtype=torch.bool
    )
    geometry_active = _as_time_batch(
        trace, "geometry_critical_active", dtype=torch.bool
    )
    oracle_active = _as_time_batch(
        trace, "rlt_oracle_expert_active", dtype=torch.bool
    )
    success_signal, success_label_source = _success_time_batch(
        trace,
        like=score,
        trace_path=ref.trace_path,
    )
    sl = slice(ref.start, ref.end)
    data = EpisodeData(
        episode_id=ref.episode_id,
        chunk_size=ref.chunk_size,
        score=score[sl, ref.env_idx].clone(),
        score_ready=score_ready[sl, ref.env_idx].clone(),
        stored_actor_active=stored_actor_active[sl, ref.env_idx].clone(),
        actor_active=stored_actor_active[sl, ref.env_idx].clone(),
        geometry_active=geometry_active[sl, ref.env_idx].clone(),
        oracle_active=oracle_active[sl, ref.env_idx].clone(),
        success_signal=success_signal[sl, ref.env_idx].clone(),
        success_label_source=success_label_source,
    )
    if not include_features:
        return data
    features = trace[RLT_PHASE_FEATURE_KEY][sl, ref.env_idx].to(torch.float16)
    return data, features


def split_episode_refs(
    refs: list[EpisodeRef],
    *,
    validation_fraction: float,
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    """Split complete episodes chronologically without chunk leakage."""
    if len(refs) < 2 or validation_fraction <= 0.0:
        return list(refs), []
    validation_size = max(1, round(len(refs) * validation_fraction))
    validation_size = min(validation_size, len(refs) - 1)
    ordered = sorted(refs, key=lambda ref: ref.episode_id)
    return ordered[:-validation_size], ordered[-validation_size:]


def _uniform_subset(indices: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or indices.numel() <= limit:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=limit).round().long()
    return indices[positions]


def _sample_phase_indices(
    ready: torch.Tensor,
    geometry_active: torch.Tensor,
    *,
    positive_window_chunks: int,
    max_negative_chunks: int,
    boundary_negative_chunks: int,
) -> torch.Tensor:
    valid_indices = torch.nonzero(ready, as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        return valid_indices
    geometry_indices = torch.nonzero(geometry_active, as_tuple=False).reshape(-1)
    if geometry_indices.numel() == 0:
        return _uniform_subset(valid_indices, max_negative_chunks)
    entry = int(geometry_indices[0].item())
    negative = valid_indices[valid_indices < entry]
    positive = valid_indices[
        (valid_indices >= entry)
        & (valid_indices < entry + max(1, positive_window_chunks))
    ]
    hard_negative = negative[negative >= entry - max(0, boundary_negative_chunks)]
    remaining_negative = negative[negative < entry - max(0, boundary_negative_chunks)]
    remaining_budget = max(max_negative_chunks - int(hard_negative.numel()), 0)
    sampled_negative = torch.cat(
        [_uniform_subset(remaining_negative, remaining_budget), hard_negative]
    )
    return torch.cat([sampled_negative, positive]).unique(sorted=True)


def build_training_tensors(
    refs: list[EpisodeRef],
    *,
    positive_window_chunks: int,
    max_negative_chunks: int,
    boundary_negative_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a compact boundary-focused dataset while streaming trace files."""
    features_list = []
    labels_list = []
    for ref in refs:
        data, features = _load_episode_data(ref, include_features=True)
        indices = _sample_phase_indices(
            data.score_ready,
            data.geometry_active,
            positive_window_chunks=positive_window_chunks,
            max_negative_chunks=max_negative_chunks,
            boundary_negative_chunks=boundary_negative_chunks,
        )
        if indices.numel() == 0:
            continue
        features_list.append(features[indices])
        labels_list.append(data.geometry_active[indices].to(torch.float32))
    if not features_list:
        raise RuntimeError("No score-ready phase-head samples were extracted")
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def _batch_loss(
    model: SteamPhaseHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(features.permute(1, 0, 2))
    targets = labels.unsqueeze(0).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
    )
    return loss, torch.sigmoid(logits).mean(dim=0)


@torch.no_grad()
def evaluate_classifier(
    model: SteamPhaseHead,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = positive = predicted_positive = true_positive = total = 0
    for features, labels in loader:
        features = features.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        loss, probabilities = _batch_loss(
            model, features, labels, pos_weight=pos_weight
        )
        predictions = probabilities >= 0.5
        targets = labels.to(torch.bool)
        count = int(labels.numel())
        loss_sum += float(loss.item()) * count
        correct += int((predictions == targets).sum().item())
        positive += int(targets.sum().item())
        predicted_positive += int(predictions.sum().item())
        true_positive += int((predictions & targets).sum().item())
        total += count
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "recall": true_positive / max(positive, 1),
        "precision": true_positive / max(predicted_positive, 1),
    }


def train_phase_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> SteamPhaseHead:
    """Train only the small phase head; STEAM features stay frozen."""
    ensemble_size = int(train_features.shape[1])
    input_dim = int(train_features.shape[2])
    model = SteamPhaseHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        ensemble_size=ensemble_size,
        dropout=dropout,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    positive = int(train_labels.to(torch.bool).sum().item())
    negative = int(train_labels.numel()) - positive
    if positive == 0 or negative == 0:
        raise RuntimeError(
            f"Phase-head training requires both labels, got {positive=} {negative=}"
        )
    pos_weight = torch.tensor(negative / positive, device=device)

    best_loss = math.inf
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for features, labels in train_loader:
            features = features.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(
                model, features, labels, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            count = int(labels.numel())
            running_loss += float(loss.item()) * count
            sample_count += count
        metrics = evaluate_classifier(
            model,
            validation_loader,
            device=device,
            pos_weight=pos_weight,
        )
        print(
            f"epoch={epoch:03d} train_loss={running_loss / max(sample_count, 1):.6f} "
            f"val_loss={metrics['loss']:.6f} val_accuracy={metrics['accuracy']:.3f} "
            f"val_recall={metrics['recall']:.3f} "
            f"val_precision={metrics['precision']:.3f}"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Phase-head training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model


@torch.no_grad()
def _predict_features(
    model: SteamPhaseHead,
    features: torch.Tensor,
    ready: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    probabilities = torch.zeros(ready.shape[0], dtype=torch.float32)
    indices = torch.nonzero(ready, as_tuple=False).reshape(-1)
    for start in range(0, int(indices.numel()), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_features = features[batch_indices].to(device=device)
        prediction = model.predict(batch_features.permute(1, 0, 2))
        probabilities[batch_indices] = prediction.detach().cpu()
    return probabilities


def predict_phase_records(
    model: SteamPhaseHead,
    refs: list[EpisodeRef],
    *,
    device: torch.device,
    batch_size: int,
) -> list[EpisodeData]:
    """Predict phase probabilities while retaining only scalar episode data."""
    records = []
    for ref in refs:
        data, features = _load_episode_data(ref, include_features=True)
        probabilities = _predict_features(
            model,
            features,
            data.score_ready,
            device=device,
            batch_size=batch_size,
        )
        records.append(replace(data, phase_probability=probabilities))
    return records


def _first_true(value: torch.Tensor) -> int | None:
    indices = torch.nonzero(value, as_tuple=False).reshape(-1)
    return None if indices.numel() == 0 else int(indices[0].item())


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def replay_probability_gate(
    probabilities: torch.Tensor,
    ready: torch.Tensor,
    *,
    threshold: float,
    patience_chunks: int,
) -> tuple[int | None, torch.Tensor]:
    """Replay the latched base-to-actor gate."""
    active = torch.zeros_like(ready)
    count = 0
    entered_at = None
    for index in range(int(ready.numel())):
        candidate = bool(ready[index]) and float(probabilities[index]) >= threshold
        count = count + 1 if candidate else 0
        if entered_at is None and count >= patience_chunks:
            entered_at = index
        active[index] = entered_at is not None
    return entered_at, active


def evaluate_phase_parameters(
    records: list[EpisodeData],
    *,
    threshold: float,
    patience_chunks: int,
) -> dict[str, Any]:
    geometry_positive = predicted_positive = matched = false_entry = 0
    within_one = within_two = 0
    absolute_deltas: list[float] = []
    disagreement = total = intersection = union = 0
    for record in records:
        assert record.phase_probability is not None
        predicted_entry, predicted_active = replay_probability_gate(
            record.phase_probability,
            record.score_ready,
            threshold=threshold,
            patience_chunks=patience_chunks,
        )
        geometry_entry = _first_true(record.geometry_active)
        has_geometry = geometry_entry is not None
        has_prediction = predicted_entry is not None
        geometry_positive += int(has_geometry)
        predicted_positive += int(has_prediction)
        matched += int(has_geometry and has_prediction)
        false_entry += int(has_prediction and not has_geometry)
        if has_geometry and has_prediction:
            delta = abs(predicted_entry - geometry_entry) * record.chunk_size
            absolute_deltas.append(float(delta))
            within_one += int(delta <= record.chunk_size)
            within_two += int(delta <= 2 * record.chunk_size)
        disagreement += int((predicted_active != record.geometry_active).sum().item())
        total += int(predicted_active.numel())
        intersection += int((predicted_active & record.geometry_active).sum().item())
        union += int((predicted_active | record.geometry_active).sum().item())
    count = len(records)
    geometry_negative = count - geometry_positive
    missed_entry = geometry_positive - matched
    identity_error = (missed_entry + false_entry) / max(count, 1)
    active_disagreement = disagreement / max(total, 1)
    return {
        "threshold": threshold,
        "patience_chunks": patience_chunks,
        "num_episodes": count,
        "geometry_entry_rate": geometry_positive / max(count, 1),
        "phase_entry_rate": predicted_positive / max(count, 1),
        "entry_rate_gap": abs(predicted_positive - geometry_positive) / max(count, 1),
        "geometry_entry_recall": matched / max(geometry_positive, 1),
        "false_entry_rate": false_entry / max(geometry_negative, 1),
        "entry_identity_error_rate": identity_error,
        "within_one_chunk_coverage": within_one / max(geometry_positive, 1),
        "within_two_chunks_coverage": within_two / max(geometry_positive, 1),
        "median_absolute_entry_delta_steps": _median(absolute_deltas),
        "active_disagreement_rate": active_disagreement,
        "calibration_loss": identity_error + active_disagreement,
        "critical_active_iou": intersection / max(union, 1),
    }


def _phase_thresholds(
    records: list[EpisodeData], quantiles: list[float]
) -> list[float]:
    values = [
        record.phase_probability[record.score_ready]
        for record in records
        if record.phase_probability is not None and bool(record.score_ready.any())
    ]
    if not values:
        raise RuntimeError("No score-ready phase probabilities for calibration")
    probabilities = torch.cat(values).to(torch.float32)
    candidates = {0.5}
    for quantile in quantiles:
        candidates.add(round(float(torch.quantile(probabilities, quantile)), 6))
    return sorted(candidates)


def select_phase_parameters(
    records: list[EpisodeData],
    *,
    thresholds: list[float],
    patience_values: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        evaluate_phase_parameters(
            records,
            threshold=threshold,
            patience_chunks=patience,
        )
        for threshold in thresholds
        for patience in patience_values
    ]

    def key(row: dict[str, Any]) -> tuple:
        median = row["median_absolute_entry_delta_steps"]
        return (
            float(row["calibration_loss"]),
            float(row["entry_identity_error_rate"]),
            float(row["active_disagreement_rate"]),
            float(row["entry_rate_gap"]),
            -float(row["within_one_chunk_coverage"]),
            math.inf if math.isnan(median) else float(median),
            int(row["patience_chunks"]),
        )

    if not rows:
        raise RuntimeError("Phase calibration produced no parameter rows")
    return dict(min(rows, key=key)), rows


def apply_phase_gate(
    records: list[EpisodeData],
    selected: dict[str, Any],
) -> list[EpisodeData]:
    result = []
    for record in records:
        assert record.phase_probability is not None
        _, actor_active = replay_probability_gate(
            record.phase_probability,
            record.score_ready,
            threshold=float(selected["threshold"]),
            patience_chunks=int(selected["patience_chunks"]),
        )
        result.append(replace(record, actor_active=actor_active))
    return result


def replay_expert_gate(
    record: EpisodeData,
    *,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
) -> tuple[int | None, torch.Tensor]:
    """Replay the deployed consecutive-low-score actor-to-expert gate."""
    active = torch.zeros_like(record.actor_active)
    low_progress_count = 0
    critical_chunk_count = -1
    entered_at = None
    latched = False
    for index in range(int(record.score.shape[0])):
        if not bool(record.actor_active[index]):
            critical_chunk_count = -1
            low_progress_count = 0
            continue
        critical_chunk_count += 1
        eligible = bool(record.score_ready[index]) and (
            critical_chunk_count >= warmup_chunks
        )
        low_progress = eligible and float(record.score[index]) <= threshold
        low_progress_count = low_progress_count + 1 if low_progress else 0
        if not latched and low_progress_count >= patience_chunks:
            latched = True
            entered_at = index
        active[index] = latched
    return entered_at, active


def evaluate_expert_parameters(
    records: list[EpisodeData],
    *,
    split: str,
    threshold: float,
    patience_chunks: int,
    warmup_chunks: int,
    success_horizon_steps: int,
) -> dict[str, Any]:
    outcomes = []
    for record in records:
        predicted_entry, predicted_active = replay_expert_gate(
            record,
            threshold=threshold,
            patience_chunks=patience_chunks,
            warmup_chunks=warmup_chunks,
        )
        oracle_entry = _first_true(record.oracle_active)
        actor_entry = _first_true(record.actor_active)
        success_entry = _first_true(record.success_signal)
        entry_delay = None
        if oracle_entry is not None and predicted_entry is not None:
            entry_delay = abs(predicted_entry - oracle_entry) * record.chunk_size
        actor_opportunity = None
        if actor_entry is not None and predicted_entry is not None:
            actor_opportunity = (predicted_entry - actor_entry) * record.chunk_size
        prediction_to_success = None
        if (
            predicted_entry is not None
            and success_entry is not None
            and success_entry >= predicted_entry
        ):
            prediction_to_success = (
                success_entry - predicted_entry
            ) * record.chunk_size
        outcomes.append(
            {
                "has_oracle": oracle_entry is not None,
                "has_prediction": predicted_entry is not None,
                "success": success_entry is not None,
                "critical_chunks": int(record.actor_active.sum().item()),
                "expert_chunks": int(
                    (predicted_active & record.actor_active).sum().item()
                ),
                "entry_delay_steps": entry_delay,
                "actor_opportunity_steps": actor_opportunity,
                "prediction_to_success_steps": prediction_to_success,
            }
        )

    count = len(outcomes)
    oracle_positive = sum(item["has_oracle"] for item in outcomes)
    predicted = sum(item["has_prediction"] for item in outcomes)
    successful = sum(item["success"] for item in outcomes)
    predicted_on_oracle = sum(
        item["has_prediction"] and item["has_oracle"] for item in outcomes
    )
    false_entries = sum(
        item["has_prediction"] and not item["has_oracle"] for item in outcomes
    )
    predicted_on_success = sum(
        item["has_prediction"] and item["success"] for item in outcomes
    )
    predicted_on_failure = sum(
        item["has_prediction"] and not item["success"] for item in outcomes
    )
    success_after_prediction = sum(
        item["prediction_to_success_steps"] is not None for item in outcomes
    )
    success_within_horizon = sum(
        item["prediction_to_success_steps"] is not None
        and item["prediction_to_success_steps"] <= success_horizon_steps
        for item in outcomes
    )
    critical_chunks = sum(item["critical_chunks"] for item in outcomes)
    expert_chunks = sum(item["expert_chunks"] for item in outcomes)
    entry_delays = [
        item["entry_delay_steps"]
        for item in outcomes
        if item["entry_delay_steps"] is not None
    ]
    actor_opportunities = [
        item["actor_opportunity_steps"]
        for item in outcomes
        if item["actor_opportunity_steps"] is not None
    ]
    prediction_to_success = [
        item["prediction_to_success_steps"]
        for item in outcomes
        if item["prediction_to_success_steps"] is not None
    ]
    oracle_negative = count - oracle_positive
    recall = _safe_rate(predicted_on_oracle, oracle_positive)
    false_entry_rate = _safe_rate(false_entries, oracle_negative)
    success_rate = _safe_rate(predicted_on_success, successful)
    failure_rate = _safe_rate(predicted_on_failure, count - successful)
    expert_fraction = _safe_rate(expert_chunks, critical_chunks)
    # The selection objective favors matching the geometry oracle while avoiding
    # unnecessary takeover of successful trajectories and excessive expert use.
    loss = (
        (1.0 - recall if not math.isnan(recall) else 1.0)
        + (false_entry_rate if not math.isnan(false_entry_rate) else 0.0)
        + 0.5 * (success_rate if not math.isnan(success_rate) else 0.0)
        + 0.25 * (expert_fraction if not math.isnan(expert_fraction) else 0.0)
    )
    return {
        "split": split,
        "threshold": threshold,
        "patience_chunks": patience_chunks,
        "warmup_chunks": warmup_chunks,
        "success_label_source": "+".join(
            sorted({record.success_label_source for record in records})
        ),
        "num_episodes": count,
        "successful_episodes": successful,
        "failed_episodes": count - successful,
        "oracle_positive_episodes": oracle_positive,
        "oracle_negative_episodes": oracle_negative,
        "predicted_takeover_episodes": predicted,
        "oracle_stall_recall": recall,
        "geometry_disagreement_episode_rate": false_entry_rate,
        "predicted_takeover_episode_rate": _safe_rate(predicted, count),
        "takeover_rate_on_success": success_rate,
        "takeover_rate_on_failure": failure_rate,
        "autonomous_success_after_prediction_rate": _safe_rate(
            success_after_prediction, predicted
        ),
        "autonomous_success_within_horizon_rate": _safe_rate(
            success_within_horizon, predicted
        ),
        "expert_fraction_given_critical": expert_fraction,
        "median_entry_delay_steps": _median(entry_delays),
        "median_actor_opportunity_steps": _median(actor_opportunities),
        "median_prediction_to_success_steps": _median(prediction_to_success),
        "success_episode_rate": _safe_rate(successful, count),
        "success_horizon_steps": success_horizon_steps,
        "calibration_loss": loss,
    }


def _expert_thresholds(
    records: list[EpisodeData],
    *,
    quantiles: list[float],
) -> list[float]:
    values = []
    for record in records:
        mask = record.actor_active & record.score_ready & record.score.isfinite()
        if mask.any():
            values.append(record.score[mask])
    if not values:
        raise RuntimeError(
            "No calibrated-actor, score-ready values for expert calibration"
        )
    scores = torch.cat(values).to(torch.float32)
    candidates = {0.0}
    for quantile in quantiles:
        candidates.add(round(float(torch.quantile(scores, quantile)), 6))
    return sorted(candidates, reverse=True)


def select_expert_parameters(
    records: list[EpisodeData],
    *,
    thresholds: list[float],
    patience_values: list[int],
    warmup_values: list[int],
    success_horizon_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        evaluate_expert_parameters(
            records,
            split="calibration",
            threshold=threshold,
            patience_chunks=patience,
            warmup_chunks=warmup,
            success_horizon_steps=success_horizon_steps,
        )
        for threshold in thresholds
        for patience in patience_values
        for warmup in warmup_values
    ]
    if not rows:
        raise RuntimeError("Expert calibration produced no parameter rows")

    def finite_min(value: float) -> float:
        return value if not math.isnan(value) else math.inf

    def finite_max(value: float) -> float:
        return value if not math.isnan(value) else -math.inf

    def feasible(row: dict[str, Any]) -> bool:
        recall = row["oracle_stall_recall"]
        return (
            row["predicted_takeover_episodes"] > 0
            and (math.isnan(recall) or recall >= 0.70)
            and row["takeover_rate_on_success"] <= 0.30
            and row["expert_fraction_given_critical"] <= 0.30
        )

    def key(row: dict[str, Any]) -> tuple:
        return (
            not feasible(row),
            finite_min(row["takeover_rate_on_success"]),
            finite_min(row["expert_fraction_given_critical"]),
            finite_min(row["predicted_takeover_episode_rate"]),
            -finite_max(row["oracle_stall_recall"]),
            -finite_max(row["takeover_rate_on_failure"]),
            float(row["calibration_loss"]),
        )

    selected = dict(min(rows, key=key))
    selected["profile"] = "recommended"
    return selected, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
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


def _write_recommendation(
    path: Path,
    *,
    phase_head_path: Path,
    phase: dict[str, Any],
    expert: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted_path = json.dumps(str(phase_head_path))
    content = (
        "rollout:\n"
        "  rlt_critical_phase_gate:\n"
        "    actor_switch:\n"
        "      enable: True\n"
        "      mode: active\n"
        f"      phase_head_path: {quoted_path}\n"
        f"      enter_threshold: {float(phase['threshold']):.6f}\n"
        f"      patience_chunks: {int(phase['patience_chunks'])}\n"
        "    expert_takeover:\n"
        "      enable: True\n"
        "      mode: active\n"
        f"      enter_threshold: {float(expert['threshold']):.6f}\n"
        f"      patience_chunks: {int(expert['patience_chunks'])}\n"
        f"      warmup_chunks: {int(expert['warmup_chunks'])}\n"
    )
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and calibrate both STEAM RLT gates from one trace set."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--phase-head-output", type=Path, default=None)
    parser.add_argument("--phase-grid-output", type=Path, default=None)
    parser.add_argument("--expert-grid-output", type=Path, default=None)
    parser.add_argument("--expert-profiles-output", type=Path, default=None)
    parser.add_argument("--yaml-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--phase-positive-window-chunks", type=int, default=4)
    parser.add_argument("--phase-max-negative-chunks", type=int, default=8)
    parser.add_argument("--phase-boundary-negative-chunks", type=int, default=4)
    parser.add_argument("--phase-thresholds", default="auto")
    parser.add_argument(
        "--phase-threshold-quantiles",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--phase-patience-chunks", default="1,2")
    parser.add_argument("--expert-thresholds", default="auto")
    parser.add_argument(
        "--expert-threshold-quantiles",
        default="0.05,0.10,0.20,0.30,0.40,0.50,0.60",
    )
    parser.add_argument("--expert-patience-chunks", default="3,4,5,6")
    parser.add_argument("--expert-warmup-chunks", default="4,6,8")
    parser.add_argument("--success-horizon-steps", type=int, default=100)
    parser.add_argument("--min-version", type=int, default=None)
    parser.add_argument("--max-version", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.success_horizon_steps < 0:
        parser.error("--success-horizon-steps must be non-negative")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    phase_patience = _parse_number_list(args.phase_patience_chunks, int)
    expert_patience = _parse_number_list(args.expert_patience_chunks, int)
    expert_warmup = _parse_number_list(args.expert_warmup_chunks, int)
    phase_quantiles = _parse_number_list(args.phase_threshold_quantiles, float)
    expert_quantiles = _parse_number_list(args.expert_threshold_quantiles, float)
    if not all(0.0 <= value <= 1.0 for value in phase_quantiles):
        parser.error("phase threshold quantiles must be in [0, 1]")
    if not all(0.0 <= value <= 1.0 for value in expert_quantiles):
        parser.error("expert threshold quantiles must be in [0, 1]")
    if min(phase_patience) < 1 or min(expert_patience) < 1:
        parser.error("patience values must be positive")
    if min(expert_warmup) < 1:
        parser.error("expert warmup values must be positive")

    refs = discover_episode_refs(
        args.trace_dir,
        min_version=args.min_version,
        max_version=args.max_version,
    )
    train_refs, validation_refs = split_episode_refs(
        refs,
        validation_fraction=args.validation_fraction,
    )
    if not validation_refs:
        validation_refs = list(train_refs)
    print(
        f"episodes: train={len(train_refs)} validation={len(validation_refs)}; "
        f"device={device}"
    )

    train_features, train_labels = build_training_tensors(
        train_refs,
        positive_window_chunks=args.phase_positive_window_chunks,
        max_negative_chunks=args.phase_max_negative_chunks,
        boundary_negative_chunks=args.phase_boundary_negative_chunks,
    )
    validation_features, validation_labels = build_training_tensors(
        validation_refs,
        positive_window_chunks=args.phase_positive_window_chunks,
        max_negative_chunks=args.phase_max_negative_chunks,
        boundary_negative_chunks=args.phase_boundary_negative_chunks,
    )
    if train_features.shape[1:] != validation_features.shape[1:]:
        raise RuntimeError("Train/validation STEAM phase feature shapes do not match")
    print(
        f"phase samples: train={len(train_labels)} "
        f"validation={len(validation_labels)}; "
        f"features={tuple(train_features.shape[1:])}"
    )

    phase_head = train_phase_head(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    del train_features, train_labels, validation_features, validation_labels

    phase_train_records = predict_phase_records(
        phase_head, train_refs, device=device, batch_size=args.batch_size
    )
    phase_validation_records = predict_phase_records(
        phase_head, validation_refs, device=device, batch_size=args.batch_size
    )
    if args.phase_thresholds.strip().lower() == "auto":
        phase_thresholds = _phase_thresholds(
            phase_train_records,
            quantiles=phase_quantiles,
        )
    else:
        phase_thresholds = _parse_number_list(args.phase_thresholds, float)
    if min(phase_thresholds) < 0.0 or max(phase_thresholds) > 1.0:
        parser.error("phase thresholds must be in [0, 1]")
    phase_selected, phase_rows = select_phase_parameters(
        phase_train_records,
        thresholds=phase_thresholds,
        patience_values=phase_patience,
    )
    phase_validation = evaluate_phase_parameters(
        phase_validation_records,
        threshold=float(phase_selected["threshold"]),
        patience_chunks=int(phase_selected["patience_chunks"]),
    )

    all_phase_records = phase_train_records + phase_validation_records
    calibrated_records = apply_phase_gate(all_phase_records, phase_selected)
    calibration_records = calibrated_records[: len(phase_train_records)]
    validation_records = calibrated_records[len(phase_train_records) :]
    if args.expert_thresholds.strip().lower() == "auto":
        expert_thresholds = _expert_thresholds(
            calibration_records,
            quantiles=expert_quantiles,
        )
    else:
        expert_thresholds = _parse_number_list(args.expert_thresholds, float)
    expert_selected, expert_rows = select_expert_parameters(
        calibration_records,
        thresholds=expert_thresholds,
        patience_values=expert_patience,
        warmup_values=expert_warmup,
        success_horizon_steps=args.success_horizon_steps,
    )
    expert_validation = evaluate_expert_parameters(
        validation_records,
        split="validation",
        threshold=float(expert_selected["threshold"]),
        patience_chunks=int(expert_selected["patience_chunks"]),
        warmup_chunks=int(expert_selected["warmup_chunks"]),
        success_horizon_steps=args.success_horizon_steps,
    )

    phase_output = args.phase_head_output or (args.trace_dir / "steam_phase_head.pt")
    phase_grid_output = args.phase_grid_output or phase_output.with_suffix(
        ".calibration.csv"
    )
    expert_grid_output = args.expert_grid_output or (
        args.trace_dir / "steam_expert_gate_grid.csv"
    )
    profiles_output = args.expert_profiles_output or (
        args.trace_dir / "steam_expert_gate_profiles.csv"
    )
    yaml_output = args.yaml_output or (
        args.trace_dir / "steam_gate_recommendation.yaml"
    )
    phase_metadata = {
        "recommended_enter_threshold": float(phase_selected["threshold"]),
        "recommended_patience_chunks": int(phase_selected["patience_chunks"]),
        "recommended_expert_gate": {
            "enter_threshold": float(expert_selected["threshold"]),
            "patience_chunks": int(expert_selected["patience_chunks"]),
            "warmup_chunks": int(expert_selected["warmup_chunks"]),
        },
        "phase_calibration": dict(phase_selected),
        "phase_validation": dict(phase_validation),
        "expert_calibration": dict(expert_selected),
        "expert_validation": dict(expert_validation),
        "train_episodes": len(train_refs),
        "validation_episodes": len(validation_refs),
        "seed": args.seed,
    }
    phase_head.save_checkpoint(phase_output, metadata=phase_metadata)
    _write_csv(phase_grid_output, phase_rows)
    _write_csv(expert_grid_output, expert_rows)
    profile_rows = [
        dict(expert_selected),
        {**expert_validation, "profile": "recommended"},
    ]
    _write_csv(profiles_output, profile_rows)
    _write_recommendation(
        yaml_output,
        phase_head_path=phase_output,
        phase=phase_selected,
        expert=expert_selected,
    )

    print(f"Saved phase head to {phase_output}")
    print(f"Saved phase calibration grid to {phase_grid_output}")
    print(f"Saved expert calibration grid to {expert_grid_output}")
    print(f"Saved expert profile and validation rows to {profiles_output}")
    print(f"Saved YAML recommendation to {yaml_output}")
    print("Recommended rollout.rlt_critical_phase_gate settings:")
    print(
        f"  actor_switch: threshold={float(phase_selected['threshold']):.6f} "
        f"patience={int(phase_selected['patience_chunks'])}"
    )
    print(
        f"  expert_takeover: threshold={float(expert_selected['threshold']):.6f} "
        f"patience={int(expert_selected['patience_chunks'])} "
        f"warmup={int(expert_selected['warmup_chunks'])}"
    )
    print(
        "  phase validation: recall={recall:.3f} false_entry={false_entry:.3f} "
        "within_1_chunk={within:.3f} active_iou={iou:.3f}".format(
            recall=float(phase_validation["geometry_entry_recall"]),
            false_entry=float(phase_validation["false_entry_rate"]),
            within=float(phase_validation["within_one_chunk_coverage"]),
            iou=float(phase_validation["critical_active_iou"]),
        )
    )
    print(
        "  expert validation: oracle_recall={recall:.3f} false_entry={false_entry:.3f} "
        "takeover_success={success:.3f} expert_fraction={fraction:.3f}".format(
            recall=float(expert_validation["oracle_stall_recall"]),
            false_entry=float(expert_validation["geometry_disagreement_episode_rate"]),
            success=float(expert_validation["takeover_rate_on_success"]),
            fraction=float(expert_validation["expert_fraction_given_critical"]),
        )
    )


if __name__ == "__main__":
    main()
