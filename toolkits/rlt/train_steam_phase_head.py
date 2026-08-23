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

"""Train a critical-phase head from frozen STEAM features and geometry labels."""

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rlinf.algorithms.rlt.phase_head import RLT_PHASE_FEATURE_KEY, SteamPhaseHead


@dataclass(frozen=True)
class EpisodeRef:
    """Slice of one batched trace corresponding to a complete episode."""

    episode_id: str
    trace_path: Path
    env_idx: int
    start: int
    end: int
    chunk_size: int


@dataclass(frozen=True)
class PhaseEpisode:
    """One episode of frozen STEAM features and geometry labels."""

    episode_id: str
    chunk_size: int
    features: torch.Tensor
    ready: torch.Tensor
    geometry_active: torch.Tensor


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


def discover_episode_refs(trace_dir: Path) -> list[EpisodeRef]:
    """Discover complete episodes in phase-feature trace files."""
    refs = []
    trace_paths = sorted(trace_dir.rglob("trace_*.pt"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace_*.pt files found under {trace_dir}")
    missing_features = 0
    for trace_path in trace_paths:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        features = trace.get(RLT_PHASE_FEATURE_KEY)
        if not isinstance(features, torch.Tensor):
            missing_features += 1
            continue
        if features.ndim != 4:
            raise ValueError(
                f"{RLT_PHASE_FEATURE_KEY} in {trace_path} must be [T,B,E,D]"
            )
        dones = trace.get("dones")
        if not isinstance(dones, torch.Tensor) or dones.shape != features.shape[:2]:
            raise ValueError(f"Invalid dones tensor in {trace_path}")
        for env_idx in range(features.shape[1]):
            for episode_idx, (start, end, complete) in enumerate(
                _episode_ranges(dones[:, env_idx].to(torch.bool))
            ):
                if not complete:
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
                    )
                )
    if not refs:
        suffix = (
            f"; {missing_features} files did not contain {RLT_PHASE_FEATURE_KEY}"
            if missing_features
            else ""
        )
        raise RuntimeError(
            "No complete phase-feature episodes found. Collect a new hybrid run "
            "with actor_switch.collect_phase_features=True" + suffix
        )
    return refs


def _load_episode(ref: EpisodeRef, trace: dict[str, Any]) -> PhaseEpisode:
    features = trace[RLT_PHASE_FEATURE_KEY][
        ref.start : ref.end,
        ref.env_idx,
    ]
    ready = trace["rlt_gate_score_ready"][ref.start : ref.end, ref.env_idx]
    geometry_active = trace["geometry_critical_active"][
        ref.start : ref.end,
        ref.env_idx,
    ]
    if features.ndim != 3:
        raise ValueError(f"Invalid episode features in {ref.trace_path}")
    if ready.shape != geometry_active.shape or ready.ndim != 1:
        raise ValueError(f"Invalid phase labels in {ref.trace_path}")
    return PhaseEpisode(
        episode_id=ref.episode_id,
        chunk_size=ref.chunk_size,
        features=features,
        ready=ready.to(torch.bool),
        geometry_active=geometry_active.to(torch.bool),
    )


def iter_episodes(refs: list[EpisodeRef]) -> Iterable[PhaseEpisode]:
    """Load episode slices while reusing each trace file once."""
    ordered = sorted(
        refs, key=lambda ref: (str(ref.trace_path), ref.env_idx, ref.start)
    )
    current_path = None
    trace = None
    for ref in ordered:
        if ref.trace_path != current_path:
            trace = torch.load(
                ref.trace_path,
                map_location="cpu",
                weights_only=False,
            )
            current_path = ref.trace_path
        assert trace is not None
        yield _load_episode(ref, trace)


def split_episode_refs(
    refs: list[EpisodeRef],
    *,
    validation_fraction: float,
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    """Chronologically split complete episodes without chunk leakage."""
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


def _sample_episode_indices(
    episode: PhaseEpisode,
    *,
    positive_window_chunks: int,
    max_negative_chunks: int,
    boundary_negative_chunks: int,
) -> torch.Tensor:
    valid_indices = torch.nonzero(episode.ready, as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        return valid_indices
    geometry_indices = torch.nonzero(
        episode.geometry_active,
        as_tuple=False,
    ).reshape(-1)
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
    """Extract a compact boundary-focused feature dataset."""
    features = []
    labels = []
    for episode in iter_episodes(refs):
        indices = _sample_episode_indices(
            episode,
            positive_window_chunks=positive_window_chunks,
            max_negative_chunks=max_negative_chunks,
            boundary_negative_chunks=boundary_negative_chunks,
        )
        if indices.numel() == 0:
            continue
        features.append(episode.features[indices].to(torch.float16))
        labels.append(episode.geometry_active[indices].to(torch.float32))
    if not features:
        raise RuntimeError("No score-ready phase-head samples were extracted")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


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
    probabilities = torch.sigmoid(logits).mean(dim=0)
    return loss, probabilities


@torch.no_grad()
def evaluate_classifier(
    model: SteamPhaseHead,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    """Evaluate sample-level phase classification."""
    model.eval()
    losses = []
    correct = 0
    positive = 0
    predicted_positive = 0
    true_positive = 0
    total = 0
    for features, labels in loader:
        features = features.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        loss, probabilities = _batch_loss(
            model,
            features,
            labels,
            pos_weight=pos_weight,
        )
        predictions = probabilities >= 0.5
        targets = labels.to(torch.bool)
        batch_size = int(labels.numel())
        losses.append(float(loss.item()) * batch_size)
        correct += int((predictions == targets).sum().item())
        positive += int(targets.sum().item())
        predicted_positive += int(predictions.sum().item())
        true_positive += int((predictions & targets).sum().item())
        total += batch_size
    return {
        "loss": sum(losses) / max(total, 1),
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
    """Train phase heads while keeping STEAM features frozen."""
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
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
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
        loss_sum = 0.0
        sample_count = 0
        for features, labels in train_loader:
            features = features.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(
                model,
                features,
                labels,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            batch_samples = int(labels.numel())
            loss_sum += float(loss.item()) * batch_samples
            sample_count += batch_samples
        metrics = evaluate_classifier(
            model,
            validation_loader,
            device=device,
            pos_weight=pos_weight,
        )
        print(
            f"epoch={epoch:03d} train_loss={loss_sum / max(sample_count, 1):.6f} "
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
def predict_episode(
    model: SteamPhaseHead,
    episode: PhaseEpisode,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Predict phase probability for every score-ready chunk."""
    probabilities = torch.zeros(episode.ready.shape[0], dtype=torch.float32)
    indices = torch.nonzero(episode.ready, as_tuple=False).reshape(-1)
    for start in range(0, int(indices.numel()), batch_size):
        batch_indices = indices[start : start + batch_size]
        features = episode.features[batch_indices].to(device=device)
        prediction = model.predict(features.permute(1, 0, 2))
        probabilities[batch_indices] = prediction.probability.detach().cpu()
    return probabilities


def replay_probability_gate(
    probabilities: torch.Tensor,
    ready: torch.Tensor,
    *,
    threshold: float,
    patience_chunks: int,
) -> tuple[int | None, torch.Tensor]:
    """Replay a latched probability gate."""
    active = torch.zeros_like(ready)
    count = 0
    entered_at = None
    for idx in range(int(ready.numel())):
        candidate = bool(ready[idx]) and float(probabilities[idx]) >= threshold
        count = count + 1 if candidate else 0
        if entered_at is None and count >= patience_chunks:
            entered_at = idx
        active[idx] = entered_at is not None
    return entered_at, active


def _first_true(value: torch.Tensor) -> int | None:
    indices = torch.nonzero(value, as_tuple=False).reshape(-1)
    return None if indices.numel() == 0 else int(indices[0].item())


def evaluate_gate_parameters(
    predictions: list[tuple[PhaseEpisode, torch.Tensor]],
    *,
    threshold: float,
    patience_chunks: int,
) -> dict[str, float | int]:
    """Evaluate entry identity, timing, and full active-mask agreement."""
    geometry_positive = 0
    predicted_positive = 0
    matched = 0
    false_entry = 0
    within_one = 0
    within_two = 0
    absolute_deltas = []
    disagreement_chunks = 0
    total_chunks = 0
    intersection_chunks = 0
    union_chunks = 0
    for episode, probabilities in predictions:
        predicted_entry, predicted_active = replay_probability_gate(
            probabilities,
            episode.ready,
            threshold=threshold,
            patience_chunks=patience_chunks,
        )
        geometry_entry = _first_true(episode.geometry_active)
        geometry_entered = geometry_entry is not None
        predicted_entered = predicted_entry is not None
        geometry_positive += int(geometry_entered)
        predicted_positive += int(predicted_entered)
        matched += int(geometry_entered and predicted_entered)
        false_entry += int(predicted_entered and not geometry_entered)
        if geometry_entered and predicted_entered:
            delta = abs(predicted_entry - geometry_entry) * episode.chunk_size
            absolute_deltas.append(float(delta))
            within_one += int(delta <= episode.chunk_size)
            within_two += int(delta <= 2 * episode.chunk_size)
        disagreement_chunks += int(
            (predicted_active != episode.geometry_active).sum().item()
        )
        total_chunks += int(predicted_active.numel())
        intersection_chunks += int(
            (predicted_active & episode.geometry_active).sum().item()
        )
        union_chunks += int((predicted_active | episode.geometry_active).sum().item())
    num_episodes = len(predictions)
    geometry_negative = num_episodes - geometry_positive
    missed_entry = geometry_positive - matched
    entry_identity_error_rate = (missed_entry + false_entry) / max(num_episodes, 1)
    active_disagreement_rate = disagreement_chunks / max(total_chunks, 1)
    return {
        "threshold": threshold,
        "patience_chunks": patience_chunks,
        "num_episodes": num_episodes,
        "geometry_entry_rate": geometry_positive / max(num_episodes, 1),
        "phase_entry_rate": predicted_positive / max(num_episodes, 1),
        "entry_rate_gap": abs(predicted_positive - geometry_positive)
        / max(num_episodes, 1),
        "geometry_entry_recall": matched / max(geometry_positive, 1),
        "false_entry_rate": false_entry / max(geometry_negative, 1),
        "entry_identity_error_rate": entry_identity_error_rate,
        "within_one_chunk_coverage": within_one / max(geometry_positive, 1),
        "within_two_chunks_coverage": within_two / max(geometry_positive, 1),
        "median_absolute_entry_delta_steps": (
            statistics.median(absolute_deltas) if absolute_deltas else math.inf
        ),
        "active_disagreement_rate": active_disagreement_rate,
        "calibration_loss": entry_identity_error_rate + active_disagreement_rate,
        "critical_active_iou": intersection_chunks / max(union_chunks, 1),
    }


def calibrate_probability_gate(
    predictions: list[tuple[PhaseEpisode, torch.Tensor]],
    *,
    thresholds: list[float],
    patience_values: list[int],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Select the gate whose latched active mask best matches geometry."""
    rows = [
        evaluate_gate_parameters(
            predictions,
            threshold=threshold,
            patience_chunks=patience,
        )
        for threshold in thresholds
        for patience in patience_values
    ]

    def key(row: dict[str, float | int]) -> tuple:
        return (
            float(row["calibration_loss"]),
            float(row["entry_identity_error_rate"]),
            float(row["active_disagreement_rate"]),
            float(row["entry_rate_gap"]),
            -float(row["within_one_chunk_coverage"]),
            -float(row["within_two_chunks_coverage"]),
            float(row["median_absolute_entry_delta_steps"]),
            -float(row["geometry_entry_recall"]),
            int(row["patience_chunks"]),
        )

    return dict(min(rows, key=key)), rows


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_number_list(value: str, cast) -> list:
    result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return result


def _automatic_thresholds(
    predictions: list[tuple[PhaseEpisode, torch.Tensor]],
) -> list[float]:
    values = [
        probabilities[episode.ready]
        for episode, probabilities in predictions
        if bool(episode.ready.any())
    ]
    if not values:
        raise RuntimeError("No score-ready phase probabilities for calibration")
    probabilities = torch.cat(values).to(torch.float32)
    quantiles = torch.linspace(0.0, 1.0, steps=101)
    thresholds = {
        round(float(value.item()), 6)
        for value in torch.quantile(probabilities, quantiles)
    }
    thresholds.add(0.5)
    return sorted(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an RLT critical-phase head from STEAM feature traces."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--positive-window-chunks", type=int, default=4)
    parser.add_argument("--max-negative-chunks", type=int, default=8)
    parser.add_argument("--boundary-negative-chunks", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        default="auto",
        help="Comma-separated probabilities, or 'auto' for validation quantiles.",
    )
    parser.add_argument("--patience-chunks", default="1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")
    if args.positive_window_chunks < 1 or args.max_negative_chunks < 1:
        parser.error("sample-window arguments must be positive")
    thresholds = (
        None
        if args.thresholds.strip().lower() == "auto"
        else _parse_number_list(args.thresholds, float)
    )
    patience_values = _parse_number_list(args.patience_chunks, int)
    if thresholds is not None and (min(thresholds) < 0.0 or max(thresholds) > 1.0):
        parser.error("--thresholds values must be in [0, 1]")
    if min(patience_values) < 1:
        parser.error("--patience-chunks values must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    refs = discover_episode_refs(args.trace_dir)
    train_refs, validation_refs = split_episode_refs(
        refs,
        validation_fraction=args.validation_fraction,
    )
    train_features, train_labels = build_training_tensors(
        train_refs,
        positive_window_chunks=args.positive_window_chunks,
        max_negative_chunks=args.max_negative_chunks,
        boundary_negative_chunks=args.boundary_negative_chunks,
    )
    validation_features, validation_labels = build_training_tensors(
        validation_refs,
        positive_window_chunks=args.positive_window_chunks,
        max_negative_chunks=args.max_negative_chunks,
        boundary_negative_chunks=args.boundary_negative_chunks,
    )
    if train_features.shape[1:] != validation_features.shape[1:]:
        raise RuntimeError("Train/validation STEAM phase feature shapes do not match")
    print(
        f"episodes: train={len(train_refs)} validation={len(validation_refs)}; "
        f"samples: train={len(train_labels)} validation={len(validation_labels)}; "
        f"features={tuple(train_features.shape[1:])}; device={device}"
    )

    model = train_phase_head(
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
    del train_features, validation_features

    predictions = []
    for episode in iter_episodes(validation_refs):
        probabilities = predict_episode(
            model,
            episode,
            device=device,
            batch_size=args.batch_size,
        )
        predictions.append((episode, probabilities))
    if thresholds is None:
        thresholds = _automatic_thresholds(predictions)
    selected, rows = calibrate_probability_gate(
        predictions,
        thresholds=thresholds,
        patience_values=patience_values,
    )

    output = args.output or (args.trace_dir / "steam_phase_head.pt")
    calibration_output = output.with_suffix(".calibration.csv")
    _write_csv(calibration_output, rows)
    metadata = {
        "recommended_enter_threshold": float(selected["threshold"]),
        "recommended_patience_chunks": int(selected["patience_chunks"]),
        "calibration": dict(selected),
        "train_episodes": len(train_refs),
        "validation_episodes": len(validation_refs),
        "seed": args.seed,
    }
    model.save_checkpoint(output, metadata=metadata)
    print(f"Saved phase head to {output}")
    print(f"Saved calibration grid to {calibration_output}")
    print("Recommended rollout.rlt_critical_phase_gate.actor_switch settings:")
    print(f"  phase_head_path: {output}")
    print(f"  enter_threshold: {float(selected['threshold']):.3f}")
    print(f"  patience_chunks: {int(selected['patience_chunks'])}")
    print(
        "  validation: calibration_loss={loss:.3f} "
        "entry_identity_error={identity_error:.3f} disagreement={disagreement:.3f} "
        "entry_rate_gap={rate_gap:.3f} recall={recall:.3f} "
        "false_entry={false_entry:.3f} within_1_chunk={within_one:.3f} "
        "median_abs_delta={median_abs:.1f} active_iou={iou:.3f}".format(
            loss=float(selected["calibration_loss"]),
            identity_error=float(selected["entry_identity_error_rate"]),
            disagreement=float(selected["active_disagreement_rate"]),
            rate_gap=float(selected["entry_rate_gap"]),
            recall=float(selected["geometry_entry_recall"]),
            false_entry=float(selected["false_entry_rate"]),
            within_one=float(selected["within_one_chunk_coverage"]),
            median_abs=float(selected["median_absolute_entry_delta_steps"]),
            iou=float(selected["critical_active_iou"]),
        )
    )


if __name__ == "__main__":
    main()
