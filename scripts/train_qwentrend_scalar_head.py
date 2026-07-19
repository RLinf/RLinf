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

"""Train a scalar reward head on frozen Qwen rollout features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rlinf.models.embodiment.reward.vlm_reward_model import ScalarPotentialHead


def load_potential_shards(pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    paths = (
        sorted(Path().glob(pattern))
        if not pattern.startswith("/")
        else sorted(Path(pattern).parent.glob(Path(pattern).name))
    )
    if not paths:
        raise ValueError(f"No feature shards match {pattern}")
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    return (
        torch.cat([payload["features"].float() for payload in payloads]),
        torch.cat([payload["targets"].float() for payload in payloads]),
    )


def load_progress_shards(pattern: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    paths = sorted(Path(pattern).parent.glob(Path(pattern).name))
    if not paths:
        raise ValueError(f"No progress shards match {pattern}")
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    labels = [label for payload in payloads for label in payload["labels"]]
    return (
        torch.cat([payload["features"].float() for payload in payloads]),
        torch.cat([payload["teacher_deltas"].float() for payload in payloads]),
        labels,
    )


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


@torch.no_grad()
def predict(
    model: nn.Module, features: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start : start + batch_size].to(device))
        outputs.append(torch.sigmoid(logits).cpu())
    return torch.cat(outputs)


def evaluate(
    model: nn.Module,
    potential_features: torch.Tensor,
    potential_targets: torch.Tensor,
    progress_features: torch.Tensor,
    progress_deltas: torch.Tensor,
    progress_labels: list[str],
    device: torch.device,
    batch_size: int,
    deadband: float,
) -> dict[str, Any]:
    values = predict(model, potential_features, device, batch_size)
    pair_values = predict(
        model,
        progress_features.reshape(-1, progress_features.shape[-1]),
        device,
        batch_size,
    ).reshape(-1, 2)
    predicted_deltas = pair_values[:, 1] - pair_values[:, 0]
    predicted_labels = [
        "up" if value > deadband else "down" if value < -deadband else "same"
        for value in predicted_deltas.tolist()
    ]
    return {
        "potential_mae": float(torch.abs(values - potential_targets).mean()),
        "potential_mse": float(torch.mean((values - potential_targets) ** 2)),
        "potential_spearman": safe_spearman(values.numpy(), potential_targets.numpy()),
        "delta_spearman": safe_spearman(
            predicted_deltas.numpy(), progress_deltas.numpy()
        ),
        "direction_accuracy": float(
            np.mean(
                [
                    prediction == target
                    for prediction, target in zip(predicted_labels, progress_labels)
                ]
            )
        ),
        "predicted_delta_mean": float(predicted_deltas.mean()),
        "predicted_delta_std": float(predicted_deltas.std()),
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_features, train_targets = load_potential_shards(args.train_pattern)
    eval_features, eval_targets = load_potential_shards(args.eval_pattern)
    progress_features, progress_deltas, progress_labels = load_progress_shards(
        args.progress_pattern
    )
    train_progress_features, train_progress_deltas, _ = load_progress_shards(
        args.train_progress_pattern
    )
    model = ScalarPotentialHead(
        train_features.shape[-1], args.hidden_dim, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(train_features, train_targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    progress_dataset = TensorDataset(train_progress_features, train_progress_deltas)
    progress_loader = DataLoader(
        progress_dataset, batch_size=args.batch_size, shuffle=True
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            value_losses = []
            delta_losses = []
            local_rank_losses = []
            progress_iterator = iter(progress_loader)
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                logits = model(features)
                value_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets
                )
                permutation = torch.randperm(len(targets), device=device)
                target_difference = targets - targets[permutation]
                rank_mask = target_difference.abs() >= args.rank_min_gap
                if rank_mask.any():
                    logit_difference = logits - logits[permutation]
                    rank_loss = nn.functional.softplus(
                        -torch.sign(target_difference[rank_mask])
                        * logit_difference[rank_mask]
                    ).mean()
                else:
                    rank_loss = logits.sum() * 0.0
                try:
                    pair_features, pair_targets = next(progress_iterator)
                except StopIteration:
                    progress_iterator = iter(progress_loader)
                    pair_features, pair_targets = next(progress_iterator)
                pair_features = pair_features.to(device)
                pair_targets = pair_targets.to(device)
                pair_logits = model(
                    pair_features.reshape(-1, pair_features.shape[-1])
                ).reshape(-1, 2)
                predicted_deltas = torch.sigmoid(pair_logits[:, 1]) - torch.sigmoid(
                    pair_logits[:, 0]
                )
                delta_loss = nn.functional.smooth_l1_loss(
                    predicted_deltas, pair_targets, beta=args.delta_beta
                )
                local_rank_mask = pair_targets.abs() >= args.local_rank_min_gap
                if local_rank_mask.any():
                    local_logit_differences = pair_logits[:, 1] - pair_logits[:, 0]
                    local_rank_loss = nn.functional.softplus(
                        -torch.sign(pair_targets[local_rank_mask])
                        * local_logit_differences[local_rank_mask]
                    ).mean()
                else:
                    local_rank_loss = pair_logits.sum() * 0.0
                loss = (
                    value_loss
                    + args.rank_weight * rank_loss
                    + args.delta_weight * delta_loss
                    + args.local_rank_weight * local_rank_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                delta_losses.append(float(delta_loss.detach().cpu()))
                local_rank_losses.append(float(local_rank_loss.detach().cpu()))

            if epoch % args.eval_interval != 0 and epoch != args.epochs:
                continue
            metrics = evaluate(
                model,
                eval_features,
                eval_targets,
                progress_features,
                progress_deltas,
                progress_labels,
                device,
                args.eval_batch_size,
                args.progress_deadband,
            )
            metrics.update(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "train_value_loss": float(np.mean(value_losses)),
                    "train_delta_loss": float(np.mean(delta_losses)),
                    "train_local_rank_loss": float(np.mean(local_rank_losses)),
                }
            )
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            score = metrics["potential_spearman"] + metrics["delta_spearman"]
            if score > best_score:
                best_score = score
                best_metrics = metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "input_dim": int(train_features.shape[-1]),
                            "hidden_dim": args.hidden_dim,
                            "dropout": args.dropout,
                        },
                        "metrics": metrics,
                    },
                    output_dir / "best.pt",
                )
            print(json.dumps(metrics))
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pattern", required=True)
    parser.add_argument("--eval-pattern", required=True)
    parser.add_argument("--progress-pattern", required=True)
    parser.add_argument("--train-progress-pattern", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--rank-min-gap", type=float, default=0.1)
    parser.add_argument("--delta-weight", type=float, default=10.0)
    parser.add_argument("--delta-beta", type=float, default=0.03)
    parser.add_argument("--local-rank-weight", type=float, default=0.2)
    parser.add_argument("--local-rank-min-gap", type=float, default=0.01)
    parser.add_argument("--progress-deadband", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
