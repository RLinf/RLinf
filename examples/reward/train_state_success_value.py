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

"""Train a task-agnostic state success value model from collected episodes.

The model predicts a success-conditioned potential from state history. For a
successful episode, targets increase toward 1 near the final successful state;
for a failed episode, targets are 0. The saved checkpoint can then be used to
label image windows for QwenTrend VLM reward SFT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass
class ValueConfig:
    """Hyperparameters and whitening stats stored inside a teacher checkpoint."""

    state_dim: int
    history_size: int
    hidden_dim: int
    num_layers: int
    dropout: float
    gamma: float
    target_mode: str
    mean: list[float]
    std: list[float]


class StateSuccessValue(nn.Module):
    """Small MLP value model over flattened state history."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return per-sample logits for success-potential BCE training."""
        return self.net(states).squeeze(-1)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _stack_history(states: list[np.ndarray], idx: int, history_size: int) -> np.ndarray:
    first = states[0]
    frames = []
    for offset in range(history_size - 1, -1, -1):
        hist_idx = idx - offset
        frames.append(states[hist_idx] if hist_idx >= 0 else first)
    return np.concatenate(frames, axis=0).astype(np.float32)


def _build_targets(length: int, success: bool, gamma: float, mode: str) -> np.ndarray:
    if not success:
        return np.zeros(length, dtype=np.float32)
    if mode == "discounted_success":
        return np.asarray(
            [gamma ** (length - 1 - idx) for idx in range(length)],
            dtype=np.float32,
        )
    if mode == "linear_success":
        if length == 1:
            return np.ones(1, dtype=np.float32)
        return np.linspace(0.0, 1.0, length, dtype=np.float32)
    raise ValueError(f"Unsupported target mode: {mode}")


def load_state_dataset(
    raw_data_path: str,
    history_size: int,
    gamma: float,
    target_mode: str,
    val_split: float,
    seed: int,
    max_episodes: int | None,
) -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict[str, Any]
]:
    """Load episode states into whitened train/eval tensors and summary metadata.

    Returns:
        ``((train_x, train_y), (val_x, val_y), metadata)``.
    """
    pkl_files = sorted(glob(os.path.join(raw_data_path, "*.pkl")))
    if max_episodes is not None:
        pkl_files = pkl_files[:max_episodes]
    if not pkl_files:
        raise ValueError(f"No episode pkl files found in {raw_data_path}")

    rng = random.Random(seed)
    rng.shuffle(pkl_files)
    val_count = max(1, int(round(len(pkl_files) * val_split)))
    val_files = set(pkl_files[:val_count])

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    episode_counts = {
        "train_success": 0,
        "train_fail": 0,
        "val_success": 0,
        "val_fail": 0,
    }

    state_dim = None
    for pkl_path in tqdm(pkl_files, desc="Loading state episodes", unit="episode"):
        with open(pkl_path, "rb") as f:
            episode = pickle.load(f)
        observations = episode.get("observations", [])
        if not observations:
            continue
        states = []
        for obs in observations:
            if "states" not in obs:
                continue
            states.append(_to_numpy(obs["states"]).reshape(-1))
        if not states:
            continue
        if state_dim is None:
            state_dim = int(states[0].shape[0])
        if any(int(state.shape[0]) != state_dim for state in states):
            continue

        success = bool(episode.get("success", False))
        targets = _build_targets(len(states), success, gamma, target_mode)
        inputs = np.stack(
            [_stack_history(states, idx, history_size) for idx in range(len(states))],
            axis=0,
        )
        if pkl_path in val_files:
            val_x.append(inputs)
            val_y.append(targets)
            episode_counts["val_success" if success else "val_fail"] += 1
        else:
            train_x.append(inputs)
            train_y.append(targets)
            episode_counts["train_success" if success else "train_fail"] += 1

    if state_dim is None or not train_x or not val_x:
        raise ValueError("Failed to build non-empty train/eval state datasets")

    train_x_arr = np.concatenate(train_x, axis=0)
    train_y_arr = np.concatenate(train_y, axis=0)
    val_x_arr = np.concatenate(val_x, axis=0)
    val_y_arr = np.concatenate(val_y, axis=0)
    metadata = {
        "num_episodes": len(pkl_files),
        "state_dim": state_dim,
        "history_size": history_size,
        "train_samples": int(train_x_arr.shape[0]),
        "val_samples": int(val_x_arr.shape[0]),
        **episode_counts,
    }
    return (train_x_arr, train_y_arr), (val_x_arr, val_y_arr), metadata


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate BCE loss, MSE, and MAE of sigmoid predictions on ``loader``."""
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            losses.append(float(loss.detach().cpu()))
            preds.append(torch.sigmoid(logits).detach().cpu())
            targets.append(batch_y.detach().cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    return {
        "loss": float(np.mean(losses)),
        "mse": float(mse),
        "mae": float(mae),
        "pred_mean": float(pred.mean().item()),
        "target_mean": float(target.mean().item()),
    }


def train(args: argparse.Namespace) -> None:
    """Train the MLP state-success teacher and write ``best.pt`` / ``final.pt``."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, metadata = load_state_dataset(
        raw_data_path=args.raw_data_path,
        history_size=args.history_size,
        gamma=args.gamma,
        target_mode=args.target_mode,
        val_split=args.val_split,
        seed=args.seed,
        max_episodes=args.max_episodes,
    )
    train_x, train_y = train_data
    val_x, val_y = val_data

    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    train_ds = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_x.astype(np.float32)),
        torch.from_numpy(val_y.astype(np.float32)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = ValueConfig(
        state_dim=int(metadata["state_dim"]),
        history_size=args.history_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gamma=args.gamma,
        target_mode=args.target_mode,
        mean=mean.squeeze(0).astype(float).tolist(),
        std=std.squeeze(0).astype(float).tolist(),
    )
    model = StateSuccessValue(
        input_dim=cfg.state_dim * cfg.history_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.max_steps or (args.epochs * math.ceil(len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
    )

    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    global_step = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                logits = model(batch_x)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                global_step += 1
                pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

                if global_step % args.eval_interval == 0:
                    val_metrics = evaluate(model, val_loader, device)
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": float(loss.detach().cpu()),
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                    metrics_file.write(json.dumps(row) + "\n")
                    metrics_file.flush()
                    if val_metrics["loss"] < best_val:
                        best_val = val_metrics["loss"]
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "config": asdict(cfg),
                                "metadata": metadata,
                                "step": global_step,
                                "val_metrics": val_metrics,
                            },
                            output_dir / "best.pt",
                        )
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break

    final_metrics = evaluate(model, val_loader, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "metadata": metadata,
            "step": global_step,
            "val_metrics": final_metrics,
        },
        output_dir / "final.pt",
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "best_val_loss": best_val,
                "final_metrics": final_metrics,
                "global_step": global_step,
                "checkpoint": str(output_dir / "final.pt"),
            },
            f,
            indent=2,
        )
    logger.info("%s", json.dumps(final_metrics, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for state-success value teacher training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument(
        "--target-mode",
        choices=("discounted_success", "linear_success"),
        default="discounted_success",
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
