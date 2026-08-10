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

"""Flat Success training entry: teacher MLP, feature extract, scalar head.

Stages:
  teacher       Train StateSuccessValue from episode states
  extract       Extract frozen VLM features for one shard
  scalar_head   Train ScalarPotentialHead on feature shards

Example:
  python examples/reward/train_vlm_trend_success_model.py --stage teacher ...
  python examples/reward/train_vlm_trend_success_model.py --stage extract ...
  python examples/reward/train_vlm_trend_success_model.py --stage scalar_head ...

Multi-GPU extract: launch one --stage extract per rank with --rank/--world-size
(same pattern as the collection loop in the docs).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from rlinf.data.datasets.vlm import VLMTrendRewardSFTDataset
from rlinf.data.datasets.vlm_trend_io import (
    extract_extra_view_image,
    potential_prompt,
    source_episode_hash,
    to_numpy_float32,
    to_uint8_rgb,
)
from rlinf.models.embodiment.reward.vlm_reward_model import (
    ScalarPotentialHead,
    VLMRewardModel,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.state_success_value import (
    StateSuccessValue,
    stack_state_history,
)

logger = get_logger()

# --- stage: teacher ---


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
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError) as exc:
            logger.warning("Skipping unreadable episode %s: %s", pkl_path, exc)
            continue
        observations = episode.get("observations", [])
        if not observations:
            continue
        states = []
        for obs in observations:
            if "states" not in obs:
                continue
            states.append(to_numpy_float32(obs["states"]).reshape(-1))
        if not states:
            continue
        if state_dim is None:
            state_dim = int(states[0].shape[0])
        if any(int(state.shape[0]) != state_dim for state in states):
            continue

        success = bool(episode.get("success", False))
        targets = _build_targets(len(states), success, gamma, target_mode)
        inputs = np.stack(
            [
                stack_state_history(states, idx, history_size)
                for idx in range(len(states))
            ],
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


def evaluate_teacher(
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


def run_teacher(args: argparse.Namespace) -> None:
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
                    val_metrics = evaluate_teacher(model, val_loader, device)
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

    final_metrics = evaluate_teacher(model, val_loader, device)
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


# --- stage: extract ---


def read_rows(path: Path, sample_type: str) -> list[dict[str, Any]]:
    """Load JSONL manifest rows filtered to a single ``sample_type``."""
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [
        row for row in rows if row["segment_metadata"]["sample_type"] == sample_type
    ]


def load_frames(path: str) -> tuple[list[Any], list[Any]]:
    """Load dual-view frame lists from a window pickle."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["main_frames"], payload["extra_view_frames"]


@functools.lru_cache(maxsize=1)
def load_episode(path: str) -> dict[str, Any]:
    """Load and cache a full episode pickle for history reconstruction."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_history_frames(
    row: dict[str, Any], end_step: int, history_size: int
) -> tuple[list[Any], list[Any]]:
    """Load a left-padded dual-view history ending at an original rollout step."""
    source_window = int(row["segment_metadata"]["window_size"])
    sample_type = row["segment_metadata"]["sample_type"]
    if history_size == source_window and row["pkl_path"] != row["source_episode_path"]:
        main, extra = load_frames(row["pkl_path"])
        if sample_type == "progress":
            source_end = int(row["segment_metadata"]["end_step"])
            if end_step == source_end:
                return main[source_window:], extra[source_window:]
            return main[:source_window], extra[:source_window]
        return main, extra

    episode = load_episode(row["source_episode_path"])
    observations = episode["observations"]
    if end_step < 0 or end_step >= len(observations):
        raise IndexError(
            f"end_step {end_step} outside episode with {len(observations)} steps"
        )
    indices = list(range(max(0, end_step - history_size + 1), end_step + 1))
    indices = [indices[0]] * (history_size - len(indices)) + indices
    main_frames = []
    extra_frames = []
    for index in indices:
        observation = observations[index]
        main = observation.get("main_images")
        extra = observation.get("third_view_images")
        if extra is None:
            extra = extract_extra_view_image(observation.get("extra_view_images"))
        if main is None or extra is None:
            raise ValueError(
                f"Missing dual-view image at {row['source_episode_path']}:{index}"
            )
        main_frames.append(to_uint8_rgb(main))
        extra_frames.append(to_uint8_rgb(extra))
    return main_frames, extra_frames


@torch.no_grad()
def encode(
    model: VLMRewardModel,
    prompts: list[str],
    videos: list[list[Any]],
) -> torch.Tensor:
    """Encode prompt/video batches into pooled VLM features on CPU."""
    _, inputs, _ = VLMTrendRewardSFTDataset.process_inputs(
        processor=model._processor,
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[[prompt] for prompt in prompts],
        videos=videos,
        answer_text=None,
    )
    inputs = {
        key: value.to(model._model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    # Reuse the reward model's pooling so offline features match online inference.
    return model.extract_prompt_features(inputs).cpu()


def extract_potential(
    model: VLMRewardModel,
    rows: list[dict[str, Any]],
    batch_size: int,
    history_size: int,
) -> dict[str, Any]:
    """Extract potential features and teacher targets for dense-head training."""
    features = []
    targets = []
    successes = []
    source_paths = []
    end_steps = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        videos = []
        for row in batch:
            end_step = int(row["segment_metadata"]["end_step"])
            main, extra = load_history_frames(row, end_step, history_size)
            videos.append([main, extra])
        prompts = [potential_prompt(row["task"], history_size, 10) for row in batch]
        features.append(encode(model, prompts, videos))
        targets.extend(float(row["supervision"]["teacher_value"]) for row in batch)
        successes.extend(bool(row["segment_metadata"]["success"]) for row in batch)
        source_paths.extend(row["source_episode_path"] for row in batch)
        end_steps.extend(int(row["segment_metadata"]["end_step"]) for row in batch)
    return {
        "features": torch.cat(features).to(torch.float16),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "successes": torch.tensor(successes, dtype=torch.bool),
        "source_paths": source_paths,
        "end_steps": torch.tensor(end_steps, dtype=torch.int32),
    }


def extract_progress(
    model: VLMRewardModel,
    rows: list[dict[str, Any]],
    batch_size: int,
    history_size: int,
) -> dict[str, Any]:
    """Extract paired earlier/current features for progress pairwise training."""
    pair_features = []
    deltas = []
    labels = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = []
        videos = []
        for row in batch:
            metadata = row["segment_metadata"]
            clip_size = int(metadata["window_size"])
            end_step = int(metadata["end_step"])
            earlier_end = int(metadata["start_step"]) + clip_size - 1
            earlier = load_history_frames(row, earlier_end, history_size)
            current = load_history_frames(row, end_step, history_size)
            prompt = potential_prompt(row["task"], history_size, 10)
            prompts.extend([prompt, prompt])
            videos.extend([[earlier[0], earlier[1]], [current[0], current[1]]])
            deltas.append(float(row["supervision"]["teacher_delta"]))
            labels.append(row["answer"])
        encoded = encode(model, prompts, videos)
        pair_features.append(encoded.reshape(len(batch), 2, -1))
    return {
        "features": torch.cat(pair_features).to(torch.float16),
        "teacher_deltas": torch.tensor(deltas, dtype=torch.float32),
        "labels": labels,
    }


def run_extract(args: argparse.Namespace) -> None:
    """Shard a manifest, extract VLM features, and write a rank-local shard."""
    rows = read_rows(Path(args.manifest), args.sample_type)
    # Keep every window from an episode on one rank. The episode pickle is large,
    # so row-wise sharding makes every rank deserialize nearly every episode.
    rows = [
        row
        for row in rows
        if source_episode_hash(row["source_episode_path"]) % args.world_size
        == args.rank
    ]
    rows.sort(
        key=lambda row: (
            row["source_episode_path"],
            row["segment_metadata"]["end_step"],
        )
    )
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    cfg = OmegaConf.create(
        {
            "model_path": args.model_path,
            "lora_path": args.checkpoint,
            "precision": "bf16",
            "inference_mode": "generate",
            "input_builder_name": "vlm_trend_reward_input_builder",
            "input_builder_params": {
                "history_buffer_names": ["history_window"],
                "prompt_template": (
                    "You are currently performing the task: {task}. "
                    "Given the current state, predict the success potential."
                ),
            },
            "reward_parser_name": "base_reward_parser",
            "reward_parser_params": {},
        }
    )
    model = VLMRewardModel(cfg)
    model._model.to(args.device).eval()
    if args.sample_type == "potential":
        payload = extract_potential(model, rows, args.batch_size, args.history_size)
    else:
        payload = extract_progress(model, rows, args.batch_size, args.history_size)
    payload["metadata"] = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "sample_type": args.sample_type,
        "rank": args.rank,
        "world_size": args.world_size,
        "num_samples": len(rows),
        "history_size": args.history_size,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    logger.info("%s", json.dumps(payload["metadata"], indent=2))


# --- stage: scalar_head ---


def load_potential_shards(pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate potential feature shards matching ``pattern``.

    Returns:
        ``(features, targets)`` float tensors concatenated along the batch dim.
    """
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
    """Load and concatenate progress feature shards matching ``pattern``.

    Returns:
        ``(pair_features, teacher_deltas, labels)`` where each feature row is a
        pair of windows and ``labels`` are ``up`` / ``same`` / ``down`` strings.
    """
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
    """Return Spearman correlation, or ``0.0`` when the statistic is non-finite."""
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


@torch.no_grad()
def predict(
    model: nn.Module, features: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    """Run the scalar head and return sigmoid probabilities on CPU."""
    model.eval()
    outputs = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start : start + batch_size].to(device))
        outputs.append(torch.sigmoid(logits).cpu())
    return torch.cat(outputs)


def evaluate_scalar_head(
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
    """Evaluate potential fit and progress-direction accuracy.

    Returns:
        Metrics including MAE/MSE/Spearman on potentials and direction accuracy
        on progress pairs using ``deadband`` for up/same/down labels.
    """
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


def _pairwise_rank_loss(
    logits: torch.Tensor, targets: torch.Tensor, min_gap: float
) -> torch.Tensor:
    """Softplus pairwise ranking loss over a random permutation of the batch."""
    permutation = torch.randperm(len(targets), device=targets.device)
    target_difference = targets - targets[permutation]
    rank_mask = target_difference.abs() >= min_gap
    if not rank_mask.any():
        return logits.sum() * 0.0
    logit_difference = logits - logits[permutation]
    return nn.functional.softplus(
        -torch.sign(target_difference[rank_mask]) * logit_difference[rank_mask]
    ).mean()


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    progress_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Run one optimization epoch and return mean train losses."""
    model.train()
    losses: list[float] = []
    value_losses: list[float] = []
    delta_losses: list[float] = []
    local_rank_losses: list[float] = []
    progress_iterator = iter(progress_loader)
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        value_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        rank_loss = _pairwise_rank_loss(logits, targets, args.rank_min_gap)
        try:
            pair_features, pair_targets = next(progress_iterator)
        except StopIteration:
            progress_iterator = iter(progress_loader)
            pair_features, pair_targets = next(progress_iterator)
        pair_features = pair_features.to(device)
        pair_targets = pair_targets.to(device)
        pair_logits = model(pair_features.reshape(-1, pair_features.shape[-1])).reshape(
            -1, 2
        )
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
    return {
        "train_loss": float(np.mean(losses)),
        "train_value_loss": float(np.mean(value_losses)),
        "train_delta_loss": float(np.mean(delta_losses)),
        "train_local_rank_loss": float(np.mean(local_rank_losses)),
    }


def run_scalar_head(args: argparse.Namespace) -> None:
    """Train the scalar potential head and write ``best.pt`` / metrics."""
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
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )
    progress_loader = DataLoader(
        TensorDataset(train_progress_features, train_progress_deltas),
        batch_size=args.batch_size,
        shuffle=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            train_losses = _train_one_epoch(
                model, optimizer, loader, progress_loader, args, device
            )
            if epoch % args.eval_interval != 0 and epoch != args.epochs:
                continue
            metrics = evaluate_scalar_head(
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
            metrics.update({"epoch": epoch, **train_losses})
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
            logger.info("%s", json.dumps(metrics))
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )


def _add_teacher_args(parser: argparse.ArgumentParser) -> None:
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


def _add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-type", choices=("potential", "progress"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)


def _add_scalar_head_args(parser: argparse.ArgumentParser) -> None:
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


_STAGE_ADDERS = {
    "teacher": _add_teacher_args,
    "extract": _add_extract_args,
    "scalar_head": _add_scalar_head_args,
}
_STAGE_RUNNERS = {
    "teacher": run_teacher,
    "extract": run_extract,
    "scalar_head": run_scalar_head,
}


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(
        description="VLM Trend Success training stages.",
        add_help=False,
    )
    pre.add_argument(
        "--stage",
        choices=tuple(_STAGE_ADDERS),
        help="Which Success training stage to run.",
    )
    pre.add_argument("-h", "--help", action="store_true")
    args, remaining = pre.parse_known_args(argv)
    if args.stage is None:
        if args.help:
            pre.print_help()
            print(
                "\nStage-specific flags follow --stage. Example:\n"
                "  python examples/reward/train_vlm_trend_success_model.py "
                "--stage teacher --help"
            )
            sys.exit(0)
        pre.error("--stage is required")

    if args.help:
        remaining = ["--help"]

    parser = argparse.ArgumentParser(
        description=f"VLM Trend Success train (--stage {args.stage})"
    )
    _STAGE_ADDERS[args.stage](parser)
    stage_args = parser.parse_args(remaining)
    _STAGE_RUNNERS[args.stage](stage_args)


if __name__ == "__main__":
    main()
