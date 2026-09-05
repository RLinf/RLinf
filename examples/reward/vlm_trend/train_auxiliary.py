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

"""Train the VLM Trend state teacher and scalar potential head."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from examples.reward.vlm_trend.data import (
    first_success_transition,
    transition_observations,
)
from rlinf.models.embodiment.modules.utils import make_mlp
from rlinf.models.embodiment.reward.vlm_trend_success_potential_reward_model import (
    ScalarPotentialHead,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

TrainStep = Callable[
    [nn.Module, tuple[torch.Tensor, ...]],
    tuple[torch.Tensor, dict[str, float]],
]
Evaluate = Callable[[nn.Module], dict[str, float]]


def load_feature_shards(
    pattern: str, target_key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate feature tensors and one target field from matching shards."""
    path = Path(pattern)
    shards = sorted(path.parent.glob(path.name))
    if not shards:
        raise ValueError(f"No feature shards match {pattern}")
    payloads = [
        torch.load(shard, map_location="cpu", weights_only=False) for shard in shards
    ]
    return (
        torch.cat([payload["features"].float() for payload in payloads]),
        torch.cat([payload[target_key].float() for payload in payloads]),
    )


def fit_head(
    model: nn.Module,
    loader: DataLoader,
    cfg: DictConfig,
    checkpoint_config: dict[str, Any],
    train_step: TrainStep,
    evaluate: Evaluate,
    max_steps: int,
    eval_every: int,
    cosine_schedule: bool = False,
) -> None:
    """Run the common optimizer, evaluation, and checkpoint loop."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_steps),
        )
        if cosine_schedule
        else None
    )
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    step = 0

    def save(name: str, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": checkpoint_config,
                "metrics": metrics,
                "step": step,
            },
            output_dir / name,
        )

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(int(cfg.epochs)):
            for batch in loader:
                model.train()
                batch = tuple(value.to(device) for value in batch)
                loss, train_metrics = train_step(model, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if step % eval_every and step < max_steps:
                    continue
                metrics = evaluate(model)
                score = metrics.pop("_score")
                metrics.update(step=step, epoch=epoch + 1, **train_metrics)
                stream.write(json.dumps(metrics) + "\n")
                stream.flush()
                logger.info("%s", json.dumps(metrics))
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    save("best.pt", metrics)
                if step >= max_steps:
                    break
            if step >= max_steps:
                break

    final_metrics = evaluate(model)
    final_score = final_metrics.pop("_score")
    if final_score > best_score:
        best_metrics = {**final_metrics, "step": step}
        save("best.pt", best_metrics)
    save("final.pt", final_metrics)
    (output_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2), encoding="utf-8"
    )


def train_state_teacher(cfg: DictConfig) -> None:
    """Train the state-success teacher used to label potential examples."""
    paths = sorted(
        path for root in cfg.raw_data_paths for path in Path(str(root)).glob("*.pkl")
    )
    if cfg.get("max_episodes") is not None:
        paths = paths[: int(cfg.max_episodes)]
    if len(paths) < 2:
        raise ValueError("Teacher training needs at least two readable episodes")
    random.shuffle(paths)
    val_count = min(
        len(paths) - 1,
        max(1, int(round(len(paths) * float(cfg.val_split)))),
    )
    val_paths = set(paths[:val_count])
    splits: dict[str, list[np.ndarray]] = {
        "train_x": [],
        "train_y": [],
        "val_x": [],
        "val_y": [],
    }
    state_dim = None
    history_size = int(cfg.history_size)

    for path in paths:
        try:
            with path.open("rb") as handle:
                episode = pickle.load(handle)
        except (EOFError, pickle.UnpicklingError, OSError) as error:
            logger.warning("Skipping unreadable episode %s: %s", path, error)
            continue
        observations, _ = transition_observations(episode)
        first_success = first_success_transition(episode, len(observations))
        if first_success is not None:
            observations = observations[: first_success + 1]
        states = [
            np.asarray(
                observation["states"].detach().cpu().numpy()
                if torch.is_tensor(observation["states"])
                else observation["states"],
                dtype=np.float32,
            ).reshape(-1)
            for observation in observations
            if "states" in observation
        ]
        if not states:
            continue
        state_dim = state_dim or int(states[0].shape[0])
        if any(int(state.shape[0]) != state_dim for state in states):
            continue

        features = np.stack(
            [
                np.concatenate(
                    [
                        states[max(0, index - offset)]
                        for offset in range(history_size - 1, -1, -1)
                    ]
                ).astype(np.float32)
                for index in range(len(states))
            ]
        )
        success = bool(episode.get("success", False))
        if not success:
            targets = np.zeros(len(states), dtype=np.float32)
        elif cfg.target_mode == "discounted_success":
            targets = np.asarray(
                [
                    float(cfg.gamma) ** (len(states) - 1 - index)
                    for index in range(len(states))
                ],
                dtype=np.float32,
            )
        elif cfg.target_mode == "linear_success":
            targets = (
                np.ones(1, dtype=np.float32)
                if len(states) == 1
                else np.linspace(0.0, 1.0, len(states), dtype=np.float32)
            )
        else:
            raise ValueError(f"Unsupported target mode: {cfg.target_mode}")

        prefix = "val" if path in val_paths else "train"
        splits[f"{prefix}_x"].append(features)
        splits[f"{prefix}_y"].append(targets)

    if state_dim is None or any(not values for values in splits.values()):
        raise ValueError("Failed to build non-empty teacher train/eval splits")
    arrays = {key: np.concatenate(values) for key, values in splits.items()}
    mean = arrays["train_x"].mean(axis=0)
    std = np.maximum(arrays["train_x"].std(axis=0), 1e-6)
    train_x = torch.from_numpy(((arrays["train_x"] - mean) / std).astype(np.float32))
    val_x = torch.from_numpy(((arrays["val_x"] - mean) / std).astype(np.float32))
    train_y = torch.from_numpy(arrays["train_y"].astype(np.float32))
    val_y = torch.from_numpy(arrays["val_y"].astype(np.float32))
    checkpoint_config = {
        "state_dim": state_dim,
        "history_size": history_size,
        "hidden_dim": int(cfg.hidden_dim),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "gamma": float(cfg.gamma),
        "target_mode": str(cfg.target_mode),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }
    model_layers = make_mlp(
        state_dim * history_size,
        [int(cfg.hidden_dim)] * int(cfg.num_layers) + [1],
        act_builder=nn.SiLU,
        last_act=False,
        use_layer_norm=True,
    )
    if cfg.dropout > 0:
        layers_with_dropout = []
        for layer in model_layers:
            layers_with_dropout.append(layer)
            if isinstance(layer, nn.SiLU):
                layers_with_dropout.append(nn.Dropout(float(cfg.dropout)))
        model_layers = layers_with_dropout
    model = nn.Sequential(*model_layers)
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )

    def train_batch(
        current_model: nn.Module, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        features, targets = batch
        loss = nn.functional.binary_cross_entropy_with_logits(
            current_model(features).squeeze(-1), targets
        )
        return loss, {"train_loss": float(loss.detach())}

    @torch.no_grad()
    def evaluate(current_model: nn.Module) -> dict[str, float]:
        current_model.eval()
        logits = torch.cat(
            [
                current_model(val_x[start : start + int(cfg.batch_size)].to(device))
                .squeeze(-1)
                .cpu()
                for start in range(0, len(val_x), int(cfg.batch_size))
            ]
        )
        loss = nn.functional.binary_cross_entropy_with_logits(logits, val_y)
        mae = torch.abs(torch.sigmoid(logits) - val_y).mean()
        return {
            "val_loss": float(loss),
            "model_potential": float(1.0 - mae),
            "_score": -float(loss),
        }

    fit_head(
        model,
        loader,
        cfg,
        checkpoint_config,
        train_batch,
        evaluate,
        int(cfg.max_steps),
        int(cfg.eval_interval),
        cosine_schedule=True,
    )


def train_scalar_head(cfg: DictConfig) -> None:
    """Train the scalar VLM potential head from frozen feature shards."""
    train_features, train_targets = load_feature_shards(
        str(cfg.train_pattern), "targets"
    )
    eval_features, eval_targets = load_feature_shards(str(cfg.eval_pattern), "targets")
    train_pairs, train_deltas = load_feature_shards(
        str(cfg.train_progress_pattern), "teacher_deltas"
    )
    eval_pairs, eval_deltas = load_feature_shards(
        str(cfg.progress_pattern), "teacher_deltas"
    )
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    model = ScalarPotentialHead(
        int(train_features.shape[-1]),
        int(cfg.hidden_dim),
        float(cfg.dropout),
    ).to(device)

    def train_batch(
        current_model: nn.Module, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        features, targets = batch
        pair_indices = torch.randint(len(train_pairs), (len(features),))
        pair_features = train_pairs[pair_indices].to(device)
        pair_targets = train_deltas[pair_indices].to(device)
        logits = current_model(features)
        value_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        permutation = torch.randperm(len(targets), device=targets.device)
        target_gap = targets - targets[permutation]
        rank_mask = target_gap.abs() >= float(cfg.rank_min_gap)
        rank_loss = (
            nn.functional.softplus(
                -torch.sign(target_gap[rank_mask])
                * (logits - logits[permutation])[rank_mask]
            ).mean()
            if rank_mask.any()
            else logits.sum() * 0.0
        )

        pair_logits = current_model(
            pair_features.reshape(-1, pair_features.shape[-1])
        ).reshape(-1, 2)
        predicted_deltas = torch.sigmoid(pair_logits[:, 1]) - torch.sigmoid(
            pair_logits[:, 0]
        )
        delta_loss = nn.functional.smooth_l1_loss(
            predicted_deltas,
            pair_targets,
            beta=float(cfg.delta_beta),
        )
        local_mask = pair_targets.abs() >= float(cfg.local_rank_min_gap)
        local_loss = (
            nn.functional.softplus(
                -torch.sign(pair_targets[local_mask])
                * (pair_logits[:, 1] - pair_logits[:, 0])[local_mask]
            ).mean()
            if local_mask.any()
            else pair_logits.sum() * 0.0
        )
        loss = (
            value_loss
            + float(cfg.rank_weight) * rank_loss
            + float(cfg.delta_weight) * delta_loss
            + float(cfg.local_rank_weight) * local_loss
        )
        return loss, {
            "train_loss": float(loss.detach()),
            "train_value_loss": float(value_loss.detach()),
            "train_delta_loss": float(delta_loss.detach()),
        }

    @torch.no_grad()
    def evaluate(current_model: nn.Module) -> dict[str, float]:
        from scipy.stats import spearmanr

        current_model.eval()

        def predict(features: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    torch.sigmoid(
                        current_model(
                            features[start : start + int(cfg.eval_batch_size)].to(
                                device
                            )
                        )
                    ).cpu()
                    for start in range(0, len(features), int(cfg.eval_batch_size))
                ]
            )

        def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
            value = spearmanr(left.numpy(), right.numpy()).statistic
            return float(value) if np.isfinite(value) else 0.0

        values = predict(eval_features)
        pair_values = predict(eval_pairs.reshape(-1, eval_pairs.shape[-1])).reshape(
            -1, 2
        )
        predicted_deltas = pair_values[:, 1] - pair_values[:, 0]
        potential_correlation = correlation(values, eval_targets)
        delta_correlation = correlation(predicted_deltas, eval_deltas)
        return {
            "model_potential": float(1.0 - torch.abs(values - eval_targets).mean()),
            "potential_spearman": potential_correlation,
            "delta_spearman": delta_correlation,
            "_score": potential_correlation + delta_correlation,
        }

    fit_head(
        model,
        loader,
        cfg,
        {
            "input_dim": int(train_features.shape[-1]),
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
        },
        train_batch,
        evaluate,
        int(cfg.epochs) * len(loader),
        int(cfg.eval_interval) * len(loader),
    )


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> None:
    """Run the selected auxiliary training stage."""
    logger.info(
        "%s",
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2),
    )
    stage = str(cfg.auxiliary.stage)
    stage_cfg = cfg.auxiliary.get(stage)
    stages = {
        "teacher": train_state_teacher,
        "scalar_head": train_scalar_head,
    }
    if stage_cfg is None or stage not in stages:
        raise ValueError(f"Unsupported or missing auxiliary stage: {stage}")
    random.seed(int(stage_cfg.seed))
    np.random.seed(int(stage_cfg.seed))
    torch.manual_seed(int(stage_cfg.seed))
    stages[stage](stage_cfg)


if __name__ == "__main__":
    main()
