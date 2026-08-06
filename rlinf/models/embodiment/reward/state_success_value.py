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

"""Shared state-success teacher model used by VLM Trend preprocess scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


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


def stack_state_history(
    states: list[np.ndarray], idx: int, history_size: int
) -> np.ndarray:
    """Concatenate a trailing history window ending at ``idx``."""
    first = states[0]
    frames = []
    for offset in range(history_size - 1, -1, -1):
        hist_idx = idx - offset
        frames.append(states[hist_idx] if hist_idx >= 0 else first)
    return np.concatenate(frames, axis=0).astype(np.float32)


def load_value_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[StateSuccessValue, dict[str, Any], np.ndarray, np.ndarray]:
    """Load a trained ``StateSuccessValue`` teacher and its normalization stats.

    Returns:
        ``(model, config, mean, std)`` with the model in eval mode on ``device``.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = StateSuccessValue(
        input_dim=int(cfg["state_dim"]) * int(cfg["history_size"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(cfg["mean"], dtype=np.float32)
    std = np.asarray(cfg["std"], dtype=np.float32)
    return model, cfg, mean, std


def score_states(
    model: StateSuccessValue,
    cfg: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    states: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Score each state with the teacher after history stacking and whitening.

    Returns:
        Per-timestep success probabilities of shape ``(T,)``.
    """
    history_size = int(cfg["history_size"])
    inputs = np.stack(
        [stack_state_history(states, idx, history_size) for idx in range(len(states))],
        axis=0,
    )
    inputs = (inputs - mean[None, :]) / std[None, :]
    scores = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


__all__ = [
    "StateSuccessValue",
    "load_value_model",
    "score_states",
    "stack_state_history",
]
