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

"""Small critical-phase classifier over frozen STEAM fused features."""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PHASE_HEAD_FORMAT_VERSION = 1
RLT_PHASE_FEATURE_KEY = "rlt_gate_phase_features"


class SteamPhaseHead(nn.Module):
    """Per-member MLP heads over a frozen STEAM ensemble representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 512,
        ensemble_size: int = 1,
        dropout: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or ensemble_size < 1:
            raise ValueError(
                "input_dim, hidden_dim, and ensemble_size must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(ensemble_size)
        self.dropout = float(dropout)
        self.heads = nn.ModuleList()
        with torch.random.fork_rng():
            for member_idx in range(self.ensemble_size):
                torch.manual_seed(int(seed) + member_idx)
                self.heads.append(
                    nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_dim, 1),
                    )
                )

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(
                "STEAM phase features must have shape [E, B, D] or [B, D], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] != self.ensemble_size:
            raise ValueError(
                "STEAM phase-head ensemble mismatch: "
                f"checkpoint expects {self.ensemble_size}, got {features.shape[0]}"
            )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                "STEAM phase-head feature mismatch: "
                f"checkpoint expects {self.input_dim}, got {features.shape[-1]}"
            )
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-member logits with shape ``[E, B]``."""
        features = self._normalize_features(features)
        logits = []
        for member_idx, head in enumerate(self.heads):
            parameter = next(head.parameters())
            member_features = features[member_idx].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            logits.append(head(member_features).squeeze(-1))
        return torch.stack(logits, dim=0)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Return the mean critical-phase probability across ensemble heads."""
        return torch.sigmoid(self(features)).mean(dim=0)

    def checkpoint_payload(self, *, metadata: dict[str, Any] | None = None) -> dict:
        """Build a portable checkpoint payload."""
        return {
            "format_version": PHASE_HEAD_FORMAT_VERSION,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "ensemble_size": self.ensemble_size,
            "dropout": self.dropout,
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atomically save the phase head and calibration metadata."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(self.checkpoint_payload(metadata=metadata), temporary_path)
        os.replace(temporary_path, output_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["SteamPhaseHead", dict[str, Any]]:
        """Load a phase-head checkpoint and return its metadata."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid STEAM phase-head checkpoint at {path}")
        version = int(payload.get("format_version", -1))
        if version != PHASE_HEAD_FORMAT_VERSION:
            raise ValueError(f"Unsupported STEAM phase-head format {version} at {path}")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"STEAM phase-head checkpoint has no state_dict: {path}")
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            ensemble_size=int(payload["ensemble_size"]),
            dropout=float(payload.get("dropout", 0.1)),
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid STEAM phase-head metadata at {path}")
        return model, dict(metadata)


__all__ = [
    "PHASE_HEAD_FORMAT_VERSION",
    "RLT_PHASE_FEATURE_KEY",
    "SteamPhaseHead",
]
