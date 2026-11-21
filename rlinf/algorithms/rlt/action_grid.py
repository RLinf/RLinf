# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""One time-grid contract for RLT targets, environment samples and references.

Rewards and observed human labels stay on the environment clock. Networks and
the dense executor share target knots; observations must never be upsampled into
invented execution labels. This module does not import robot or runtime owners.
"""

import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionGrid:
    """Describe one chunk without conflating target knots and env samples."""

    env_steps: int
    reference_steps: int
    offset: int
    row_scale: float = 1.0
    dense: bool = False
    aligned: bool = False

    def __post_init__(self) -> None:
        if self.row_scale == 1.0:
            object.__setattr__(self, "dense", False)
        if self.env_steps < 1 or self.reference_steps < 1 or self.offset < 0:
            raise ValueError("action grid needs positive horizons and a nonnegative offset")
        if not math.isfinite(self.row_scale) or self.row_scale < 1:
            raise ValueError("action grid row_scale must be finite and >= 1")
        last = self.offset + self.target_stride * (self.target_steps - 1)
        if self.aligned and last > self.reference_steps - 1 + 1e-7:
            raise ValueError(f"target grid ends at reference row {last}, outside {self.reference_steps} rows")

    @property
    def target_steps(self) -> int:
        if self.aligned and self.dense:
            return int(math.ceil(self.row_scale * self.env_steps - 1e-9))
        return self.env_steps

    @property
    def target_stride(self) -> float:
        return 1.0 if not self.aligned or self.dense else self.row_scale

    @property
    def rate_ratio(self) -> float:
        """Target knots per environment period; used for physical-time shaping."""
        return self.row_scale / self.target_stride if self.aligned else 1.0

    @property
    def changes_grid(self) -> bool:
        return self.aligned and (self.target_steps != self.env_steps or self.target_stride != 1)

    @classmethod
    def from_config(cls, cfg: Any, *, default_offset: int | None = None) -> "ActionGrid":
        """Use resolved checkpoint/config values, with legacy defaults only."""
        get = cfg.get if hasattr(cfg, "get") else lambda key, default=None: getattr(cfg, key, default)
        steps = int(get("num_action_chunks"))
        offset = os.environ.get("RLT_REF_OFFSET", "0") if default_offset is None else default_offset
        return cls(
            env_steps=steps,
            reference_steps=int(get("ref_num_action_chunks", steps)),
            offset=int(get("action_grid_offset", offset)),
            row_scale=float(get("action_grid_scale", 1.0)),
            dense=bool(get("action_grid_dense", False)),
            aligned=bool(get("executor_action_grid", False)),
        )

    @staticmethod
    def sample(rows, positions):
        """Linearly sample B,T,D tensors, preserving gradients and exact knots."""
        import torch

        positions = torch.as_tensor(positions, device=rows.device, dtype=torch.float64)
        if rows.ndim != 3 or positions.ndim != 1:
            raise ValueError("action sampling needs B,T,D rows and 1D positions")
        if bool(((positions < -1e-7) | (positions > rows.shape[1] - 1 + 1e-7)).any()):
            raise ValueError("action sampling positions exceed the supplied target knots")
        lo = positions.floor().long().clamp(0, rows.shape[1] - 1)
        hi = (lo + 1).clamp(max=rows.shape[1] - 1)
        w = (positions - lo).to(rows.dtype).reshape(1, -1, 1)
        return rows[:, lo] * (1 - w) + rows[:, hi] * w

    def reference_targets(self, reference):
        """Reference values at precisely the knots consumed by the executor."""
        return self.sample(reference, [self.offset + self.target_stride * i for i in range(self.target_steps)])

    def reference_env(self, reference):
        """Reference on the environment clock, for observation-aligned BC."""
        stride = self.row_scale if self.aligned else 1.0
        return self.sample(reference, [self.offset + stride * i for i in range(self.env_steps)])

    def to_env(self, targets):
        """Project predictions onto label times; never interpolate labels."""
        if targets.shape[1] != self.target_steps:
            raise ValueError(f"expected {self.target_steps} target knots, got {targets.shape[1]}")
        return self.sample(targets, [self.rate_ratio * i for i in range(self.env_steps)])

    def execution_reference(self, reference, targets):
        """Reference with commanded knots, solely for the next RTC prefix."""
        if self.target_stride != 1:
            raise ValueError("corrected RTC stitching requires dense target knots")
        out = reference.clone()
        out[:, self.offset : self.offset + self.target_steps] = targets.to(out)
        return out

    def metadata(self, batch_size: int, *, device=None):
        """Persist the complete grid with each plan (schema 1, float64)."""
        import torch

        values = [1, self.env_steps, self.target_steps, self.reference_steps,
                  self.offset, self.row_scale, int(self.dense)]
        return torch.tensor(values, dtype=torch.float64, device=device).repeat(batch_size, 1)

    def validate_metadata(self, metadata) -> None:
        """Refuse replay from another speed, horizon, offset or grid schema."""
        import torch

        if not isinstance(metadata, torch.Tensor) or metadata.ndim != 2 or metadata.shape[1] != 7:
            raise ValueError("executor-grid replay needs target_action_grid metadata; convert legacy data explicitly")
        meta = metadata.to(torch.float64)
        # 11-09: schema-0 rows are placeholder blocks that carry no plan (an episode's first block,
        # record_transition=False); they are not trained on, so only planned rows are compared.
        planned = meta[:, 0] != 0
        if not bool(planned.any()):
            return
        expected = self.metadata(int(planned.sum()), device=metadata.device)
        if not torch.allclose(meta[planned], expected, rtol=0, atol=1e-9):
            raise ValueError("replay target_action_grid differs from the current model time grid")

    @classmethod
    def from_metadata(cls, metadata) -> "ActionGrid":
        """Recover and validate a homogeneous rollout batch's explicit grid."""
        if metadata.ndim != 2 or metadata.shape[1] != 7:
            raise ValueError("target_action_grid must have B,7 metadata fields")
        row = metadata[0].detach().cpu().tolist()
        grid = cls(int(row[1]), int(row[3]), int(row[4]), float(row[5]), bool(row[6]), True)
        grid.validate_metadata(metadata)
        return grid
