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

"""Return normalization transform for value learning."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from openpi.transforms import DataTransformFn

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """Normalization statistics (compatible with OpenPI format)."""

    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None


def _dict_to_norm_stats(data: dict[str, Any]) -> NormStats:
    """Convert dictionary back to NormStats object."""
    return NormStats(
        mean=np.array(data["mean"]),
        std=np.array(data["std"]),
        q01=np.array(data["q01"]) if data.get("q01") is not None else None,
        q99=np.array(data["q99"]) if data.get("q99") is not None else None,
        min=np.array(data["min"]) if data.get("min") is not None else None,
        max=np.array(data["max"]) if data.get("max") is not None else None,
    )


def load_stats(norm_stats_path: Path) -> dict[str, NormStats]:
    """Load normalization stats from a JSON file in OpenPI format."""
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {norm_stats_path}")

    with open(norm_stats_path, "r") as f:
        data = json.load(f)

    if "norm_stats" in data:
        data = data["norm_stats"]

    return {key: _dict_to_norm_stats(stats_dict) for key, stats_dict in data.items()}


class ReturnNormalizer(DataTransformFn):
    """Normalize return values for value model training.

    Two modes controlled by ``normalize_to_minus_one_zero``:
    - True  (default): maps to (-1, 0) via ``value / abs(return_min)``
    - False: maps to (0, 1) via ``(value - return_min) / (return_max - return_min)``
    """

    def __init__(
        self,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        norm_stats: Optional[dict[str, NormStats]] = None,
        norm_stats_path: Optional[Path] = None,
        return_key: str = "return",
        keep_continuous: bool = True,
        normalize_to_minus_one_zero: bool = True,
    ):
        self.return_key = return_key
        self.keep_continuous = keep_continuous
        self.normalize_to_minus_one_zero = normalize_to_minus_one_zero

        if return_min is not None and return_max is not None:
            self.return_min = return_min
            self.return_max = return_max
        elif norm_stats is not None:
            self._load_from_norm_stats(norm_stats)
        elif norm_stats_path is not None:
            self._load_from_norm_stats(load_stats(Path(norm_stats_path)))
        else:
            raise ValueError(
                "Must provide either (return_min, return_max), norm_stats, "
                "or norm_stats_path"
            )

        logger.info(
            "ReturnNormalizer: return_min=%.4f, return_max=%.4f, range=%s",
            self.return_min,
            self.return_max,
            "(-1, 0)" if self.normalize_to_minus_one_zero else "(0, 1)",
        )

    def _load_from_norm_stats(self, norm_stats: dict[str, NormStats]):
        if "return" not in norm_stats:
            raise ValueError("norm_stats must contain 'return' key")
        rs = norm_stats["return"]
        self.return_min = float(rs.min[0] if hasattr(rs.min, "__len__") else rs.min)
        self.return_max = float(rs.max[0] if hasattr(rs.max, "__len__") else rs.max)

    def normalize_value(self, value: float) -> float:
        if self.normalize_to_minus_one_zero:
            denom = abs(self.return_min) if self.return_min != 0 else 1.0
            return value / denom
        else:
            span = self.return_max - self.return_min
            if span == 0:
                return 0.0
            return (value - self.return_min) / span

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.return_key not in data:
            return data

        raw = data[self.return_key]
        if isinstance(raw, torch.Tensor):
            raw = raw.item() if raw.numel() == 1 else raw.cpu().numpy()
        elif isinstance(raw, np.ndarray):
            raw = raw.item() if raw.size == 1 else float(raw.flatten()[0])

        result = dict(data)
        result["return_normalized"] = self.normalize_value(float(raw))

        if not self.keep_continuous:
            del result[self.return_key]

        return result
