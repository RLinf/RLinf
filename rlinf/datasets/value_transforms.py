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

"""
Value transforms for RL datasets.

This module provides:
- Return normalization and discretization transforms for value learning
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from rlinf.datasets.lerobot.normalize import (
    NormStats,
    load_stats,
)
from rlinf.datasets.lerobot.transforms import DataTransformFn

logger = logging.getLogger(__name__)

class ReturnDiscretizer(DataTransformFn):
    """Normalize continuous return values for value model training.

    Optionally normalizes return values to the (-1, 0) range as per the paper:
    "we normalize the values predicted to be between (-1, 0). Since we train
    on diverse tasks that have very different typical lengths, we normalize
    the values per task based on the maximum episode length of the task."

    When ``normalize_to_minus_one_zero=True`` (default), outputs
    ``return_normalized`` (float in [-1, 0]) into the sample dict.
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
            self.raw_return_min = return_min
            self.raw_return_max = return_max
        elif norm_stats is not None:
            self._load_from_norm_stats(norm_stats)
        elif norm_stats_path is not None:
            stats = load_stats(Path(norm_stats_path))
            self._load_from_norm_stats(stats)
        else:
            raise ValueError(
                "Must provide either (return_min, return_max), norm_stats, or norm_stats_path"
            )

        if self.normalize_to_minus_one_zero:
            self.norm_factor = (
                abs(self.raw_return_min) if self.raw_return_min != 0 else 1.0
            )
        else:
            self.norm_factor = 1.0

        logger.info("ReturnDiscretizer initialized:")
        logger.info(
            f"  raw_return_min={self.raw_return_min}, raw_return_max={self.raw_return_max}"
        )
        logger.info(f"  normalize_to_minus_one_zero={self.normalize_to_minus_one_zero}")
        logger.info(f"  norm_factor={self.norm_factor}")

    def _load_from_norm_stats(self, norm_stats: dict[str, NormStats]):
        if "return" not in norm_stats:
            raise ValueError("norm_stats must contain 'return' key")

        return_stats = norm_stats["return"]
        self.raw_return_min = float(
            return_stats.min[0]
            if hasattr(return_stats.min, "__len__")
            else return_stats.min
        )
        self.raw_return_max = float(
            return_stats.max[0]
            if hasattr(return_stats.max, "__len__")
            else return_stats.max
        )

    def normalize_value(self, value: float) -> float:
        if not self.normalize_to_minus_one_zero:
            return value
        return value / self.norm_factor

    def denormalize_value(self, normalized: float) -> float:
        if not self.normalize_to_minus_one_zero:
            return normalized
        return normalized * self.norm_factor

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.return_key not in data:
            return data

        return_value = data[self.return_key]

        if isinstance(return_value, torch.Tensor):
            return_value = (
                return_value.item()
                if return_value.numel() == 1
                else return_value.cpu().numpy()
            )
        elif isinstance(return_value, np.ndarray):
            return_value = (
                return_value.item()
                if return_value.size == 1
                else float(return_value.flatten()[0])
            )

        raw_value = float(return_value)

        result = dict(data)

        if self.normalize_to_minus_one_zero:
            result["return_normalized"] = self.normalize_value(raw_value)

        if not self.keep_continuous:
            del result[self.return_key]

        return result
