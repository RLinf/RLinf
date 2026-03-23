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
- Value bin special tokens (<v0>, <v1>, ...) for discretized return prediction
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

# ============================================================================
# Value bin tokens
# ============================================================================

# Default number of bins (matching paper: 201 bins for [-1, 0] range)
DEFAULT_NUM_VALUE_BINS = 201

# Token format: <v0>, <v1>, ..., <v200>
VALUE_TOKEN_PREFIX = "<v"
VALUE_TOKEN_SUFFIX = ">"


def get_value_token(bin_id: int) -> str:
    """Get the special token string for a bin ID."""
    return f"{VALUE_TOKEN_PREFIX}{bin_id}{VALUE_TOKEN_SUFFIX}"


def get_all_value_tokens(num_bins: int = DEFAULT_NUM_VALUE_BINS) -> list[str]:
    """Get list of all value bin tokens."""
    return [get_value_token(i) for i in range(num_bins)]


def parse_value_token(token: str) -> int:
    """Parse a value token string to get the bin ID."""
    if not token.startswith(VALUE_TOKEN_PREFIX) or not token.endswith(
        VALUE_TOKEN_SUFFIX
    ):
        raise ValueError(f"Invalid value token format: {token}")

    bin_str = token[len(VALUE_TOKEN_PREFIX) : -len(VALUE_TOKEN_SUFFIX)]
    return int(bin_str)


def add_value_tokens_to_tokenizer(
    tokenizer, num_bins: int = DEFAULT_NUM_VALUE_BINS
) -> dict[str, int]:
    """Add value bin tokens to a tokenizer."""
    value_tokens = get_all_value_tokens(num_bins)
    tokenizer.add_special_tokens({"additional_special_tokens": value_tokens})

    token_to_id = {
        token: tokenizer.convert_tokens_to_ids(token) for token in value_tokens
    }

    return token_to_id


# ============================================================================
# Return discretization and normalization
# ============================================================================


class ReturnDiscretizer(DataTransformFn):
    """Discretize continuous return values into bin IDs.

    Discretizes return values into N bins evenly spaced between [min, max].
    The bin ID is converted to a string token for use with language models.

    Supports optional normalization to (-1, 0) range as per the paper:
    "we normalize the values predicted to be between (-1, 0). Since we train
    on diverse tasks that have very different typical lengths, we normalize
    the values per task based on the maximum episode length of the task."
    """

    def __init__(
        self,
        num_bins: int = 201,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        norm_stats: Optional[dict[str, NormStats]] = None,
        norm_stats_path: Optional[Path] = None,
        return_key: str = "return",
        output_key: str = "return_token",
        keep_continuous: bool = True,
        normalize_to_minus_one_zero: bool = True,
    ):
        self.num_bins = num_bins
        self.return_key = return_key
        self.output_key = output_key
        self.keep_continuous = keep_continuous
        self.normalize_to_minus_one_zero = normalize_to_minus_one_zero

        # Load raw stats
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

        # Set discretization range based on normalization mode
        if self.normalize_to_minus_one_zero:
            self.return_min = -1.0
            self.return_max = 0.0
            self.norm_factor = (
                abs(self.raw_return_min) if self.raw_return_min != 0 else 1.0
            )
        else:
            self.return_min = self.raw_return_min
            self.return_max = self.raw_return_max
            self.norm_factor = 1.0

        self.bin_width = (self.return_max - self.return_min) / self.num_bins

        logger.info("ReturnDiscretizer initialized:")
        logger.info(
            f"  raw_return_min={self.raw_return_min}, raw_return_max={self.raw_return_max}"
        )
        logger.info(f"  normalize_to_minus_one_zero={self.normalize_to_minus_one_zero}")
        logger.info(f"  norm_factor={self.norm_factor}")
        logger.info(
            f"  discretization_range=({self.return_min}, {self.return_max}), num_bins={self.num_bins}"
        )

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

    def discretize(self, value: float) -> int:
        if self.normalize_to_minus_one_zero:
            value = self.normalize_value(value)

        clipped = np.clip(value, self.return_min, self.return_max)
        bin_id = int((clipped - self.return_min) / self.bin_width)
        return min(max(bin_id, 0), self.num_bins - 1)

    def undiscretize(self, bin_id: int) -> float:
        normalized = self.return_min + (bin_id + 0.5) * self.bin_width
        return normalized

    def undiscretize_to_raw(self, bin_id: int) -> float:
        normalized = self.undiscretize(bin_id)
        return self.denormalize_value(normalized)

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

        bin_id = self.discretize(raw_value)

        result = dict(data)
        result[self.output_key] = get_value_token(bin_id)
        result["return_bin_id"] = bin_id

        if self.normalize_to_minus_one_zero:
            result["return_normalized"] = self.normalize_value(raw_value)

        if not self.keep_continuous:
            del result[self.return_key]

        return result


class ReturnNormalizer(DataTransformFn):
    """Normalize return values using z-score or min-max normalization."""

    def __init__(
        self,
        norm_stats: Optional[dict[str, NormStats]] = None,
        norm_stats_path: Optional[Path] = None,
        method: str = "minmax",
        return_key: str = "return",
    ):
        self.method = method
        self.return_key = return_key

        if norm_stats is not None:
            self._load_stats(norm_stats)
        elif norm_stats_path is not None:
            stats = load_stats(Path(norm_stats_path))
            self._load_stats(stats)
        else:
            raise ValueError("Must provide either norm_stats or norm_stats_path")

    def _load_stats(self, norm_stats: dict[str, NormStats]):
        if "return" not in norm_stats:
            raise ValueError("norm_stats must contain 'return' key")

        stats = norm_stats["return"]
        self.mean = float(
            stats.mean[0] if hasattr(stats.mean, "__len__") else stats.mean
        )
        self.std = float(stats.std[0] if hasattr(stats.std, "__len__") else stats.std)
        self.min = float(stats.min[0] if hasattr(stats.min, "__len__") else stats.min)
        self.max = float(stats.max[0] if hasattr(stats.max, "__len__") else stats.max)

        if stats.q01 is not None and stats.q99 is not None:
            self.q01 = float(
                stats.q01[0] if hasattr(stats.q01, "__len__") else stats.q01
            )
            self.q99 = float(
                stats.q99[0] if hasattr(stats.q99, "__len__") else stats.q99
            )
        else:
            self.q01 = self.min
            self.q99 = self.max

    def normalize(self, value: float) -> float:
        if self.method == "zscore":
            return (value - self.mean) / (self.std + 1e-8)
        elif self.method == "minmax":
            return (value - self.min) / (self.max - self.min + 1e-8)
        elif self.method == "quantile":
            return (value - self.q01) / (self.q99 - self.q01 + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def denormalize(self, value: float) -> float:
        if self.method == "zscore":
            return value * self.std + self.mean
        elif self.method == "minmax":
            return value * (self.max - self.min) + self.min
        elif self.method == "quantile":
            return value * (self.q99 - self.q01) + self.q01
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.return_key not in data:
            return data

        return_value = data[self.return_key]

        if isinstance(return_value, torch.Tensor):
            normalized = self.normalize(
                return_value.item()
                if return_value.numel() == 1
                else return_value.cpu().numpy().item()
            )
            result = dict(data)
            result[self.return_key] = torch.tensor(normalized, dtype=return_value.dtype)
        elif isinstance(return_value, np.ndarray):
            normalized = self.normalize(
                return_value.item()
                if return_value.size == 1
                else float(return_value.flatten()[0])
            )
            result = dict(data)
            result[self.return_key] = np.array([normalized], dtype=return_value.dtype)
        else:
            result = dict(data)
            result[self.return_key] = self.normalize(float(return_value))

        return result


def create_return_discretizer(
    num_bins: int = 201,
    norm_stats_path: Optional[str] = None,
    return_min: Optional[float] = None,
    return_max: Optional[float] = None,
    **kwargs,
) -> ReturnDiscretizer:
    """Factory function to create ReturnDiscretizer."""
    if norm_stats_path:
        return ReturnDiscretizer(
            num_bins=num_bins,
            norm_stats_path=Path(norm_stats_path),
            **kwargs,
        )
    elif return_min is not None and return_max is not None:
        return ReturnDiscretizer(
            num_bins=num_bins,
            return_min=return_min,
            return_max=return_max,
            **kwargs,
        )
    else:
        raise ValueError(
            "Must provide either norm_stats_path or (return_min, return_max)"
        )
