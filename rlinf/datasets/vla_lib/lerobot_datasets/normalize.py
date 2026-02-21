# Copyright 2025 The RLinf Authors.
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

"""Normalization utilities for VLA datasets.

This module provides classes for computing and applying normalization statistics,
adapted from OpenPI but compatible with our PyTorch-based data pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class NormStats:
    """Normalization statistics (compatible with OpenPI format)."""

    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None  # 1st quantile
    q99: Optional[np.ndarray] = None  # 99th quantile
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None


class RunningStats:
    """Compute running statistics of a batch of vectors (adapted from OpenPI)."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch) -> None:
        """Update the running statistics with a batch of vectors."""
        # Handle torch tensors
        if torch.is_tensor(batch):
            batch = batch.detach().cpu().numpy()
        elif not isinstance(batch, np.ndarray):
            batch = np.array(batch)

        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics as a NormStats object."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self._mean,
            std=stddev,
            q01=q01,
            q99=q99,
            min=self._min.copy(),
            max=self._max.copy(),
        )

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(
                self._min[i], self._max[i], self._num_quantile_bins + 1
            )

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[i]
            )

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def _norm_stats_to_dict(norm_stats: NormStats) -> dict[str, Any]:
    """Convert NormStats object to dictionary for JSON serialization."""
    return {
        "mean": norm_stats.mean.tolist(),
        "std": norm_stats.std.tolist(),
        "q01": norm_stats.q01.tolist() if norm_stats.q01 is not None else None,
        "q99": norm_stats.q99.tolist() if norm_stats.q99 is not None else None,
        "min": norm_stats.min.tolist() if norm_stats.min is not None else None,
        "max": norm_stats.max.tolist() if norm_stats.max is not None else None,
    }


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


def save_stats(norm_stats: dict[str, NormStats], directory: Path) -> None:
    """Save normalization stats to a directory in OpenPI format."""
    serializable_stats = {"norm_stats": {}}
    for key, stats in norm_stats.items():
        serializable_stats["norm_stats"][key] = _norm_stats_to_dict(stats)

    norm_stats_path = directory / "norm_stats.json"
    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(norm_stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"Saved norm stats to {norm_stats_path}")


def load_stats(norm_stats_path: Path) -> dict[str, NormStats]:
    """Load normalization stats from a JSON file in OpenPI format."""
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {norm_stats_path}")

    with open(norm_stats_path, "r") as f:
        data = json.load(f)

    # Handle OpenPI format with "norm_stats" wrapper
    if "norm_stats" in data:
        data = data["norm_stats"]

    return {key: _dict_to_norm_stats(stats_dict) for key, stats_dict in data.items()}


def get_mixture_norm_stats_path(
    norm_stats_dir: Path, dataset_name: str
) -> Optional[Path]:
    """Get the norm_stats path for a specific dataset in a mixture.

    Expected structure: norm_stats_dir/{dataset_name}/norm_stats.json

    Args:
        norm_stats_dir: Base norm_stats directory for the mixture
        dataset_name: Name of the dataset

    Returns:
        Path to norm_stats.json if exists, None otherwise
    """
    norm_stats_path = Path(norm_stats_dir) / dataset_name / "norm_stats.json"
    if norm_stats_path.exists():
        return norm_stats_path
    return None


def validate_mixture_norm_stats(
    norm_stats_dir: Path, dataset_names: list[str]
) -> dict[str, bool]:
    """Validate that norm_stats exist for all datasets in a mixture.

    Args:
        norm_stats_dir: Base norm_stats directory
        dataset_names: List of dataset names to check

    Returns:
        Dict mapping dataset_name -> bool (True if stats exist)
    """
    results = {}
    for name in dataset_names:
        norm_stats_path = get_mixture_norm_stats_path(norm_stats_dir, name)
        results[name] = norm_stats_path is not None
    return results


def load_mixture_norm_stats(
    norm_stats_dir: Path, dataset_names: list[str]
) -> dict[str, dict[str, NormStats]]:
    """Load normalization stats for all datasets in a mixture.

    Args:
        norm_stats_dir: Base norm_stats directory
        dataset_names: List of dataset names

    Returns:
        Dict mapping dataset_name -> {key: NormStats}
    """
    all_stats = {}
    for name in dataset_names:
        norm_stats_path = get_mixture_norm_stats_path(norm_stats_dir, name)
        if norm_stats_path:
            all_stats[name] = load_stats(norm_stats_path)
    return all_stats
