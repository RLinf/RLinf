# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained normalization stats + quantile (un)normalization.

Vendored re-implementation of the pieces of ``openpi.shared.normalize`` and the
quantile branches of ``openpi.transforms.Normalize`` / ``Unnormalize`` that the
BEHAVIOR pi05 eval path uses, so the package does not depend on the installed
``openpi`` distribution. The math is kept byte-identical to upstream (verified
by a cross-check test against the installed ``openpi``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

# Matches openpi's `1e-6` denominator epsilon in the quantile (un)normalization.
_EPS = 1e-6


@dataclasses.dataclass
class NormStats:
    """Per-key normalization statistics (mean/std and 1st/99th quantiles)."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(assets_dir, asset_id) -> dict[str, NormStats]:
    """Load BEHAVIOR norm stats from exactly ``{assets_dir}/{asset_id}/norm_stats.json``.

    Both ``assets_dir`` and ``asset_id`` are required and must be non-empty — a
    blank value is not a value and must never silently fall back to bare
    (non-task-0000) stats. A missing artifact raises ``FileNotFoundError`` rather
    than resolving a different file, so the eval model factory and the SFT data
    loader always load the same canonical task-0000 distribution. The on-disk
    format is ``{"norm_stats": {key: {mean, std, q01, q99}}}``.
    """
    if assets_dir is None or not str(assets_dir).strip():
        raise FileNotFoundError(
            "BEHAVIOR norm stats require a non-empty assets_dir (no default is "
            "applied); set actor.model.openpi.assets_dir in the YAML."
        )
    if asset_id is None or not str(asset_id).strip():
        raise FileNotFoundError(
            "BEHAVIOR norm stats require a non-empty asset_id (no default is "
            "applied); set actor.model.openpi.asset_id in the YAML."
        )
    path = pathlib.Path(assets_dir).expanduser() / asset_id / "norm_stats.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"BEHAVIOR norm_stats.json not found at {path} "
            f"(assets_dir={str(assets_dir)!r}, asset_id={asset_id!r})."
        )
    data = json.loads(path.read_text())
    raw = data["norm_stats"] if "norm_stats" in data else data
    out: dict[str, NormStats] = {}
    for key, stats in raw.items():
        out[key] = NormStats(
            mean=np.asarray(stats["mean"]),
            std=np.asarray(stats["std"]),
            q01=np.asarray(stats["q01"]) if stats.get("q01") is not None else None,
            q99=np.asarray(stats["q99"]) if stats.get("q99") is not None else None,
        )
    return out


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Map ``x`` to ``[-1, 1]`` using q01/q99 (openpi quantile normalize)."""
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile normalization requires q01 and q99.")
    q01 = stats.q01[..., : x.shape[-1]]
    q99 = stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + _EPS) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Invert :func:`normalize_quantile` (openpi quantile unnormalize).

    If the stats cover fewer dims than ``x``, the trailing dims are passed
    through unchanged, matching openpi's behavior.
    """
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile unnormalization requires q01 and q99.")
    q01, q99 = stats.q01, stats.q99
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        head = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + _EPS) + q01
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + _EPS) + q01
