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

"""Signed-stride ↔ bin-index mapping for STEAM binary value learning.

This is the label-encoding math shared by the data pipeline (which turns a
sampled frame stride into a long bin-index target) and the model/decoder side
(which turns predicted bin probabilities back into an expected signed stride).
It is intentionally free of any dataset / I/O dependency so both sides can
import it without pulling in LeRobot.

Bin layout (binary / multi-bin mode):
    * ``num_bins`` is even and ``2K`` is an integer multiple of ``num_bins``,
      so every bin owns the same number of consecutive signed strides.
    * Bins ``[0, num_bins // 2)`` are regressive (negative strides) and bins
      ``[num_bins // 2, num_bins)`` are progressive (positive strides).
    * The binary degenerate case ``num_bins == 2`` reduces to
      ``0 = regress``, ``1 = progress``.
"""

import numpy as np
import torch


def _signed_stride_to_bin(stride: int, K: int, num_bins: int) -> int:
    """Map a signed stride in ``{-K,...,-1,1,...,K}`` to a bin index.

    Layout:
        * ``pos = stride + K``         if stride < 0  (pos ∈ [0, K))
        * ``pos = stride + K - 1``     if stride > 0  (pos ∈ [K, 2K))
        * ``bin_idx = (pos * num_bins) // (2 * K)``   in [0, num_bins)

    With ``num_bins`` even and ``2K % num_bins == 0`` (enforced at
    :class:`PairDataset` construction), the sign split lands exactly at
    ``num_bins // 2``: bins ``[0, num_bins // 2)`` are regressive and
    bins ``[num_bins // 2, num_bins)`` are progressive.

    Raises:
        ValueError: if ``stride == 0`` (sampling excludes zero strides)
            or if ``abs(stride) > K``.
    """
    if stride == 0:
        raise ValueError(
            "_signed_stride_to_bin does not accept stride == 0; the multi-bin "
            "sampling path skips i == 0."
        )
    if abs(stride) > K:
        raise ValueError(
            f"_signed_stride_to_bin requires |stride| <= K, got stride={stride}, K={K}."
        )
    pos = stride + K if stride < 0 else stride + K - 1
    return int((pos * num_bins) // (2 * K))


def _scaled_signed_stride_to_bin(scaled_stride: float, K: int, num_bins: int) -> int:
    """Bin a length-scaled signed stride over ``[-K, K]`` (bin width ``2K/num_bins``).

    The scaled stride ``signed_stride * L_max / L_ep`` is rounded to the nearest
    integer-equivalent stride, clamped to ``[-K, K] \\ {0}`` with the sign
    preserved, then mapped through :func:`_signed_stride_to_bin` — the *same*
    layout (and resolution) as the unscaled path. Consequences:

    * An episode of length ``L_max`` (``scale == 1``) reproduces the unscaled
      bins exactly.
    * Shorter episodes (``scale > 1``) push a fixed frame stride into a higher
      progress bin and saturate at ``±K`` earlier — a stride covering fraction
      ``K / L_max`` of its episode already hits the extreme bin.
    * Longer episodes (``scale < 1``) under-reach: their max stride lands below
      the extreme bin.

    Only the position on the ``±K`` axis is length-dependent; the regress/
    progress split at ``half`` is preserved, so the model loss and the
    bin-index decoders (``_predicted_signed_value``) stay unchanged.
    """
    s = int(round(float(scaled_stride)))
    if s == 0:  # a non-zero stride must keep a direction, never "no progress"
        s = 1 if scaled_stride > 0 else -1
    s = max(-int(K), min(int(K), s))
    return _signed_stride_to_bin(s, int(K), num_bins)


def bin_centers(K: int, num_bins: int) -> np.ndarray:
    """Return the ``[num_bins]`` signed-stride centers for the bin layout.

    Each bin owns a contiguous set of ``strides_per_bin = 2K / num_bins``
    signed strides; the center is their arithmetic mean. By construction
    :func:`_signed_stride_to_bin` maps every stride into the bin whose
    center is closest (for even ``strides_per_bin`` the boundary ties
    are absorbed by the half-integer offsets).

    Examples:
        * ``K=8, num_bins=8``  → ``[-7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5]``
        * ``K=4, num_bins=4``  → ``[-3.5, -1.5, 1.5, 3.5]``
        * ``K=K, num_bins=2``  → ``[-K/2, K/2]`` — binary degenerate:
          :math:`E[s] / K = 2 \\cdot p_\\text{progress} - 1`, matching
          the existing ``2·P − 1`` signed-confidence derivation.

    Raises:
        ValueError: ``num_bins`` not even or ``2K % num_bins != 0``.
    """
    if num_bins < 2 or num_bins % 2 != 0:
        raise ValueError(f"num_bins must be >= 2 and even, got {num_bins}")
    if (2 * K) % num_bins != 0:
        raise ValueError(
            f"bin_centers requires 2*K to be a multiple of num_bins; "
            f"got K={K}, num_bins={num_bins} (2K={2 * K})."
        )
    strides_per_bin = (2 * K) // num_bins
    half = num_bins // 2
    # Regressive bins: cover signed strides [-K, -1] in order.
    # Progressive bins: cover signed strides [1, K] in order.
    # Center of a regressive bin b ∈ [0, half): midpoint of its
    # strides_per_bin consecutive strides starting at -K + b * strides_per_bin.
    # Center of a progressive bin b ∈ [half, num_bins): midpoint starting
    # at 1 + (b - half) * strides_per_bin.
    centers = np.empty(num_bins, dtype=np.float32)
    for b in range(num_bins):
        if b < half:
            low = -K + b * strides_per_bin
        else:
            low = 1 + (b - half) * strides_per_bin
        high = low + strides_per_bin - 1
        centers[b] = (low + high) / 2.0
    return centers


def expected_signed_stride(probs, K: int, num_bins: int):
    """Return ``E[s] = Σ_b probs[..., b] * bin_centers[b]``.

    Backend-polymorphic: if ``probs`` is a :class:`torch.Tensor` the
    computation stays on the input's device / dtype; otherwise falls
    back to numpy. The last dim of ``probs`` must equal ``num_bins``.

    For the binary degenerate case ``num_bins == 2``, equals
    ``K * (probs[..., 1] - probs[..., 0]) = K * (2·p_progress - 1)``.
    Dividing by ``K`` gives a ``[-1, 1]``-range signed confidence score
    consistent with the cumulative-progress integrator used in the
    visualize script.
    """
    centers_np = bin_centers(K, num_bins)
    if isinstance(probs, torch.Tensor):
        if probs.shape[-1] != num_bins:
            raise ValueError(
                f"probs last dim must be num_bins={num_bins}, got {tuple(probs.shape)}"
            )
        centers_t = torch.as_tensor(centers_np, dtype=probs.dtype, device=probs.device)
        return (probs * centers_t).sum(dim=-1)
    probs_np = np.asarray(probs)
    if probs_np.shape[-1] != num_bins:
        raise ValueError(
            f"probs last dim must be num_bins={num_bins}, got {probs_np.shape}"
        )
    return (probs_np * centers_np).sum(axis=-1)


def entropy_nats(probs) -> np.ndarray:
    """Per-sample categorical entropy in nats; ``[..., num_bins]`` -> ``[...]``.

    Probabilities are clipped to ``[1e-12, 1]`` before the ``-Σ p·log p`` sum so
    a zero bin never produces a ``nan``. Used by the advantage pipeline to log
    aggregate and per-member predictive entropy alongside the signed score.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


__all__ = [
    "_signed_stride_to_bin",
    "_scaled_signed_stride_to_bin",
    "bin_centers",
    "entropy_nats",
    "expected_signed_stride",
]
