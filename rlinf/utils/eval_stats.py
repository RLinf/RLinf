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

"""Statistical helpers for interpreting embodied evaluation success rates.

Evaluation metrics such as ``eval/env/success_once`` are reported as the mean
of a binary (success/failure) outcome over all evaluated episodes
(see ``rlinf.utils.metric_utils.compute_evaluate_metrics``). When the number
of evaluation episodes is small, this mean is a binomial proportion and its
run-to-run fluctuation is dominated by sampling noise rather than by changes
in the underlying policy. The helpers below quantify that noise so
practitioners can tell statistical noise apart from real training
instability.
"""

from __future__ import annotations

import math

# Two-sided z-score for the most commonly used confidence levels.
Z_95 = 1.959963984540054
Z_99 = 2.5758293035489004

_Z_SCORES = {0.90: 1.6448536269514722, 0.95: Z_95, 0.99: Z_99}


def _z_score(confidence: float) -> float:
    """Return the two-sided z-score for a confidence level in (0, 1)."""
    if confidence in _Z_SCORES:
        return _Z_SCORES[confidence]
    # Normal quantile via the complementary error function inverse.
    # Acklam-style inversion is overkill here; use bisection on erfc.
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid / math.sqrt(2.0)) > (1.0 - confidence):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def validate_success_rate(success_rate: float) -> float:
    """Validate that ``success_rate`` is a proportion in [0, 1]."""
    rate = float(success_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"success_rate must be in [0, 1], got {success_rate!r}")
    return rate


def validate_num_episodes(num_episodes: int) -> int:
    """Validate that ``num_episodes`` is a positive integer."""
    episodes = int(num_episodes)
    if episodes <= 0:
        raise ValueError(f"num_episodes must be > 0, got {num_episodes!r}")
    return episodes


def success_rate_std(success_rate: float, num_episodes: int) -> float:
    """Standard deviation of a binomial success-rate estimate.

    The evaluated ``success_once`` metric is the mean of a Bernoulli
    indicator over ``num_episodes`` episodes, so its standard deviation is
    ``sqrt(p * (1 - p) / n)``.

    Args:
        success_rate: Observed mean success rate in [0, 1].
        num_episodes: Number of evaluation episodes the rate was averaged over.

    Returns:
        Standard deviation of the estimated success rate.
    """
    p = validate_success_rate(success_rate)
    n = validate_num_episodes(num_episodes)
    return math.sqrt(p * (1.0 - p) / n)


def normal_interval(
    success_rate: float, num_episodes: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wald (normal-approximation) confidence interval for a success rate.

    Simple and intuitive, but can fall outside [0, 1] for very small
    episode counts. Prefer :func:`wilson_interval` when ``num_episodes`` is
    below ~40.
    """
    p = validate_success_rate(success_rate)
    n = validate_num_episodes(num_episodes)
    z = _z_score(confidence)
    half_width = z * math.sqrt(p * (1.0 - p) / n)
    return (p - half_width, p + half_width)


def wilson_interval(
    success_rate: float, num_episodes: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial success rate.

    Unlike the Wald interval, the Wilson interval always stays within
    [0, 1] and has good coverage even for a small number of episodes,
    which is the common regime for embodied rollouts.

    Returns:
        ``(lower, upper)`` bounds of the confidence interval.
    """
    p = validate_success_rate(success_rate)
    n = validate_num_episodes(num_episodes)
    z = _z_score(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def episodes_needed_for_width(
    target_width: float, success_rate: float = 0.5, confidence: float = 0.95
) -> int:
    """Number of episodes needed so the 95%-CI half width <= ``target_width``.

    The most conservative (widest) binomial is at p = 0.5; passing the
    actually observed rate yields a tighter estimate.

    Returns:
        Minimum number of evaluation episodes (at least 1).
    """
    if target_width <= 0:
        raise ValueError(f"target_width must be > 0, got {target_width!r}")
    p = validate_success_rate(success_rate)
    z = _z_score(confidence)
    n = (z * z * p * (1.0 - p)) / (target_width * target_width)
    return max(1, math.ceil(n))


def two_proportion_z_test(
    rate_a: float,
    num_episodes_a: int,
    rate_b: float,
    num_episodes_b: int,
) -> tuple[float, float]:
    """Two-proportion z-test comparing two independent evaluation results.

    Use this to decide whether a change in ``success_once`` between two
    evaluations (e.g. consecutive validation steps) is statistically
    meaningful or likely noise.

    Returns:
        ``(z_statistic, two_sided_p_value)``.
    """
    p1 = validate_success_rate(rate_a)
    p2 = validate_success_rate(rate_b)
    n1 = validate_num_episodes(num_episodes_a)
    n2 = validate_num_episodes(num_episodes_b)

    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    std_err = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if std_err == 0.0:
        return 0.0, 1.0
    z = (p1 - p2) / std_err
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p_value


def is_within_noise(
    rate_a: float,
    num_episodes_a: int,
    rate_b: float,
    num_episodes_b: int,
    confidence: float = 0.95,
) -> bool:
    """Return True when two success rates are statistically indistinguishable."""
    z, p_value = two_proportion_z_test(rate_a, num_episodes_a, rate_b, num_episodes_b)
    alpha = 1.0 - confidence
    return p_value > alpha
