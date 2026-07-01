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

"""Model-agnostic advantage labelling shared by offline value-model pipelines.

RECAP and STEAM produce a per-frame continuous advantage score in very different
ways (RECAP: N-step TD over a regressed value model; STEAM: worst-of-N ensemble
of a progress critic), but they turn that score into a boolean positive/negative
label with the *same* top-fraction quantile rule. This module owns that shared
post-processing — the quantile threshold and the boolean labelling — so both
pipelines (and any future value-model pipeline) keep one definition of it,
independent of how the continuous score was computed.

The only convention that differs is the comparison at the threshold: RECAP uses
``>=`` and STEAM uses ``>``; :func:`apply_boolean_label` exposes that as the
``inclusive`` flag so each caller reproduces its own behaviour verbatim.
"""

import numpy as np


def quantile_threshold(scores, positive_fraction: float) -> float:
    """Threshold whose top ``positive_fraction`` of ``scores`` is positive.

    ``threshold = percentile(scores, (1 - positive_fraction) * 100)`` — e.g.
    ``positive_fraction=0.3`` returns the 70th percentile, so the top 30% of
    scores lie at/above it. This is the exact rule shared by RECAP's
    ``positive_quantile`` and STEAM's ``rollout_quantile`` / ``expert_quantile``.

    Args:
        scores: 1-D array-like of continuous advantage scores.
        positive_fraction: Fraction of top scores to treat as positive, in
            ``(0, 1)``.

    Returns:
        The threshold value as a Python float.
    """
    return float(
        np.percentile(np.asarray(scores), (1.0 - float(positive_fraction)) * 100.0)
    )


def apply_boolean_label(continuous, threshold: float, *, inclusive: bool = True):
    """Boolean positive-advantage label from a continuous score and threshold.

    ``inclusive=True`` → ``continuous >= threshold`` (RECAP convention);
    ``inclusive=False`` → ``continuous > threshold`` (STEAM convention).

    Accepts a numpy array or pandas Series and returns the elementwise boolean
    of the same type. Callers own any data-type-specific override on top of this
    (e.g. forcing every ``sft`` frame True regardless of threshold).
    """
    if inclusive:
        return continuous >= threshold
    return continuous > threshold


__all__ = ["quantile_threshold", "apply_boolean_label"]
