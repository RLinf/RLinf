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

"""Quaternion helpers for poses given as ``xyz`` plus an ``xyzw`` quaternion."""

import numpy as np
from numpy.typing import ArrayLike


def normalize(q: ArrayLike) -> np.ndarray:
    """Return ``q`` scaled to unit length."""
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return q / n


def quat_slerp(q0: ArrayLike, q1: ArrayLike, t: float) -> np.ndarray:
    """Spherically interpolate between two quaternions."""

    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = np.dot(q0, q1)

    # Align quaternion hemispheres to follow the shortest path.
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if np.isscalar(t):
        t_arr = np.linspace(0, 1, t, dtype=float)
    else:
        t_arr = np.array(t, dtype=float)

    results = []

    # Use normalized linear interpolation for nearly identical quaternions.
    if dot > 0.9995:
        for tt in t_arr:
            q = normalize(q0 + tt * (q1 - q0))
            results.append(q)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        for tt in t_arr:
            theta = theta_0 * tt
            s0 = np.sin(theta_0 - theta) / sin_theta_0
            s1 = np.sin(theta) / sin_theta_0
            q = s0 * q0 + s1 * q1
            results.append(q)

    results = np.stack(results)
    return results
