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

import numpy as np
from scipy.spatial.transform import Rotation as R


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to the principal range ``[-pi, pi)``."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def clip_euler_to_target_window(
    euler: np.ndarray,
    target_euler: np.ndarray,
    lower_euler: np.ndarray,
    upper_euler: np.ndarray,
) -> np.ndarray:
    """Clip Euler angles using the shortest wrapped delta to the target pose.

    The target-centered window remains continuous across the ``-pi``/``pi``
    boundary.

    Args:
        euler: Current Euler angles in radians.
        target_euler: Target Euler angles in radians.
        lower_euler: Lower bound expressed in the same absolute Euler convention.
        upper_euler: Upper bound expressed in the same absolute Euler convention.

    Returns:
        Clipped angles wrapped to ``[-pi, pi)``.
    """
    delta = wrap_to_pi(euler - target_euler)
    lower_delta = lower_euler - target_euler
    upper_delta = upper_euler - target_euler
    clipped_delta = np.clip(delta, lower_delta, upper_delta)
    return wrap_to_pi(target_euler + clipped_delta)


def construct_adjoint_matrix(tcp_pose: np.ndarray) -> np.ndarray:
    """Construct the adjoint matrix for an ``xyz + quaternion`` TCP pose."""
    rotation = R.from_quat(tcp_pose[3:].copy()).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose: np.ndarray) -> np.ndarray:
    """Construct a homogeneous transform from an ``xyz + quaternion`` pose."""
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T
