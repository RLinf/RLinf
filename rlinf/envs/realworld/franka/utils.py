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


def normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return q / n


# geometry
def quat_slerp(q0, q1, t):
    """
    Slerp q0 and q1 with t steps.
    This is computed by the 4-dim sphere. And it does not depend on the order of "xyzw".
    """

    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = np.dot(q0, q1)

    # ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if np.isscalar(t):
        t_arr = np.linspace(0, 1, t, dtype=float)
    else:
        t_arr = np.array(t, dtype=float)

    results = []

    # nearly identical → fallback to LERP
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


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
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


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T
