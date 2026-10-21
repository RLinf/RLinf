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

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def pose_inv(pose):
    """
    Computes the inverse of homogenous pose matrices.

    Note that the inverse of a pose matrix is the following:
    [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    Args:
        pose (np.array): batch of pose matrices with last 2 dimensions of (4, 4)


    Returns:
        inv_pose (np.array): batch of inverse pose matrices with last 2 dimensions of (4, 4)
    """
    num_axes = len(pose.shape)
    assert num_axes >= 2

    inv_pose = np.zeros_like(pose)

    # gymnastics to take transpose of last 2 dimensions
    inv_pose[..., :3, :3] = np.transpose(
        pose[..., :3, :3], tuple(range(num_axes - 2)) + (num_axes - 1, num_axes - 2)
    )

    # note: numpy matmul wants shapes [..., 3, 3] x [..., 3, 1] -> [..., 3, 1] so we add a dimension and take it away after
    inv_pose[..., :3, 3] = np.matmul(-inv_pose[..., :3, :3], pose[..., :3, 3:4])[..., 0]
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def quat2euler(
    quat: np.ndarray, order: str = "xyz", degrees: bool = False
) -> np.ndarray:
    """
    Convert a quaternion to Euler angles.

    Args:
        quat (np.ndarray): shape (4,), quaternion in [x, y, z, w] format
        order (str): rotation order, default is 'xyz'
        degrees (bool): if True, return angles in degrees; else in radians

    Returns:
        np.ndarray: shape (3,), Euler angles in specified order
    """
    assert quat.shape == (4,), "Input quaternion must have shape (4,)"
    rot = R.from_quat(quat)  # assumes [x, y, z, w]
    return rot.as_euler(order, degrees=degrees)


def pose2matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert xyz+[w,x,y,z] (7,) pose to a 4x4 homogeneous transformation matrix.
    """
    assert pose.shape == (7,)
    xyz = pose[:3]
    quat_wxyz = pose[3:]
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    )  # convert to [x,y,z,w]
    rot_mat = R.from_quat(quat_xyzw).as_matrix()  # [3,3]
    mat = np.eye(4)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = xyz
    return mat


def pose2matrix_torch(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert xyz+[w,x,y,z] (7,) pose tensor to a 4x4 homogeneous transformation matrix.

    Args:
        pose (torch.Tensor): shape (7,), in the format [x, y, z, w, qx, qy, qz]

    Returns:
        torch.Tensor: (4, 4) homogeneous transformation matrix
    """
    assert pose.shape == (7,), f"Expected pose shape (7,), got {pose.shape}"

    xyz = pose[:3]
    quat_wxyz = pose[3:]
    # Convert to [x, y, z, w]
    qx, qy, qz, qw = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]

    # Normalize quaternion for numerical stability
    norm = torch.sqrt(qx * qx + qy * qy + qz * qz + qw * qw + 1e-8)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Compute rotation matrix
    rot_mat = torch.tensor(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=pose.dtype,
        device=pose.device,
    )

    # Compose homogeneous transform
    mat = torch.eye(4, dtype=pose.dtype, device=pose.device)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = xyz

    return mat


def matrix2pose(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 transformation matrix to xyz+[w,x,y,z] (7,) pose.
    """
    assert mat.shape == (4, 4)
    xyz = mat[:3, 3]
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()  # [x, y, z, w]
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    )  # convert to [w,x,y,z]
    return np.concatenate([xyz, quat_wxyz])


def pose2matrix_batch(pose: np.ndarray) -> np.ndarray:
    """
    Convert xyz+[w,x,y,z] pose(s) to 4x4 homogeneous transformation matrix/matrices.

    Args:
        pose: (7,) or (N, 7) numpy array.

    Returns:
        mat: (4, 4) if input (7,), or (N, 4, 4) if input (N, 7).
    """
    pose = np.atleast_2d(pose)  # (N, 7)
    assert pose.shape[1] == 7

    xyz = pose[:, :3]  # (N, 3)
    quat_wxyz = pose[:, 3:]  # (N, 4)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # convert to (x, y, z, w), shape (N, 4)

    rot_mats = R.from_quat(quat_xyzw).as_matrix()  # (N, 3, 3)

    mats = np.tile(np.eye(4), (pose.shape[0], 1, 1))  # (N, 4, 4)
    mats[:, :3, :3] = rot_mats
    mats[:, :3, 3] = xyz

    if pose.shape[0] == 1:
        return mats[0]  # return (4,4) if input was (7,)
    return mats


def pose2matrix_batch_torch(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert xyz+[w,x,y,z] pose(s) to 4x4 homogeneous transformation matrix/matrices.

    Args:
        pose: (7,) or (N, 7) torch tensor.

    Returns:
        mat: (4, 4) if input (7,), or (N, 4, 4) if input (N, 7).
    """
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)  # (1, 7)
    assert pose.shape[1] == 7, f"Expected pose shape (N,7), got {pose.shape}"

    xyz = pose[:, :3]  # (N, 3)
    quat_wxyz = pose[:, 3:]  # (N, 4)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # convert to [x,y,z,w] for math consistency

    qx, qy, qz, qw = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    norm = torch.sqrt(qx * qx + qy * qy + qz * qz + qw * qw + 1e-8)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Rotation matrix (vectorized)
    rot_mats = torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qw),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx**2 + qy**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )  # (N, 3, 3)

    # Homogeneous transformation matrices
    N = pose.shape[0]
    mats = (
        torch.eye(4, dtype=pose.dtype, device=pose.device).unsqueeze(0).repeat(N, 1, 1)
    )
    mats[:, :3, :3] = rot_mats
    mats[:, :3, 3] = xyz

    if N == 1:
        return mats[0]
    return mats


def matrix2pose_batch(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 transformation matrix/matrices to xyz+[w,x,y,z] pose(s).

    Args:
        mat: (4, 4) or (N, 4, 4) numpy array.

    Returns:
        pose: (7,) if input (4,4), or (N, 7) if input (N, 4, 4).
    """
    mat = np.atleast_3d(mat)  # (N, 4, 4)
    assert mat.shape[1:] == (4, 4)

    xyz = mat[:, :3, 3]  # (N, 3)

    quat_xyzw = R.from_matrix(mat[:, :3, :3]).as_quat()  # (N, 4), (x,y,z,w)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]  # (w,x,y,z)

    pose = np.concatenate([xyz, quat_wxyz], axis=1)  # (N, 7)

    if mat.shape[0] == 1:
        return pose[0]  # return (7,) if input was (4,4)
    return pose


def matrix2pose_batch_torch(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert 4x4 transformation matrix/matrices to xyz+[w,x,y,z] pose(s).

    Args:
        mat: (4, 4) or (N, 4, 4) torch tensor.

    Returns:
        pose: (7,) if input (4,4), or (N, 7) if input (N, 4, 4).
    """
    if mat.ndim == 2:
        mat = mat.unsqueeze(0)  # (1, 4, 4)
    assert mat.shape[1:] == (4, 4), f"Expected (N,4,4), got {mat.shape}"

    # Extract translation
    xyz = mat[:, :3, 3]  # (N, 3)

    # Extract rotation matrix
    Rm = mat[:, :3, :3]  # (N, 3, 3)

    # Compute quaternion [x, y, z, w] using the standard algorithm
    qw = (
        torch.sqrt(torch.clamp(1.0 + Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2], min=0.0))
        / 2.0
    )
    qx = (
        torch.sqrt(torch.clamp(1.0 + Rm[:, 0, 0] - Rm[:, 1, 1] - Rm[:, 2, 2], min=0.0))
        / 2.0
    )
    qy = (
        torch.sqrt(torch.clamp(1.0 - Rm[:, 0, 0] + Rm[:, 1, 1] - Rm[:, 2, 2], min=0.0))
        / 2.0
    )
    qz = (
        torch.sqrt(torch.clamp(1.0 - Rm[:, 0, 0] - Rm[:, 1, 1] + Rm[:, 2, 2], min=0.0))
        / 2.0
    )

    # Handle signs based on rotation matrix elements
    qx = qx * torch.sign(Rm[:, 2, 1] - Rm[:, 1, 2] + 1e-8)
    qy = qy * torch.sign(Rm[:, 0, 2] - Rm[:, 2, 0] + 1e-8)
    qz = qz * torch.sign(Rm[:, 1, 0] - Rm[:, 0, 1] + 1e-8)

    quat_xyzw = torch.stack([qx, qy, qz, qw], dim=-1)  # (N, 4)
    quat_xyzw = quat_xyzw / (quat_xyzw.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to [w,x,y,z]
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    pose = torch.cat([xyz, quat_wxyz], dim=-1)  # (N, 7)

    if pose.shape[0] == 1:
        return pose[0]
    return pose
