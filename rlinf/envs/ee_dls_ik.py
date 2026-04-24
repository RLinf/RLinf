from __future__ import annotations

import numpy as np
import torch


_SMALL_EPS = 1e-8


def quat_multiply_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply batched quaternions in wxyz convention."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions in wxyz convention."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conjugate_torch(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def normalize_quat_torch(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=_SMALL_EPS)


def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), _SMALL_EPS, None)


def axis_angle_to_quat_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    safe_angle = angle.clamp(min=_SMALL_EPS)
    axis = axis_angle / safe_angle
    half_angle = 0.5 * angle
    quat = torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=-1)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=axis_angle.device, dtype=axis_angle.dtype)
    return torch.where((angle < _SMALL_EPS).expand_as(quat), identity, quat)


def axis_angle_to_quat_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = np.clip(angle, _SMALL_EPS, None)
    axis = axis_angle / safe_angle
    half_angle = 0.5 * angle
    quat = np.concatenate([np.cos(half_angle), axis * np.sin(half_angle)], axis=-1)
    small = angle < _SMALL_EPS
    if np.any(small):
        quat = quat.copy()
        quat[small[..., 0]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=axis_angle.dtype)
    return quat


def quat_error_to_rotvec_torch(target_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    q_err = normalize_quat_torch(quat_multiply_torch(target_quat, quat_conjugate_torch(current_quat)))
    sign = torch.where(q_err[..., :1] < 0.0, -1.0, 1.0)
    q_err = q_err * sign
    vec = q_err[..., 1:]
    vec_norm = torch.norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vec_norm, q_err[..., :1].clamp(min=_SMALL_EPS))
    axis = vec / vec_norm.clamp(min=_SMALL_EPS)
    rotvec = axis * angle
    return torch.where((vec_norm < _SMALL_EPS).expand_as(rotvec), 2.0 * vec, rotvec)


def quat_error_to_rotvec_np(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    q_err = normalize_quat_np(quat_multiply_np(target_quat, quat_conjugate_np(current_quat)))
    q_err = np.where(q_err[..., :1] < 0.0, -q_err, q_err)
    vec = q_err[..., 1:]
    vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vec_norm, np.clip(q_err[..., :1], _SMALL_EPS, None))
    axis = vec / np.clip(vec_norm, _SMALL_EPS, None)
    rotvec = axis * angle
    small = vec_norm < _SMALL_EPS
    if np.any(small):
        rotvec = rotvec.copy()
        rotvec[small[..., 0]] = 2.0 * vec[small[..., 0]]
    return rotvec


def decode_delta_action_torch(
    actions: torch.Tensor,
    pos_scale: float,
    rot_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_delta = actions[:, :3].clamp(-1.0, 1.0) * pos_scale
    rot_delta = actions[:, 3:6].clamp(-1.0, 1.0) * rot_scale
    gripper = torch.where(actions[:, 6:7] >= 0.0, 1.0, -1.0)
    return pos_delta, rot_delta, gripper


def decode_delta_action_np(
    actions: np.ndarray,
    pos_scale: float,
    rot_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_delta = np.clip(actions[:, :3], -1.0, 1.0) * pos_scale
    rot_delta = np.clip(actions[:, 3:6], -1.0, 1.0) * rot_scale
    gripper = np.where(actions[:, 6:7] >= 0.0, 1.0, -1.0)
    return pos_delta, rot_delta, gripper


def solve_dls_ik_torch(
    jacobian: torch.Tensor,
    current_q: torch.Tensor,
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    damping: float,
    integration_dt: float,
    max_dq: float,
    joint_lower: torch.Tensor | None = None,
    joint_upper: torch.Tensor | None = None,
    pos_gain: float = 0.95,
    rot_gain: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_err = target_pos - current_pos
    rot_err = quat_error_to_rotvec_torch(target_quat, current_quat)
    twist = torch.cat([pos_gain * pos_err / integration_dt, rot_gain * rot_err / integration_dt], dim=-1)
    jt = jacobian.transpose(1, 2)
    eye = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype).unsqueeze(0)
    system = jacobian @ jt + (damping**2) * eye
    dq = (jt @ torch.linalg.solve(system, twist.unsqueeze(-1))).squeeze(-1)
    if max_dq > 0.0:
        dq_abs_max = dq.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
        dq = dq * torch.clamp(max_dq / dq_abs_max, max=1.0)
    q_target = current_q + dq * integration_dt
    if joint_lower is not None and joint_upper is not None:
        q_target = torch.max(torch.min(q_target, joint_upper), joint_lower)
    return q_target, twist


def solve_dls_ik_np(
    jacobian: np.ndarray,
    current_q: np.ndarray,
    current_pos: np.ndarray,
    current_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    damping: float,
    integration_dt: float,
    max_dq: float,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    pos_gain: float = 0.95,
    rot_gain: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    pos_err = target_pos - current_pos
    rot_err = quat_error_to_rotvec_np(target_quat, current_quat)
    twist = np.concatenate(
        [pos_gain * pos_err / integration_dt, rot_gain * rot_err / integration_dt],
        axis=-1,
    )
    jt = np.swapaxes(jacobian, -1, -2)
    system = jacobian @ jt + (damping**2) * np.eye(jacobian.shape[1], dtype=jacobian.dtype)[None, :, :]
    dq = np.matmul(jt, np.linalg.solve(system, twist[..., None]))[..., 0]
    if max_dq > 0.0:
        dq_abs_max = np.maximum(np.abs(dq).max(axis=-1, keepdims=True), 1.0)
        dq = dq * np.minimum(max_dq / dq_abs_max, 1.0)
    q_target = current_q + dq * integration_dt
    if joint_lower is not None and joint_upper is not None:
        q_target = np.clip(q_target, joint_lower, joint_upper)
    return q_target, twist
