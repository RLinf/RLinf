# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tianji M6 kinematics and the 16D joint contract (rtj joint Stage2).

The vendor SDK forward kinematics (``NativeKinematics.matrix``, libKine) is the
modified-DH chain below: 200 random joint configurations of either arm match it
to 2e-7 mm and 5e-10 in rotation (2026-10-27). It is written here in torch so
the actor's joint actions reach the critic's TCP action space differentiably,
and so the TCP side of every stored joint action is the native FK exactly.

Spaces (per arm A then B):
- joint16: J1..J7 [rad], commanded closure [0, 1]; the joint VLA's action and
  state semantics (``tianji_joint.joint_state``).
- tcp20: learned TCP at flange +Z 160 mm [m], rot6 (first two rotation
  columns), commanded closure; the absolute TCP semantics of the 20D contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# torch is imported inside the torch functions: the contract and the numpy FK
# also serve CPU-only launcher processes (preflight imports the backend).

JOINT_DIM = 16
TCP_DIM = 20
JOINT_DIMS = tuple(d for a0 in (0, 8) for d in range(a0, a0 + 7))
GRIPPER_DIMS = (7, 15)
TCP_POSE_DIMS = tuple(d for d in range(TCP_DIM) if d not in (9, 19))
TCP_FLANGE_Z_M = 0.160
JOINT_SEMANTICS = "tianji.absolute_joint16.v1"
# rtj joint Stage2 transition rows carry both spaces; this names each field.
JOINT_ROW_SEMANTICS = "tianji.rtj_joint16_tcp20.v1"
JOINT_ACTION_SPACES = {
    "state": "tcp20 physical: learned TCP [m], rot6, commanded closure",
    "next_state": "tcp20 physical",
    "state_joint": "joint16 physical: A J1-7 [rad], A closure, B J1-7, B closure (measured joints)",
    "next_state_joint": "joint16 physical",
    "action": "joint16 RL frame (TianjiJointContract): the commanded joint target",
    "action_tcp": "tcp20 RL frame (TianjiContract): native-equivalent FK of action",
    "exec_action": "tcp20 RL frame: measured executed TCP (critic input)",
    "exec_action_joint": "joint16 RL frame: commanded joint step, else measured joints",
    "intervene_action": "joint16 RL frame: commanded teleop joints, else measured (actor BC)",
    "intervene_action_tcp": "tcp20 RL frame: measured TCP of the takeover step",
}

# Modified DH (Craig): alpha [deg], a [mm], d [mm], theta offset [deg]. Rows 0-6
# are J1-J7, row 7 is the fixed flange. Both arms share it (arm-local frames).
DH = (
    (0.0, 0.0, 174.5, 0.0),
    (90.0, 0.0, 0.0, 0.0),
    (-90.0, 0.0, 287.0, 0.0),
    (90.0, 18.0, 0.0, 180.0),
    (90.0, 18.0, 314.0, 180.0),
    (90.0, 0.0, 0.0, 90.0),
    (90.0, 0.0, 0.0, 90.0),
    (90.0, 0.0, 95.0, 90.0),
)
# The SDK's own joint limits (ccs_m6_40 PNVA), the owner's M6_V4_joint_limit.
JOINT_LIMITS_DEG = ((-170, 170), (-120, 120), (-170, 170), (-145, 60), (-170, 170), (-60, 60), (-90, 90))


def _joint_affine():
    lo = np.radians([lo for lo, _ in JOINT_LIMITS_DEG])
    hi = np.radians([hi for _, hi in JOINT_LIMITS_DEG])
    scale = np.concatenate([(hi - lo) / 2, [0.5]] * 2).astype(np.float32)
    bias = np.concatenate([(hi + lo) / 2, [0.5]] * 2).astype(np.float32)
    lower = np.concatenate([lo, [0.0]] * 2)
    upper = np.concatenate([hi, [1.0]] * 2)
    return scale, bias, lower, upper


JOINT_ACTION_SCALE, JOINT_ACTION_BIAS, JOINT_LOWER, JOINT_UPPER = _joint_affine()


def flange_transform(q: torch.Tensor) -> torch.Tensor:
    """[..., 7] joint angles [rad] -> [..., 4, 4] flange pose [m], arm-local base."""
    import torch

    if q.shape[-1] != 7:
        raise ValueError(f"flange_transform needs [..., 7] joints, got {tuple(q.shape)}")
    dh = torch.as_tensor(DH, dtype=q.dtype, device=q.device)
    alpha = torch.deg2rad(dh[:, 0])
    a, d = dh[:, 1] / 1000.0, dh[:, 2] / 1000.0
    theta = torch.cat([q, q.new_zeros(q.shape[:-1] + (1,))], dim=-1) + torch.deg2rad(dh[:, 3])
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = torch.cos(alpha).expand_as(ct), torch.sin(alpha).expand_as(ct)
    zero, one = torch.zeros_like(ct), torch.ones_like(ct)
    # Rx(alpha) Tx(a) Rz(theta) Tz(d)
    rows = (
        torch.stack([ct, -st, zero, a.expand_as(ct)], dim=-1),
        torch.stack([st * ca, ct * ca, -sa, -sa * d], dim=-1),
        torch.stack([st * sa, ct * sa, ca, ca * d], dim=-1),
        torch.stack([zero, zero, zero, one], dim=-1),
    )
    links = torch.stack(rows, dim=-2)  # [..., 8, 4, 4]
    pose = links[..., 0, :, :]
    for i in range(1, len(DH)):
        pose = pose @ links[..., i, :, :]
    return pose


def tcp20_from_joint16(physical: torch.Tensor) -> torch.Tensor:
    """[..., 16] joints + closures -> [..., 20] learned TCP, rot6, closures."""
    import torch

    if physical.shape[-1] != JOINT_DIM:
        raise ValueError(f"tcp20_from_joint16 needs [..., 16], got {tuple(physical.shape)}")
    parts = []
    for j0 in (0, 8):
        pose = flange_transform(physical[..., j0 : j0 + 7])
        rotation, position = pose[..., :3, :3], pose[..., :3, 3]
        parts += [
            position + TCP_FLANGE_Z_M * rotation[..., :, 2],
            rotation[..., :, 0],
            rotation[..., :, 1],
            physical[..., j0 + 7 : j0 + 8],
        ]
    return torch.cat(parts, dim=-1)


def _flange_np(q):
    """numpy twin of flange_transform for per-step use (no torch overhead)."""
    q = np.asarray(q, dtype=np.float64)
    pose = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i, (alpha, a, d, theta0) in enumerate(DH):
        theta = np.radians(theta0) + (q[..., i] if i < 7 else 0.0)
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
        link = np.zeros(q.shape[:-1] + (4, 4))
        link[..., 0, 0], link[..., 0, 1], link[..., 0, 3] = ct, -st, a / 1000.0
        link[..., 1, 0], link[..., 1, 1], link[..., 1, 2], link[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d / 1000.0
        link[..., 2, 0], link[..., 2, 1], link[..., 2, 2], link[..., 2, 3] = st * sa, ct * sa, ca, ca * d / 1000.0
        link[..., 3, 3] = 1.0
        pose = pose @ link
    return pose


def tcp20_from_joint16_np(physical) -> np.ndarray:
    """numpy [..., 16] -> float32 [..., 20], evaluated in float64."""
    physical = np.asarray(physical, dtype=np.float64)
    if physical.shape[-1] != JOINT_DIM:
        raise ValueError(f"tcp20_from_joint16_np needs [..., 16], got {physical.shape}")
    parts = []
    for j0 in (0, 8):
        pose = _flange_np(physical[..., j0 : j0 + 7])
        rotation, position = pose[..., :3, :3], pose[..., :3, 3]
        parts += [
            position + TCP_FLANGE_Z_M * rotation[..., :, 2],
            rotation[..., :, 0],
            rotation[..., :, 1],
            physical[..., j0 + 7 : j0 + 8],
        ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def tcp20_rate(joint16: torch.Tensor, qvel14: torch.Tensor) -> torch.Tensor:
    """[..., 18] time derivative of the TCP pose dims along the joint velocities."""
    import torch

    qdot = joint16.new_zeros(joint16.shape)
    qdot[..., list(JOINT_DIMS)] = qvel14.to(joint16)
    _, rate = torch.func.jvp(tcp20_from_joint16, (joint16,), (qdot,))
    return rate[..., list(TCP_POSE_DIMS)]


@dataclass(frozen=True)
class TianjiJointContract:
    """16D absolute joint actions in the RL frame [-1, 1] (rtj joint Stage2).

    Joints are normalized by the SDK's own limits (centre and half range) and
    closures as in the TCP contract (0.5 / 0.5), so every executable joint
    target is inside the frame. Decoding clips joints to those limits.
    """

    state_dim: int = JOINT_DIM
    action_dim: int = JOINT_DIM
    action_scale: np.ndarray = field(default_factory=lambda: JOINT_ACTION_SCALE.copy())
    semantics: str = JOINT_SEMANTICS

    def __post_init__(self):
        if (self.state_dim, self.action_dim) != (JOINT_DIM, JOINT_DIM):
            raise ValueError("the joint contract is state16/action16")
        if not np.array_equal(self.action_scale, JOINT_ACTION_SCALE):
            raise ValueError("the joint contract's affine scale is fixed")

    @property
    def action_bias(self):
        return JOINT_ACTION_BIAS.copy()

    def validate_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (JOINT_DIM,) or not np.isfinite(state).all():
            raise ValueError("joint state must be finite float32[16]")
        if np.any(state[list(GRIPPER_DIMS)] < 0) or np.any(state[list(GRIPPER_DIMS)] > 1):
            raise ValueError("commanded closure must be in [0,1]")
        return state

    def validate_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != (JOINT_DIM,)
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1.000001)
        ):
            raise ValueError("joint Stage2 action must be finite float32[16] in [-1,1]")
        return np.clip(action, -1, 1)

    def encode_action(self, physical):
        return (np.asarray(physical, dtype=np.float32) - JOINT_ACTION_BIAS) / JOINT_ACTION_SCALE

    def decode_action(self, action):
        physical = self.validate_action(action).astype(np.float64) * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS
        return np.clip(physical, JOINT_LOWER, JOINT_UPPER)

    def encode_clipped(self, physical):
        """RL frame of a measured/commanded joint vector, clipped into [-1, 1]."""
        return np.clip(self.encode_action(physical), -1.0, 1.0).astype(np.float32)


JOINT_CONTRACT = TianjiJointContract()


def joint_rl_to_tcp_rl(actions: torch.Tensor) -> torch.Tensor:
    """Differentiable [..., 16] joint RL frame -> [..., 20] TCP RL frame.

    The bridge from the actor's joint actions to the critic's TCP actions:
    decode with the joint contract, native-equivalent FK, encode with the TCP
    contract. Closures pass through (same 0.5 / 0.5 affine in both frames).
    """
    from .tianji_contract import DEFAULT_CONTRACT

    scale = actions.new_tensor(JOINT_ACTION_SCALE)
    bias = actions.new_tensor(JOINT_ACTION_BIAS)
    physical = actions * scale + bias
    tcp = tcp20_from_joint16(physical)
    return (tcp - actions.new_tensor(DEFAULT_CONTRACT.action_bias)) / actions.new_tensor(
        DEFAULT_CONTRACT.action_scale
    )


def joint_chunk_rl_to_tcp_rl(flat: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """[B, C*16] joint RL chunk -> [B, C*20] TCP RL chunk (differentiable)."""
    batch = flat.shape[0]
    return joint_rl_to_tcp_rl(flat.reshape(batch, chunk_len, JOINT_DIM)).reshape(batch, -1)


def joint_rl_to_tcp_rl_np(actions) -> np.ndarray:
    """numpy version of joint_rl_to_tcp_rl, evaluated in float64."""
    from .tianji_contract import DEFAULT_CONTRACT

    physical = np.asarray(actions, dtype=np.float64) * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS
    tcp = tcp20_from_joint16_np(physical).astype(np.float64)
    return ((tcp - DEFAULT_CONTRACT.action_bias) / DEFAULT_CONTRACT.action_scale).astype(np.float32)
