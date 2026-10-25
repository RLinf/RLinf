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

"""Reference clips and the 680-d goal. The 9-d anchor is not built here."""

from __future__ import annotations

from pathlib import Path

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    CONTROL_FPS,
    DT,
    G1_JOINT_NAMES,
    GOAL_FRAME_DIM,
    GOAL_FRAMES,
    GOAL_REF_DIM,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.features import (
    default_state,
    heading_yaw,
    projected_gravity,
    quat_mul,
    quat_to_rotmat,
    quat_to_rotvec,
    rotmat_to_6d,
    yaw_rotation,
)


def _frame_goal(state: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
    yaw = heading_yaw(state[..., 3:7])
    r_heading = yaw_rotation(yaw)
    lin_h = torch.einsum("...ji,...j->...i", r_heading, state[..., ROOT_LIN_VEL])
    return torch.cat(
        (
            state[..., JOINT_POS] - q0,
            state[..., JOINT_VEL],
            projected_gravity(state[..., 3:7]),
            state[..., ROOT_ANG_VEL],
            lin_h,
            state[..., 2:3],
        ),
        dim=-1,
    )


class MotionLibrary:
    """Stores clips as [N, T_n, 71] packed into a ragged list."""

    def __init__(self, clips: list[torch.Tensor], q0: torch.Tensor | None = None):
        if not clips:
            raise ValueError("MotionLibrary needs at least one clip")
        self.clips = [c.float() for c in clips]
        self.lengths = torch.tensor([c.shape[0] for c in self.clips], dtype=torch.long)
        width = int(self.clips[0].shape[-1])
        longest = int(self.lengths.max())
        packed = torch.zeros(
            len(self.clips),
            longest,
            width,
            dtype=torch.float32,
            device=self.clips[0].device,
        )
        for i, clip in enumerate(self.clips):
            packed[i, : clip.shape[0]] = clip
        self._packed = packed
        self.q0 = (
            q0.float()
            if q0 is not None
            else torch.zeros(ACTION_DIM, dtype=torch.float32)
        )

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    def clip_length(self, motion_id: torch.Tensor) -> torch.Tensor:
        return self.lengths.to(motion_id.device)[motion_id]

    def state_at(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Gather reference states without a Python loop.

        Indexing stays on the stored device. Only the gathered rows move to
        the query device, so a CPU library is not copied onto the GPU.
        """
        squeezed = motion_id.dim() == 0
        if squeezed:
            motion_id = motion_id.unsqueeze(0)
            time = time.unsqueeze(0)
        query_device = motion_id.device
        store = self._packed.device
        mid = motion_id.to(device=store).long()
        index = time.to(device=store).long().clamp_min(0)
        lengths = self.lengths.to(device=store)
        index = torch.minimum(index, lengths[mid] - 1)
        out = self._packed[mid, index]
        if out.device != query_device:
            out = out.to(device=query_device)
        return out.squeeze(0) if squeezed else out

    def goal_ref(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """680-d reference: frames t .. t+9, last frame held past the clip end."""
        q0 = self.q0.to(device=motion_id.device)
        offsets = torch.arange(GOAL_FRAMES, device=motion_id.device)
        times = time.long().unsqueeze(-1) + offsets
        ids = motion_id.long().unsqueeze(-1).expand_as(times)
        states = self.state_at(ids.reshape(-1), times.reshape(-1))
        states = states.view(*times.shape, states.shape[-1])
        goal = _frame_goal(states, q0.view(*([1] * (states.dim() - 1)), -1))
        goal = goal.reshape(*motion_id.shape, -1)
        if goal.shape[-1] != GOAL_REF_DIM:
            raise RuntimeError(
                f"goal_ref produced {goal.shape[-1]} dims, expected {GOAL_REF_DIM}"
            )
        return goal

    def goal730(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """730-d goal: 10 future frames, 73 numbers each.

        This follows ``MotionLib.goal`` in zita-ch/supermimic_vibed, not the
        68-d paper frame. Frames are ``t+1`` through ``t+10``, and a request
        past the clip repeats the last frame. Position and yaw-removed
        rotation are expressed in the current reference heading. Linear
        velocity is in that future root frame; angular velocity is already
        stored in the root frame. Joint velocity is scaled by 0.1, as in
        that implementation. Joint angles are the raw reference, not
        ``q - q0``.
        """
        squeezed = motion_id.dim() == 0
        if squeezed:
            motion_id = motion_id.unsqueeze(0)
            time = time.unsqueeze(0)
        offsets = torch.arange(1, GOAL_FRAMES + 1, device=motion_id.device)
        times = time.long().unsqueeze(-1) + offsets
        length = self.lengths.to(device=motion_id.device)[motion_id.long()]
        times = torch.minimum(times, (length - 1).unsqueeze(-1)).clamp_min(0)
        current = self.state_at(motion_id, time.long())
        flat_id = motion_id.long().unsqueeze(-1).expand_as(times).reshape(-1)
        future = self.state_at(flat_id, times.reshape(-1)).view(*times.shape, STATE_DIM)
        heading = yaw_rotation(heading_yaw(current[..., ROOT_QUAT]))
        heading_t = heading.transpose(-1, -2)
        delta = future[..., :3] - current[..., :3].unsqueeze(-2)
        rel_pos = torch.matmul(heading_t.unsqueeze(-3), delta.unsqueeze(-1)).squeeze(-1)
        future_rot = quat_to_rotmat(future[..., ROOT_QUAT])
        rel_rot = rotmat_to_6d(torch.matmul(heading_t.unsqueeze(-3), future_rot))
        lin_local = torch.matmul(
            future_rot.transpose(-1, -2), future[..., ROOT_LIN_VEL].unsqueeze(-1)
        ).squeeze(-1)
        frame = torch.cat(
            (
                rel_pos,
                rel_rot,
                lin_local,
                future[..., ROOT_ANG_VEL],
                future[..., JOINT_POS],
                future[..., JOINT_VEL] * 0.1,
            ),
            dim=-1,
        )
        goal = frame.reshape(frame.shape[0], -1)
        if goal.shape[-1] != 730:
            raise RuntimeError(f"goal730 produced {goal.shape[-1]} dims, expected 730")
        return goal.squeeze(0) if squeezed else goal

    def reference_velocity_command(
        self, motion_id: torch.Tensor, time: torch.Tensor, window: int = 25
    ) -> torch.Tensor:
        """Centered mean of heading-frame ``(vx, vy)`` and world yaw rate.

        Appendix G uses 25 frames. The window is clipped to the clip, so the
        first and last frames average fewer than 25 samples only at the edge
        where the index is repeated by the clamp; the mean is still over 25
        slots, with the end frame held.
        """
        half = window // 2
        offsets = torch.arange(-half, half + 1, device=motion_id.device)
        times = time.long().unsqueeze(-1) + offsets
        length = self.lengths.to(device=motion_id.device)[motion_id.long()]
        times = torch.minimum(times, (length - 1).unsqueeze(-1)).clamp_min(0)
        flat_id = motion_id.long().unsqueeze(-1).expand_as(times).reshape(-1)
        states = self.state_at(flat_id, times.reshape(-1)).view(*times.shape, STATE_DIM)
        quat = states[..., ROOT_QUAT]
        heading_t = yaw_rotation(heading_yaw(quat)).transpose(-1, -2)
        planar = torch.matmul(
            heading_t, states[..., ROOT_LIN_VEL].unsqueeze(-1)
        ).squeeze(-1)[..., :2]
        world_ang = torch.matmul(
            quat_to_rotmat(quat), states[..., ROOT_ANG_VEL].unsqueeze(-1)
        ).squeeze(-1)
        command = torch.cat((planar, world_ang[..., 2:3]), dim=-1)
        return command.mean(dim=-2)

    def past_clip_end(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """True once ``time`` is past the last stored frame.

        ``state_at`` still returns that last frame, so a terminal step holds
        the reference instead of reading off the end of the clip.
        """
        return time.long() >= self.clip_length(motion_id)


def synthetic_clips(
    num_clips: int = 40,
    clip_len: int = 200,
    device: torch.device | str = "cpu",
) -> MotionLibrary:
    """Deterministic stand-in for LAFAN1, used by analytic smoke and tests."""
    clips = []
    t = torch.arange(clip_len, dtype=torch.float32)
    for i in range(num_clips):
        state = default_state(clip_len, "cpu", torch.float32)
        freq = 0.04 + 0.01 * (i % 5)
        state[:, 0] = 0.02 * t
        state[:, JOINT_POS] = 0.25 * torch.sin(2 * torch.pi * freq * t).unsqueeze(-1)
        state[:, JOINT_VEL] = (
            0.25 * 2 * torch.pi * freq * torch.cos(2 * torch.pi * freq * t)
        ).unsqueeze(-1)
        clips.append(state.to(device))
    return MotionLibrary(clips)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _slerp(q0: torch.Tensor, q1: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    # Fall back to normalized lerp when the quaternions are nearly identical.
    close = (dot > 0.9995).squeeze(-1)
    u = u.unsqueeze(-1)
    lerped = torch.nn.functional.normalize((1.0 - u) * q0 + u * q1, dim=-1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-8)
    slerped = (
        torch.sin((1.0 - u) * omega) * q0 + torch.sin(u * omega) * q1
    ) / sin_omega
    return torch.where(close.unsqueeze(-1), lerped, slerped)


def resample_states(state: torch.Tensor, src_fps: float, dst_fps: float = CONTROL_FPS) -> torch.Tensor:
    """Linear positions, slerp quaternions, velocities from the new samples."""
    count = int(state.shape[0])
    if count < 2:
        raise ValueError("a reference clip needs at least two frames to resample")
    duration = (count - 1) / float(src_fps)
    new_count = int(round(duration * float(dst_fps))) + 1
    src_t = torch.linspace(0.0, duration, count)
    dst_t = torch.linspace(0.0, duration, new_count)
    idx = torch.searchsorted(src_t, dst_t, right=True).clamp(1, count - 1)
    t0 = src_t[idx - 1]
    t1 = src_t[idx]
    u = (dst_t - t0) / (t1 - t0).clamp_min(1e-8)
    u_col = u.unsqueeze(-1)
    out = torch.empty(new_count, STATE_DIM, dtype=state.dtype)
    out[:, :3] = (1.0 - u_col) * state[idx - 1, :3] + u_col * state[idx, :3]
    out[:, JOINT_POS] = (1.0 - u_col) * state[idx - 1, JOINT_POS] + u_col * state[idx, JOINT_POS]
    out[:, ROOT_QUAT] = _slerp(state[idx - 1, ROOT_QUAT], state[idx, ROOT_QUAT], u)
    dt = 1.0 / float(dst_fps)
    out[1:, ROOT_LIN_VEL] = (out[1:, :3] - out[:-1, :3]) / dt
    out[0, ROOT_LIN_VEL] = out[1, ROOT_LIN_VEL]
    out[1:, JOINT_VEL] = (out[1:, JOINT_POS] - out[:-1, JOINT_POS]) / dt
    out[0, JOINT_VEL] = out[1, JOINT_VEL]
    delta = quat_mul(_quat_inv(out[:-1, ROOT_QUAT]), out[1:, ROOT_QUAT])
    out[1:, ROOT_ANG_VEL] = quat_to_rotvec(delta) / dt
    out[0, ROOT_ANG_VEL] = out[1, ROOT_ANG_VEL]
    return out


def decode_motion_payload(payload, source: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Check one G1 reference clip and return ``(state [T, 71], q0 [29])``.

    A bare tensor is rejected: it does not say which joint is which, or
    whether the frames are 50 Hz. ``fps`` must be 50, or 30 so the clip can
    be resampled onto the control rate. ``joint_names`` must equal
    ``G1_JOINT_NAMES``. ``q0`` is the default pose subtracted in the goal.
    """
    if torch.is_tensor(payload):
        raise ValueError(
            f"{source} is a bare state tensor. A G1 reference needs fps, "
            f"joint_names ({ACTION_DIM} names), q0, and state."
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must be a dict with fps, joint_names, q0, and state")
    names = [str(name) for name in payload.get("joint_names", ())]
    if names != list(G1_JOINT_NAMES):
        raise ValueError(
            f"{source} joint order does not match the G1 29-dof list. "
            f"First mismatch or length {len(names)} vs {ACTION_DIM}."
        )
    if "fps" not in payload or "q0" not in payload or "state" not in payload:
        raise ValueError(f"{source} is missing fps, q0, or state")
    fps = float(payload["fps"])
    q0 = torch.as_tensor(payload["q0"], dtype=torch.float32).reshape(-1)
    state = torch.as_tensor(payload["state"], dtype=torch.float32)
    if q0.shape != (ACTION_DIM,):
        raise ValueError(f"{source} q0 has shape {tuple(q0.shape)}, expected ({ACTION_DIM},)")
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(f"{source} state has shape {tuple(state.shape)}, expected [T, {STATE_DIM}]")
    if not torch.isfinite(state).all() or not torch.isfinite(q0).all():
        raise ValueError(f"{source} has non-finite state or q0")
    quat_norm = state[:, ROOT_QUAT].norm(dim=-1)
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-2):
        raise ValueError(f"{source} root quaternion is not normalized")
    if abs(fps - CONTROL_FPS) < 1e-3:
        decoded = state
    elif abs(fps - 30.0) < 1e-3:
        decoded = resample_states(state, 30.0, CONTROL_FPS)
    else:
        raise ValueError(
            f"{source} fps is {fps}. Control is {CONTROL_FPS} Hz. "
            "Only a 30 Hz retarget source is resampled; other rates are rejected."
        )
    return decoded, q0


def load_motion_library(
    path: str | None,
    num_clips: int,
    device,
    allow_synthetic: bool = False,
) -> MotionLibrary:
    """Load verified ``clip_*.pt`` payloads.

    Synthetic clips exist only for the analytic backend. A LAFAN1 run with a
    missing, short, or unchecked directory raises instead of training on noise.
    """
    directory = Path(path) if path else None
    files = sorted(directory.glob("clip_*.pt")) if directory and directory.is_dir() else []
    if files:
        if len(files) < num_clips:
            raise FileNotFoundError(
                f"{directory} has {len(files)} clip_*.pt files, need {num_clips}. "
                "Refusing to substitute synthetic clips."
            )
        clips = []
        q0 = None
        for file in files[:num_clips]:
            payload = torch.load(file, map_location="cpu", weights_only=True)
            state, clip_q0 = decode_motion_payload(payload, str(file))
            if q0 is None:
                q0 = clip_q0
            elif not torch.allclose(q0, clip_q0, atol=1e-5):
                raise ValueError(f"{file} default pose does not match the other clips")
            clips.append(state.to(device))
        return MotionLibrary(clips, q0=q0)
    if allow_synthetic:
        return synthetic_clips(num_clips=num_clips, device=device)
    raise FileNotFoundError(
        "No verified G1 reference clips at motion_path. Each clip_*.pt must "
        "contain fps, the 29 G1 joint names, the default pose q0, and state "
        "[T, 71]. Refusing to substitute synthetic clips."
    )


# Touch unused imports that document the frame layout.
_ = (DT, GOAL_FRAME_DIM, STATE_DIM, quat_to_rotmat)
