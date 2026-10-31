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

import hashlib
import json
import math
from pathlib import Path

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    CONTROL_FPS,
    G1_JOINT_NAMES,
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
    heading_yaw,
    projected_gravity,
    quat_mul,
    quat_to_rotmat,
    quat_to_rotvec,
    rotmat_to_6d,
    yaw_rotation,
)

# First and last goal frame relative to the robot's reference time t. The
# step taken at t is scored against frame t+1, so the 680-d goal holds the
# current reference frame and nine after it; the 730-d goal holds ten frames
# after it. Checkpoints record the window they were trained with.
GOAL_WINDOW = {"paper": (0, GOAL_FRAMES - 1), "goal730": (1, GOAL_FRAMES)}


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
    """All clips in one contiguous ``[sum_T, 71]`` block.

    Clip ``i`` is rows ``offsets[i] .. offsets[i] + lengths[i] - 1``. Memory
    is ``O(sum_T)``; a short clip does not pay for the longest one. Queries
    still take a clip id and a frame index.
    """

    def __init__(self, clips: list[torch.Tensor], q0: torch.Tensor | None = None):
        if not clips:
            raise ValueError("MotionLibrary needs at least one clip, got none")
        lengths = [int(c.shape[0]) for c in clips]
        if min(lengths) < 1:
            raise ValueError(
                "every reference clip needs at least one frame; clip "
                f"{lengths.index(min(lengths))} has {min(lengths)}"
            )
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.offsets = torch.cumsum(self.lengths, dim=0) - self.lengths
        self.frames = torch.cat([c.float() for c in clips], dim=0)
        self._index_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        # Body position, rotation, linear and angular velocity of every frame.
        # The reference clip does not change, so training gathers these instead
        # of walking the kinematic tree again. Built on first use per device.
        self._kin_cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}
        # Source name per clip, for per-clip evaluation rows.
        self.clip_names = [f"clip_{i:03d}" for i in range(len(clips))]
        self.q0 = (
            q0.float()
            if q0 is not None
            else torch.zeros(ACTION_DIM, dtype=torch.float32)
        )

    @property
    def num_clips(self) -> int:
        """Number of clips."""
        return int(self.lengths.shape[0])

    @property
    def num_frames(self) -> int:
        """Total frames over all clips."""
        return int(self.frames.shape[0])

    def clip(self, motion_id: int) -> torch.Tensor:
        """One clip as a view into the block."""
        start = int(self.offsets[motion_id])
        return self.frames[start : start + int(self.lengths[motion_id])]

    def _index(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._index_cache:
            self._index_cache[device] = (
                self.offsets.to(device=device),
                self.lengths.to(device=device),
            )
        return self._index_cache[device]

    def clip_length(self, motion_id: torch.Tensor) -> torch.Tensor:
        """Frame count of each clip in ``motion_id``."""
        return self._index(motion_id.device)[1][motion_id]

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
        store = self.frames.device
        offsets, lengths = self._index(store)
        mid = motion_id.to(device=store).long()
        index = time.to(device=store).long().clamp_min(0)
        index = torch.minimum(index, lengths[mid] - 1)
        out = self.frames[offsets[mid] + index]
        if out.device != query_device:
            out = out.to(device=query_device)
        return out.squeeze(0) if squeezed else out

    def _kinematics(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._kin_cache.get(device)
        if cached is None:
            cached = self._build_kinematics(device)
            self._kin_cache[device] = cached
        return cached

    def _build_kinematics(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from rlinf.envs.sim.g1_mimic.g1_fk import g1_kinematics

        kin = g1_kinematics()
        # The eager depth walk, not ``forward``. A 4096-row ``forward`` on
        # CUDA would compile the tree, and the training scan inlines the
        # walk itself, so that compile would never be reused.
        walk = (
            kin._forward_levels
            if kin._level_specs is not None
            else kin._forward_serial
        )
        parts: list[tuple[torch.Tensor, ...]] = []
        frames = self.frames
        with torch.no_grad():
            for start in range(0, int(frames.shape[0]), 4096):
                batch = frames[start : start + 4096].to(
                    device=device, dtype=torch.float32
                )
                parts.append(tuple(item.detach() for item in walk(batch)))
        return tuple(torch.cat([part[i] for part in parts], dim=0) for i in range(4))

    def kinematics_at(
        self, motion_id: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tracked-body pose and velocity of the reference frames.

        Same index rules as :meth:`state_at`. The tensors are detached: the
        reference clip has no gradient. Predicted states still run FK online.
        """
        squeezed = motion_id.dim() == 0
        if squeezed:
            motion_id = motion_id.unsqueeze(0)
            time = time.unsqueeze(0)
        query_device = motion_id.device
        pos, rot, lin, ang = self._kinematics(query_device)
        offsets, lengths = self._index(query_device)
        mid = motion_id.long()
        index = time.to(device=query_device).long().clamp_min(0)
        index = torch.minimum(index, lengths[mid] - 1)
        flat = offsets[mid] + index
        out = (pos[flat], rot[flat], lin[flat], ang[flat])
        if squeezed:
            out = tuple(item.squeeze(0) for item in out)
        return out

    def goal_ref(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """680-d reference: frames t .. t+9, last frame held past the clip end."""
        q0 = self.q0.to(device=motion_id.device)
        first, last = GOAL_WINDOW["paper"]
        offsets = torch.arange(first, last + 1, device=motion_id.device)
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
        first, last = GOAL_WINDOW["goal730"]
        offsets = torch.arange(first, last + 1, device=motion_id.device)
        times = time.long().unsqueeze(-1) + offsets
        length = self.clip_length(motion_id.long())
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
        length = self.clip_length(motion_id.long())
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

    def past_clip_end(
        self, motion_id: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """True once ``time`` is past the last stored frame.

        ``state_at`` still returns that last frame, so a terminal step holds
        the reference instead of reading off the end of the clip.
        """
        return time.long() >= self.clip_length(motion_id)


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


def control_time_grid(
    num_frames: int, src_fps: float, dst_fps: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Source sample times and the destination times on ``1 / dst_fps``.

    Destination times are ``0, dt, 2 dt, ...`` while ``t`` is still inside
    the source duration ``(num_frames - 1) / src_fps``. A tail shorter than
    one control step is dropped. It is not stretched onto the source
    endpoint, so every finite difference uses the same ``dt``.
    """
    if num_frames < 2:
        raise ValueError(
            f"a reference clip needs at least two frames to resample, got {num_frames}"
        )
    dt = 1.0 / float(dst_fps)
    # i / dst_fps <= (num_frames - 1) / src_fps, in exact rational form.
    steps = int(math.floor((num_frames - 1) * float(dst_fps) / float(src_fps) + 1e-8))
    if steps < 1:
        raise ValueError(
            f"a clip of {num_frames} frames at {src_fps} fps is shorter than one "
            f"{dst_fps} Hz control step"
        )
    src_t = torch.arange(num_frames, dtype=torch.float64) / float(src_fps)
    dst_t = torch.arange(steps + 1, dtype=torch.float64) * dt
    return src_t, dst_t, dt


def _interp_weights(
    src_t: torch.Tensor, dst_t: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.searchsorted(src_t, dst_t, right=True).clamp(1, count - 1)
    t0 = src_t[idx - 1]
    t1 = src_t[idx]
    u = (dst_t - t0) / (t1 - t0).clamp_min(1e-12)
    return idx, u


def resample_series(
    values: torch.Tensor, src_fps: float, dst_fps: float = CONTROL_FPS
) -> torch.Tensor:
    """Linear resample of a per-frame array onto the control-time grid."""
    values = torch.as_tensor(values, dtype=torch.float32)
    count = int(values.shape[0])
    src_t, dst_t, _dt = control_time_grid(count, src_fps, dst_fps)
    idx, u = _interp_weights(src_t, dst_t, count)
    shape = [u.shape[0]] + [1] * (values.ndim - 1)
    u = u.to(dtype=values.dtype).reshape(shape)
    return (1.0 - u) * values[idx - 1] + u * values[idx]


def _central_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Rate at each sample: central inside, one-sided at the two ends."""
    rate = torch.empty_like(values)
    rate[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    rate[0] = (values[1] - values[0]) / dt
    rate[-1] = (values[-1] - values[-2]) / dt
    return rate


def _root_angular_velocity(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """Root-frame angular velocity from a world-frame central SO(3) difference."""
    world = torch.empty(quat.shape[0], 3, dtype=quat.dtype)
    world[1:-1] = quat_to_rotvec(quat_mul(quat[2:], _quat_inv(quat[:-2]))) / (2.0 * dt)
    world[0] = quat_to_rotvec(quat_mul(quat[1], _quat_inv(quat[0]))) / dt
    world[-1] = quat_to_rotvec(quat_mul(quat[-1], _quat_inv(quat[-2]))) / dt
    return torch.einsum("tji,tj->ti", quat_to_rotmat(quat), world)


def resample_states(
    state: torch.Tensor, src_fps: float, dst_fps: float = CONTROL_FPS
) -> torch.Tensor:
    """Linear positions, slerp quaternions, velocities on the control grid.

    Root position, joint position, and translation use linear interpolation.
    Root orientation uses slerp, so the result stays unit length and does
    not flip hemisphere between adjacent samples. Velocities are central
    differences on the same grid, so frame ``i`` holds the rate at time
    ``i * dt`` rather than half a step earlier; the two end frames use the
    one-sided difference. Angular velocity is expressed in the root frame.
    """
    count = int(state.shape[0])
    src_t, dst_t, dt = control_time_grid(count, src_fps, dst_fps)
    idx, u = _interp_weights(src_t, dst_t, count)
    u = u.to(dtype=state.dtype)
    u_col = u.unsqueeze(-1)
    out = torch.empty(dst_t.shape[0], STATE_DIM, dtype=state.dtype)
    out[:, :3] = (1.0 - u_col) * state[idx - 1, :3] + u_col * state[idx, :3]
    out[:, JOINT_POS] = (1.0 - u_col) * state[idx - 1, JOINT_POS] + u_col * state[
        idx, JOINT_POS
    ]
    out[:, ROOT_QUAT] = _slerp(state[idx - 1, ROOT_QUAT], state[idx, ROOT_QUAT], u)
    out[:, ROOT_LIN_VEL] = _central_difference(out[:, :3], dt)
    out[:, JOINT_VEL] = _central_difference(out[:, JOINT_POS], dt)
    out[:, ROOT_ANG_VEL] = _root_angular_velocity(out[:, ROOT_QUAT], dt)
    return out


def decode_motion_payload(
    payload: object, source: str
) -> tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError(
            f"{source} must be a dict with fps, joint_names, q0, and state"
        )
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
        raise ValueError(
            f"{source} q0 has shape {tuple(q0.shape)}, expected ({ACTION_DIM},)"
        )
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(
            f"{source} state has shape {tuple(state.shape)}, expected [T, {STATE_DIM}]"
        )
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


def check_manifest(directory: Path, files: list[Path]) -> list[str]:
    """Verify ``files`` against ``directory/manifest.json`` if there is one.

    Returns each clip's source name, or the file name without a manifest.
    A listed file with another SHA-256, or a file the manifest does not
    list, raises: the clips are not the ones that were prepared.
    """
    path = directory / "manifest.json"
    if not path.is_file():
        return [file.name for file in files]
    manifest = json.loads(path.read_text())
    listed = {clip["file"]: clip for clip in manifest.get("clips", [])}
    names = []
    for file in files:
        entry = listed.get(file.name)
        if entry is None:
            raise ValueError(f"{file} is not listed in {path}")
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{file} does not match its SHA-256 in {path}")
        names.append(str(entry.get("source", file.name)))
    return names


def load_motion_library(
    path: str | None, num_clips: int | None, device: torch.device | str
) -> MotionLibrary:
    """Load verified ``clip_*.pt`` payloads.

    ``num_clips`` takes the first files in name order; ``None`` takes all of
    them. A missing, short, or unchecked directory raises.
    """
    directory = Path(path) if path else None
    files = (
        sorted(directory.glob("clip_*.pt")) if directory and directory.is_dir() else []
    )
    if files:
        if num_clips is not None and len(files) < num_clips:
            raise FileNotFoundError(
                f"{directory} has {len(files)} clip_*.pt files, need {num_clips}."
            )
        selected = files if num_clips is None else files[:num_clips]
        sources = check_manifest(directory, selected)
        clips = []
        q0 = None
        for file in selected:
            payload = torch.load(file, map_location="cpu", weights_only=True)
            state, clip_q0 = decode_motion_payload(payload, str(file))
            if q0 is None:
                q0 = clip_q0
            elif not torch.allclose(q0, clip_q0, atol=1e-5):
                raise ValueError(f"{file} default pose does not match the other clips")
            clips.append(state.to(device))
        library = MotionLibrary(clips, q0=q0)
        library.clip_names = sources
        return library
    raise FileNotFoundError(
        "No verified G1 reference clips at motion_path. Each clip_*.pt must "
        "contain fps, the 29 G1 joint names, the default pose q0, and state "
        "[T, 71]."
    )
