# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Control-loop alignment, Smoothstep blending, step clamp, and velocity diagnostics.

Implements the invariants and online chunk monitor specified in LoopSim
``docs/inference.md`` §2–3 for 16D joint Tianji control (arm joints in rad,
commanded gripper closures in [0, 1]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ARM_JOINT_IDX = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
GRIPPER_IDX = [7, 15]


def subsample_chunk_for_rate(
    chunk: np.ndarray, *, model_hz: float = 30.0, control_hz: float = 30.0
) -> np.ndarray:
    """Enforce 1:1 timestep alignment between model predictions and dispatch frequency.

    Native 30 Hz returns ``chunk`` unchanged. When dispatching a 30 Hz model at
    10 Hz (``stride = 3``), slices ``chunk[0::3]`` (20 steps = 2.0 s for H=60).
    """
    arr = np.asarray(chunk)
    if control_hz <= 0 or model_hz <= 0:
        raise ValueError("model_hz and control_hz must be positive")
    ratio = float(model_hz) / float(control_hz)
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, atol=1e-6):
        raise ValueError(
            f"Unsupported model_hz={model_hz:g} / control_hz={control_hz:g}; expected integer stride"
        )
    if stride == 1:
        return arr.copy()
    return arr[::stride].copy()


def compute_elapsed_skip_steps(
    t_switch: float,
    t_obs: float,
    dt: float,
    *,
    default_skip: int = 18,
    max_infer_stale_s: float = 0.6,
) -> tuple[int, bool]:
    """Compute stale head steps ``m = round((t_switch - t_obs) / dt)`` and budget status."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    elapsed = max(0.0, float(t_switch) - float(t_obs))
    skip_m = max(0, int(round(elapsed / float(dt))))
    exceeded_budget = elapsed > float(max_infer_stale_s) + 0.5 * float(dt)
    if elapsed == 0.0 and default_skip > 0 and t_switch == t_obs:
        return 0, False
    return skip_m, exceeded_budget


def blend_new_chunk(
    q_prev_cmd: np.ndarray,
    new_chunk: np.ndarray,
    skip_m: int = 0,
    blend_steps: int = 10,
) -> np.ndarray:
    """Smooth chunk transitions via deviation-decay Smoothstep on 14 arm joints.

    Discards ``skip_m`` elapsed head steps from ``new_chunk``, computes the
    boundary offset ``dq0 = q_prev_cmd[ARM_JOINT_IDX] - rem[0, ARM_JOINT_IDX]``,
    and decays ``dq0`` over ``blend_steps`` using ``1 - (3u^2 - 2u^3)`` while
    leaving gripper dimensions ``[7, 15]`` untouched. Because ``ARM_JOINT_IDX``
    coordinates are affine between physical radians and RL ``[-1, 1]`` frame,
    blending produces identical trajectories in both spaces.
    """
    arr = np.asarray(new_chunk, dtype=np.float32)
    skip = max(0, int(skip_m))
    rem = arr[skip:].copy()
    n = min(max(0, int(blend_steps)), len(rem))
    if n <= 0:
        return rem
    q_prev = np.asarray(q_prev_cmd, dtype=np.float32).reshape(-1)
    dq0 = q_prev[ARM_JOINT_IDX] - rem[0, ARM_JOINT_IDX]
    u = np.arange(1, n + 1, dtype=np.float32)[:, None] / float(n)
    decay = 1.0 - (3.0 * u**2 - 2.0 * u**3)
    rem[:n, ARM_JOINT_IDX] += decay * dq0
    return rem


def clamp_chunk_step_velocity(
    q_prev_cmd: np.ndarray,
    target_chunk: np.ndarray,
    *,
    dt: float = 1.0 / 30.0,
    v_max_rad_s: float = 0.7854,
    joint_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    """Enforce per-step arm joint delta hard clamp ``clip(dq, -v_max*dt, +v_max*dt)``.

    Args:
        q_prev_cmd: Previous commanded 16D target (same frame as ``target_chunk``).
        target_chunk: ``[K, 16]`` target trajectory before clamping.
        dt: Control period in seconds.
        v_max_rad_s: Maximum allowed arm joint angular velocity in rad/s.
        joint_scale: Optional 16D RL-to-physical scale vector. When provided,
            clamping uses physical radians per joint (``v_max_rad_s * dt / scale[j]``).
            When None, ``target_chunk`` is assumed to already be in radians.

    Returns:
        ``(clamped_chunk, clamped_step_count, preclamp_max_vel_rad_s)``.
    """
    if dt <= 0 or v_max_rad_s <= 0:
        raise ValueError("dt and v_max_rad_s must be positive")
    chunk = np.asarray(target_chunk, dtype=np.float32).copy()
    if chunk.ndim != 2 or chunk.shape[1] != 16:
        raise ValueError(f"Expected [K, 16] chunk, got shape {chunk.shape}")
    if len(chunk) == 0:
        return chunk, 0, 0.0

    prev = np.asarray(q_prev_cmd, dtype=np.float32).reshape(16).copy()
    if joint_scale is None:
        scale_arm = np.ones(len(ARM_JOINT_IDX), dtype=np.float64)
    else:
        scale_arm = np.asarray(joint_scale, dtype=np.float64)[ARM_JOINT_IDX]

    max_step_rad = float(v_max_rad_s) * float(dt)
    max_step_frame = max_step_rad / scale_arm
    clamped_steps = 0
    preclamp_vmax = 0.0

    prev_unclamped = prev[ARM_JOINT_IDX].astype(np.float64)
    prev_cmd = prev[ARM_JOINT_IDX].astype(np.float64)
    for k in range(len(chunk)):
        tgt_arm = chunk[k, ARM_JOINT_IDX].astype(np.float64)
        raw_vel = np.max(np.abs(tgt_arm - prev_unclamped) * scale_arm) / float(dt)
        if raw_vel > preclamp_vmax:
            preclamp_vmax = float(raw_vel)
        prev_unclamped = tgt_arm

        delta = tgt_arm - prev_cmd
        clipped = np.clip(delta, -max_step_frame, max_step_frame)
        if not np.allclose(clipped, delta, atol=1e-7, rtol=0.0):
            clamped_steps += 1
        next_arm = prev_cmd + clipped
        chunk[k, ARM_JOINT_IDX] = next_arm.astype(np.float32)
        prev_cmd = next_arm

    return chunk, clamped_steps, float(preclamp_vmax)


class ChunkExecutionMonitor:
    """Online detector for velocity clamping, boundary jumps, flat-top clipping, and servo lag."""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        v_max_rad_s: float = 2.0,
        lag_warn_deg: float = 2.0,
        log_path: str | Path | None = None,
    ):
        self.dt = float(dt)
        self.v_max = float(v_max_rad_s)
        self.lag_warn_deg = float(lag_warn_deg)
        self.log_path = Path(log_path) if log_path is not None else None
        self.chunk_id = 0

    def analyze_chunk(
        self,
        q_actual_before_chunk: np.ndarray,  # (16,) physical state right before step 0
        q_target_seq: np.ndarray,           # (K, 16) un-clamped physical target sequence
        q_meas_seq: np.ndarray,             # (K, 16) physical measured state per step
    ) -> dict[str, Any]:
        """Compute per-chunk diagnostics, format log line, and optionally append JSONL."""
        self.chunk_id += 1
        q0 = np.asarray(q_actual_before_chunk, dtype=np.float64).reshape(-1)[ARM_JOINT_IDX]
        tgt = np.asarray(q_target_seq, dtype=np.float64)[:, ARM_JOINT_IDX]
        meas = np.asarray(q_meas_seq, dtype=np.float64)[:, ARM_JOINT_IDX]
        if len(tgt) == 0 or len(meas) == 0 or len(tgt) != len(meas):
            raise ValueError(
                f"Target ({tgt.shape}) and measured ({meas.shape}) sequences must be non-empty and aligned"
            )
        k_steps = len(tgt)

        # 1. Step-0 boundary jump & requested angular velocity per step (pre-clamp)
        prev_tgt = np.vstack([q0[None, :], tgt[:-1]])
        req_vel = np.abs(tgt - prev_tgt) / self.dt  # (K, 14)
        step0_jump_deg = float(np.degrees(np.abs(tgt[0] - q0).max()))
        req_vmax = float(req_vel.max())
        step_clamped = req_vel.max(axis=1) > 0.90 * self.v_max
        clamp_count = int(step_clamped.sum())
        k0_clamp = int(bool(step_clamped[0]))

        # 2. Measured angular velocity & flat-top clipping check
        prev_meas = np.vstack([q0[None, :], meas[:-1]])
        meas_vel = np.abs(meas - prev_meas) / self.dt  # (K, 14)
        meas_step_max = meas_vel.max(axis=1)
        meas_vmax = float(meas_step_max.max())
        meas_p95 = float(np.quantile(meas_step_max, 0.95))
        flat_top_suspected = bool(
            clamp_count == 0 and meas_vmax > 0.3 and meas_p95 >= 0.96 * meas_vmax
        )

        # 3. End-of-chunk tracking lag & saturation vs contact classification
        end_err_deg = np.degrees(np.abs(tgt[-1] - meas[-1]))
        end_lag_deg = float(end_err_deg.max())
        worst_j = int(end_err_deg.argmax())
        worst_j_vel = float(meas_vel[-1, worst_j])

        status = "OK"
        if k0_clamp and clamp_count == 1:
            status = "WARN_BOUNDARY_JUMP"
        elif clamp_count > 0:
            status = "WARN_CMD_CLAMP"
        elif end_lag_deg > self.lag_warn_deg:
            status = (
                "WARN_VEL_SATURATED"
                if worst_j_vel > 0.85 * self.v_max
                else "INFO_CONTACT_HOLD"
            )

        summary = (
            f"[chunk #{self.chunk_id:04d}] step0_jump={step0_jump_deg:.2f}° | "
            f"req_vmax={req_vmax:.2f} rad/s (clamp={clamp_count}/{k_steps}, k0_clamp={k0_clamp}) | "
            f"meas_vmax={meas_vmax:.2f} rad/s (p95={meas_p95:.2f}) | "
            f"end_lag={end_lag_deg:.2f}° | status={status}"
        )
        record = {
            "chunk_id": self.chunk_id,
            "k_steps": k_steps,
            "dt": self.dt,
            "v_max_rad_s": self.v_max,
            "lag_warn_deg": self.lag_warn_deg,
            "step0_jump_deg": round(step0_jump_deg, 4),
            "req_vmax_rad_s": round(req_vmax, 4),
            "clamp_count": clamp_count,
            "k0_clamp": k0_clamp,
            "meas_vmax_rad_s": round(meas_vmax, 4),
            "meas_p95_rad_s": round(meas_p95, 4),
            "end_lag_deg": round(end_lag_deg, 4),
            "worst_joint_idx": worst_j,
            "worst_joint_vel_rad_s": round(worst_j_vel, 4),
            "flat_top_suspected": flat_top_suspected,
            "status": status,
            "summary": summary,
        }
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def summarize_chunk(
        self,
        q_actual_before_chunk: np.ndarray,
        q_target_seq: np.ndarray,
        q_meas_seq: np.ndarray,
    ) -> str:
        """Analyze chunk, optionally record JSONL, and return the single-line summary."""
        return self.analyze_chunk(q_actual_before_chunk, q_target_seq, q_meas_seq)[
            "summary"
        ]
