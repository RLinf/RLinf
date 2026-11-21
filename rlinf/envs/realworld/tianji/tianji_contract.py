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

"""Stage1 physical absolute TCP poses; Stage2 uses fixed affine coordinates.

Position / 2 m, rotation columns unchanged, 2*closure-1. This encoding is
separate from the checkpoint's quantile normalization; weights stay intact.

The model's chunk length and control rate come from the deployment
(``TIANJI_ACTION_HORIZON`` / ``TIANJI_CONTROL_HZ``); every process of one run
sources the same launcher environment, so they agree on both.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def default_loopinfra_root(loopinfra_root=None) -> Path:
    """Resolve LoopInfra checkout path from explicit arg, LOOPINFRA_ROOT, or ~/LoopInfra."""
    if loopinfra_root is not None:
        return Path(loopinfra_root).expanduser()
    env_val = os.environ.get("LOOPINFRA_ROOT", "").strip()
    if env_val:
        return Path(env_val).expanduser()
    return Path.home() / "LoopInfra"

ARM_NAMES = ("A", "B")
JOINTS_PER_ARM = 7
STATE_DIM = ACTION_DIM = 20
GRIPPER_DIMS = (9, 19)
EEF_ACTION_FIELDS = ("x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "closure")
TASK_PROMPT = "Fold the flat cardboard blank into a box and place it on the right side of the table."
SEMANTICS = "tianji.absolute_tcp_rot6.v2"
# LoopInfra camera sync supports exactly these rates.
CONTROL_RATES_HZ = (10.0, 15.0, 30.0)
MAX_ACTION_HORIZON = 120
# Inference time reserved inside a prefetched chunk (A1 measured on the same
# GPU class: mean 385 ms, p90 438 ms, max 544 ms).
DEFAULT_INFERENCE_BUDGET_S = 0.6


def rotation_from_6d(value):
    """Column-major Gram–Schmidt, rejecting undefined orientations."""
    v = np.asarray(value, dtype=np.float64)
    if v.shape[-1] != 6 or not np.isfinite(v).all():
        raise ValueError("rotation must be finite [...,6]")
    x, y = v[..., :3], v[..., 3:]
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(nx < 1e-8):
        raise ValueError("degenerate rotation x axis")
    x = x / nx
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(ny < 1e-8):
        raise ValueError("degenerate rotation y axis")
    y = y / ny
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def rotation_to_6d(rotation):
    r = np.asarray(rotation)
    return np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1)


def project_pose(values):
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] != 20 or not np.isfinite(result).all():
        raise ValueError("pose must be finite [...,20]")
    for start in (0, 10):
        result[..., start + 3 : start + 9] = rotation_to_6d(
            rotation_from_6d(result[..., start + 3 : start + 9])
        )
    result[..., [9, 19]] = np.clip(result[..., [9, 19]], 0, 1)
    return result


def identity_state():
    return np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0] * 2, dtype=np.float32)


@dataclass(frozen=True)
class TianjiContract:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    action_chunk_length: int = 30
    control_hz: float = 10.0
    image_roles: tuple[str, ...] = ("head", "wrist_A", "wrist_B")
    action_scale: np.ndarray = field(
        default_factory=lambda: np.asarray(
            [2, 2, 2, 1, 1, 1, 1, 1, 1, 0.5] * 2, dtype=np.float32
        )
    )

    def __post_init__(self):
        if (self.state_dim, self.action_dim) != (20, 20):
            raise ValueError("Stage1 v2 requires state20/action20")
        if (
            type(self.action_chunk_length) is not int
            or not 2 <= self.action_chunk_length <= MAX_ACTION_HORIZON
        ):
            raise ValueError(
                f"action chunk length must be an int in 2..{MAX_ACTION_HORIZON}"
            )
        if float(self.control_hz) not in CONTROL_RATES_HZ:
            raise ValueError(f"control rate must be one of {CONTROL_RATES_HZ} Hz")
        if tuple(self.image_roles) != ("head", "wrist_A", "wrist_B"):
            raise ValueError("camera order must be head,wrist_A,wrist_B")
        if not np.array_equal(self.action_scale, [2, 2, 2, 1, 1, 1, 1, 1, 1, 0.5] * 2):
            raise ValueError("Stage2 v2 affine scale is fixed")

    @property
    def camera_order(self):
        return self.image_roles

    @property
    def control_period_s(self):
        return 1.0 / float(self.control_hz)

    def stream_interval_s(self):
        """Streaming segment length, inside LoopInfra's 0.05–0.12 s admission.

        A segment keeps coasting until the next one replaces it, so it must
        outlast one period. Its receipt (the step's end) comes 45 ms before
        the curve ends and needs the policy gripper command already sent (DM
        period 20 ms): below about 0.06 s the label can be missing. 0.12 s at
        10 Hz; 0.062 s at 30 Hz, where the receipt bounds a step to about
        37 ms (measured 26.7 Hz on the simulated collector).
        """
        override = os.environ.get("TIANJI_STREAM_INTERVAL_S")
        if override:
            value = float(override)
            if not 0.05 <= value <= 0.12:
                raise ValueError("TIANJI_STREAM_INTERVAL_S must be within 0.05..0.12 s")
            return value
        return max(0.062, min(0.12, self.control_period_s + 0.02))

    def initial_delay_steps(self):
        """Steps that execute during one Stage1 request (measured 3 at 10 Hz)."""
        return max(1, math.ceil(round(0.3 * float(self.control_hz), 6)))

    @property
    def action_bias(self):
        bias = np.zeros(20, dtype=np.float32)
        bias[[9, 19]] = 0.5
        return bias

    def validate_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("state must be finite float32[20]")
        if np.any(state[[9, 19]] < 0) or np.any(state[[9, 19]] > 1):
            raise ValueError("commanded closure must be in [0,1]")
        if not np.allclose(project_pose(state), state, atol=1e-4):
            raise ValueError("state rotation columns must be orthonormal")
        return state

    def validate_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != (20,)
            or not np.isfinite(action).all()
            or np.any(np.abs(action) > 1.000001)
        ):
            raise ValueError("Stage2 action must be finite float32[20] in [-1,1]")
        return np.clip(action, -1, 1)

    def encode_action(self, physical):
        return (
            np.asarray(physical, dtype=np.float32) - self.action_bias
        ) / self.action_scale

    def decode_action(self, action):
        return project_pose(
            self.validate_action(action) * self.action_scale + self.action_bias
        )

    def encode_gripper(self, closure):
        if not np.isfinite(closure):
            raise ValueError("closure must be finite")
        return float(np.clip(closure, 0, 1))


def contract_from_env(environ=None):
    """The deployment's contract; unset variables keep the 30-step 10 Hz model."""
    environ = os.environ if environ is None else environ
    return TianjiContract(
        action_chunk_length=int(environ.get("TIANJI_ACTION_HORIZON", "30")),
        control_hz=float(environ.get("TIANJI_CONTROL_HZ", "10")),
    )


@dataclass(frozen=True)
class PrefetchSchedule:
    """A1 prefetch timetable for one model horizon and control rate.

    ``prefetch_at`` is the 0-based step after which the next request reads
    its observation, so ``prefetch_at + 1`` steps have executed. The RL chunk
    runs ``chunk`` steps; the remaining ``ref_offset = chunk - 1 - prefetch_at``
    steps execute while the request is in flight, and the actor reads the
    reference chunk from that offset (RLT_REF_OFFSET).
    """

    horizon: int
    control_hz: float
    chunk: int
    prefetch_at: int
    ref_offset: int

    @property
    def request_after_steps(self):
        return self.prefetch_at + 1

    @property
    def inference_budget_s(self):
        return self.ref_offset / float(self.control_hz)


def prefetch_schedule(horizon, control_hz, budget_s=DEFAULT_INFERENCE_BUDGET_S):
    """Largest RL chunk that still leaves ``budget_s`` for inference.

    A 30-step 10 Hz model gives chunk 24 with the request after 18 executed
    steps; a 60-step 30 Hz model gives chunk 42 with the request after 24.
    """
    horizon, control_hz = int(horizon), float(control_hz)
    if not 0 < budget_s < horizon / control_hz:
        raise ValueError("inference budget must be shorter than one model chunk")
    offset = math.ceil(round(budget_s * control_hz, 6))
    chunk = horizon - offset
    prefetch_at = chunk - 1 - offset
    if prefetch_at < 0:
        raise ValueError(
            f"a {horizon}-step chunk at {control_hz:g} Hz cannot reserve {budget_s:g} s "
            "for inference after at least one executed step"
        )
    return PrefetchSchedule(horizon, control_hz, chunk, prefetch_at, offset)


DEFAULT_CONTRACT = contract_from_env()


STAGE_TIME_SCALE_ENV = "TIANJI_STAGE_TIME_SCALE"
MAX_STAGE_TIME_SCALE = 1.7
MIN_SCALED_DURATION_S = 0.05


def stage_time_scale_from_env(environ=os.environ) -> dict[int, float]:
    """11-04 experiment B: run one stage's policy rows faster than the model rate.

    ``TIANJI_STAGE_TIME_SCALE="4:1.25"`` sends stage 4's rows every period/1.25;
    each segment keeps its 0.062 s duration. The rows, and the executor's speed,
    acceleration and lead limits, are unchanged. Empty = off.
    The env worker keeps these chunks out of training (record_transition).
    """
    text = (environ.get(STAGE_TIME_SCALE_ENV) or "").strip()
    scales: dict[int, float] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        stage, sep, value = item.partition(":")
        try:
            stage, value = int(stage), float(value)
        except ValueError:
            raise ValueError(f"{STAGE_TIME_SCALE_ENV} item {item!r} must be stage:scale") from None
        if not sep or not 1 <= stage <= 8 or not 1.0 <= value <= MAX_STAGE_TIME_SCALE or stage in scales:
            raise ValueError(
                f"{STAGE_TIME_SCALE_ENV} item {item!r}: each stage 1-8 once, scale 1.0-{MAX_STAGE_TIME_SCALE}"
            )
        scales[stage] = value
    return scales


ROW_SCALE_ENV = "TIANJI_ROW_SCALE"
MAX_ROW_SCALE = 3.0


def row_scale_from_env(environ=os.environ) -> float:
    """Model rows per executed 30 Hz row (11-04 speed-up); unset or empty = 1.0 (off)."""
    text = (environ.get(ROW_SCALE_ENV) or "").strip()
    if not text:
        return 1.0
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{ROW_SCALE_ENV}={text!r} must be a number") from None
    if not math.isfinite(value) or not 1.0 <= value <= MAX_ROW_SCALE:
        raise ValueError(f"{ROW_SCALE_ENV}={text!r} must lie in [1, {MAX_ROW_SCALE:g}]")
    return value


DENSE_ROWS_ENV = "TIANJI_DENSE_ROWS"


def dense_rows_from_env(environ=os.environ) -> int | float:
    """11-05 dense rows (TIANJI_DENSE_ROWS=1): the executor runs every model row from the offset to
    the horizon at 1/(control_hz*k) s, k = TIANJI_ROW_SCALE (2, or 1.5 since 11-06: rows every 1/45 s,
    three model rows per two env rows); the env keeps stepping its rows at control_hz. Returns k, or 0
    when off (unset, empty or 0)."""
    text = (environ.get(DENSE_ROWS_ENV) or "").strip()
    if text in ("", "0"):
        return 0
    if text != "1":
        raise ValueError(f"{DENSE_ROWS_ENV}={text!r} must be 1, 0 or unset")
    scale = row_scale_from_env(environ)
    if scale == 1.0:
        return 0  # One target per environment step; a retained dense flag is inert.
    if scale == 1.5:
        return 1.5  # 11-06: 1.5x, every model row at 1/45 s
    if scale != int(scale) or scale < 2:
        raise ValueError(f"{DENSE_ROWS_ENV}=1 needs {ROW_SCALE_ENV}=2 or 1.5, not {scale:g}")
    if int(scale) != 2:
        raise ValueError(
            f"{DENSE_ROWS_ENV}=1 is validated for {ROW_SCALE_ENV}=2 only (rows every 1/60 s); "
            f"{ROW_SCALE_ENV}={int(scale)} would need rows every 1/{30 * int(scale)} s"
        )
    return int(scale)


def resampled_schedule(horizon, control_hz, budget_s, scale):
    """(chunk, prefetch_at, ref_offset, inflight) when each executed row advances ``scale`` model rows.

    ``inflight`` executed rows (at least ``budget_s``, rounded up until scale*inflight is whole) run
    while the next request is in flight; they cover ref_offset = scale*inflight model rows, the next
    chunk's frozen RTC prefix. ``chunk`` is the most executed rows whose model rows still fit the
    horizon: ref_offset + scale*chunk <= horizon. With scale 1 this is prefetch_schedule().
    """
    horizon, control_hz = int(horizon), float(control_hz)
    inflight = math.ceil(round(budget_s * control_hz, 6))
    while abs(scale * inflight - round(scale * inflight)) > 1e-9:
        inflight += 1
    ref_offset = int(round(scale * inflight))
    chunk = int(math.floor((horizon - ref_offset) / scale + 1e-9))
    prefetch_at = chunk - 1 - inflight
    if prefetch_at < 0:
        raise ValueError(
            f"row scale {scale:g}: {inflight} in-flight rows ({budget_s:g} s) leave no executed row "
            f"before the request in a {horizon}-row model chunk"
        )
    return chunk, prefetch_at, ref_offset, inflight
