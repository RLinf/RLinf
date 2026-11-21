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

"""Gymnasium task for the Tianji Marvin Pro + LoopInfra stack."""

from __future__ import annotations

import os
from pathlib import Path
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from .tianji_actions import observed_action
from .tianji_backend import (
    DummyTianjiBackend,
    LoopInfraPolicyBackend,
    TianjiBackend,
    joint_physical,
    retryable_admission,
    unreachable_target,
)
from .tianji_kinematics import (
    JOINT_CONTRACT,
    JOINT_DIM,
    JOINT_DIMS,
    JOINT_ROW_SEMANTICS,
    joint_rl_to_tcp_rl_np,
)
from .tianji_contract import (
    ACTION_DIM,
    DEFAULT_CONTRACT,
    STATE_DIM,
    TASK_PROMPT,
    TianjiContract,
    identity_state,
    rotation_from_6d,
    rotation_to_6d,
)
from . import tianji_joint
from .tianji_hil import TianjiHILControl
from .tianji_recorder import TianjiTransitionRecorder
from .tianji_subtasks import load_subtask_prompts
from rlinf.tianji_shadow import transport as shadow

# A1 KeyboardControl: each completed subtask earns this before Success.
MILESTONE_REWARD = 1.0
# The training panel's "step failure": the step being executed earns this on
# top of its other rewards; control continues unchanged.
STEP_FAILURE_REWARD = -1.0
# Human-takeover chunks (every real step H-mode or Pico) may complete their
# executed-action labels for Q: H-mode holds use their measured pose, and up to
# this many missing steps between two labelled ones are interpolated, if the
# two labels were captured (sync.reference_monotonic_ns) at most
# HUMAN_FILL_MAX_SPAN_S apart (takeover steps run ~40 ms: a 2-step gap spans
# ~120 ms; a pause or reset spans seconds).
HUMAN_FILL_MAX_GAP = 2
HUMAN_FILL_MAX_SPAN_S = 0.2
# A takeover chunk in which both arms and grippers stay within these bounds
# carries no action: it is no BC target, and no Q sample unless it has a reward.
STILL_POSITION_M = 0.001
STILL_ROTATION_RAD = float(np.radians(0.5))
STILL_CLOSURE = 0.02


def _quaternion(rotation):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _slerp(r0, r1, t):
    """Rotation a fraction ``t`` of the way from ``r0`` to ``r1``."""
    q0, q1 = _quaternion(r0), _quaternion(r1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1 - t) * angle) * q0 + np.sin(t * angle) * q1) / np.sin(angle)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _capture_time(observation):
    """Seconds of the observation's synchronized capture, or None.

    LoopInfra's sync.reference_monotonic_ns is when the state was sampled;
    the env's own clock only knows when it asked, and a capture can come long
    after the request.
    """
    sync = (getattr(observation, "raw", None) or {}).get("sync") or {}
    stamp = sync.get("reference_monotonic_ns")
    if sync.get("sync_valid") is not True or type(stamp) is not int or stamp <= 0:
        return None
    return stamp / 1e9


def _gap_times(start, end, captures):
    """Times of a gap's missing steps between two labelled captures.

    A missing step keeps its own capture time when that is a new capture
    between its neighbours (a label refused for control alignment, say). A
    repeated or unstamped capture has no sample time of its own: it is spaced
    evenly between the nearest known times.
    """
    times = [start, *([None] * len(captures)), end]
    for n, capture in enumerate(captures, 1):
        latest = max(v for v in times[:n] if v is not None)
        if capture is not None and latest < capture < end:
            times[n] = capture
    known = [n for n, v in enumerate(times) if v is not None]
    filled = list(times)
    for n, v in enumerate(times):
        if v is None:
            lo = max(m for m in known if m < n)
            hi = min(m for m in known if m > n)
            filled[n] = times[lo] + (times[hi] - times[lo]) * (n - lo) / (hi - lo)
    return filled[1:-1]


def _still(states) -> bool:
    """Both arms and grippers stayed within the STILL_* bounds."""
    s = np.asarray(states, dtype=np.float64)
    for o in (0, 10):
        if np.abs(s[:, o : o + 3] - s[0, o : o + 3]).max() > STILL_POSITION_M:
            return False
        if np.abs(s[:, o + 9] - s[0, o + 9]).max() > STILL_CLOSURE:
            return False
        first = rotation_from_6d(s[0, o + 3 : o + 9])
        for value in s[1:, o + 3 : o + 9]:
            cos = (np.trace(first.T @ rotation_from_6d(value)) - 1) / 2
            if np.arccos(np.clip(cos, -1, 1)) > STILL_ROTATION_RAD:
                return False
    return True
# A stream window coasts for up to ~0.4 s after its last step; a saved-pose
# Reset waits this long for both arms to stop.
RESET_SETTLE_S = 2.0
# Unreachable targets (typically B J5 at its limit) this often pause the
# episode: each one already drops its chunk and replans, so the arm would
# just stand there (Stage1 used to stop the whole run at the first one).
UNREACHABLE_PAUSE_COUNT = 3
UNREACHABLE_PAUSE_WINDOW_S = 3.0


class TianjiStepError(RuntimeError):
    """A stopped step with its original receipt and no fabricated transition."""

    def __init__(
        self, message: str, receipt: dict, observation_error: str | None = None
    ):
        self.receipt = receipt
        self.observation_error = observation_error
        super().__init__(
            f"{message}: {receipt.get('status')} {receipt.get('reason', '')} "
            f"(action_id={receipt.get('action_id')})"
        )


def _hil_raw_lean(config) -> bool:
    """11-06: training archive with the 30 Hz source recording (TIANJI_HIL_RAW_RECORDING=1):
    hil/ keeps only what hil_raw cannot replace (no transitions.jsonl / frames, no ~11.5 Hz
    camera stream, no per-inference full-size tactile)."""
    return bool(
        config.episode_archive and config.recorder_dir and config.hil_mode == "training"
        and os.environ.get("TIANJI_HIL_RAW_RECORDING", "0") == "1"
    )


@dataclass
class TianjiConfig:
    """Runtime settings for the single physical Tianji pair."""

    backend: str = "dummy"
    server_url: str = "http://127.0.0.1:8780"
    server_token: str | None = None
    read_only: bool = True
    control_profile: str = "wuji_compat"
    reference_windows: bool = True
    request_timeout_s: float = 2.0
    image_hw: tuple[int, int] = (224, 224)
    # Opt-in Rectify tensors, carried separately from the frozen VLA's RGB.
    tactile_enabled: bool = False
    # With the training episode archive: the same inference-time snapshots also
    # at full resolution, saved as the collector does (xense/<role>.mp4 + .jsonl).
    tactile_raw_archive: bool = True
    step_frequency: float = DEFAULT_CONTRACT.control_hz
    max_num_steps: int = 600
    task_description: str = TASK_PROMPT
    # Ordered subtask prompts (JSON, see tianji_subtasks). The current
    # subtask's prompt replaces task_description and its number is the stage.
    subtask_prompts_file: str | None = None
    recorder_dir: str | None = None
    recorder_source_jpeg: bool = False
    # Training: also archive every episode under recorder_dir/episodes with
    # the collector's camera stream (tianji_episode_archive), for offline RL
    # and LeRobot export. Recording only; training data is unchanged.
    episode_archive: bool = False
    # Cap on the archived camera rate (Hz); None keeps every synced group.
    episode_archive_image_hz: float | None = None
    # Stage1's receding-horizon executor needs only execution metadata between
    # inference requests.  Keep the complete RGB observation path as the
    # default so Stage2 training semantics remain unchanged.
    sparse_execution_observation: bool = False
    # Stage2 A1 prefetch: one streaming window across chunks, measured
    # execution metadata every step, and RGB only for the prefetch request,
    # read by an independent client off the control thread.
    stream_chunks: bool = False
    # Stage2 reference chunks continue the in-flight actions (RLinf RTC).
    rtc_guidance: bool = True
    # 11-03: an episode's first chunk is planned from the reset pose without RTC,
    # and the joint VLA's first waypoints are often far from it (logs: joints
    # J2/J3/J5 17-31 deg on average, TCP p90 165 mm). Its first N rows blend from
    # the measured joints to the plan (smoothstep); closures follow the plan.
    # 40 rows keep the peak commanded joint speed of the logged first chunks
    # near the 45 deg/s trial45 limit (median 44 deg/s; 30 rows: 57). At most
    # the chunk length; 0 disables. Joint action space only.
    first_chunk_ramp_steps: int = 40
    # 11-04: a stream-mode request read while the arm held (H-mode hold, operator
    # pause, Pico takeover, a skipped chunk) carries no RTC prefix, and its chunk
    # started at the VLA's step 18, 10.6 deg (median; p90 20.7, max 25.9) from the
    # held joints (11-03/04 logs, 267 resumed chunks). Its first N rows blend from
    # the measured joints to the plan like the first chunk (40 rows: peak commanded
    # speed 41 deg/s median). Both ramps end inside the chunk (at most its rows):
    # the next request's RTC prefix is the plan's own rows, so a longer ramp left
    # a step at the next chunk (2x, 16-row chunks: 10.5 deg). 0 disables.
    resume_chunk_ramp_steps: int = 40
    hil_control: bool = False
    hil_port: int = 8879
    hil_mode: str = "training"
    reset_to_saved_pose: bool = True
    reset_after_episode: bool = True
    manual_episode_control_only: bool = False
    human_action_max_interval_s: float = 0.25
    # rtj joint Stage2 ("joint"): actions, VLA reference and robot commands are
    # 16D joints (tianji_kinematics.TianjiJointContract); the critic's executed
    # action stays the measured 20D TCP, and every label is kept in both spaces.
    action_space: str = "tcp"
    contract: TianjiContract = field(default_factory=lambda: DEFAULT_CONTRACT)


class TianjiEnv(gym.Env):
    """20D image-aligned TCP state and bounded absolute-pose action encoding.

    ``action_space: joint`` (rtj joint Stage2): 16D absolute joint actions, sent
    to the owner as joint targets only; the 20D TCP state and measured TCP
    executed actions stay the critic's, and every label is kept in both spaces.
    """

    metadata = {"render_modes": []}
    supports_relative_frame = False

    def __init__(
        self, config: TianjiConfig | Mapping[str, Any] | None = None, **kwargs
    ):
        del kwargs
        if config is None:
            config = TianjiConfig()
        if isinstance(config, Mapping):
            config = TianjiConfig(**dict(config))
        self.config = config
        self.contract = config.contract
        if config.action_space not in ("tcp", "joint"):
            raise ValueError("action_space must be tcp or joint")
        # Actions (and the hold / human BC labels in action space) use this
        # contract; observations, timing and the critic's TCP labels use contract.
        self.joint_actions = config.action_space == "joint"
        self.action_contract = JOINT_CONTRACT if self.joint_actions else self.contract
        if config.step_frequency <= 0 or config.max_num_steps < 1:
            raise ValueError("step_frequency and max_num_steps must be positive")
        # The deployment's preflight requires step_frequency == TIANJI_CONTROL_HZ.
        self._car_deployment = os.environ.get("TIANJI_CAR_1118") == "1"
        self.subtask_prompts = (
            load_subtask_prompts(config.subtask_prompts_file, limit=15 if self._car_deployment else 4)
            if config.subtask_prompts_file
            else ()
        )
        # Without a panel (tests, scripted sims) set_subtask() drives this.
        self._subtask = 1 if self.subtask_prompts else 0
        self._subtask_reached = self._subtask
        self._milestones = 0
        self._step_failure = False
        # Per-step labels of the executing chunk (training): finalize_chunk_exec.
        self._chunk_records = []
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (self.action_contract.action_dim,), dtype=np.float32
        )
        h, w = config.image_hw
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "qpos": gym.spaces.Box(
                            -np.inf, np.inf, (STATE_DIM,), dtype=np.float32
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        role: gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)
                        for role in self.contract.image_roles
                    }
                ),
                "rlt_qvel": gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
                # rtj fork: the joint checkpoint's state input (aligned joints +
                # closures), also the joint actor's proprio in joint Stage2.
                **(
                    {"rlt_joint_state": gym.spaces.Box(
                        -np.inf, np.inf, (tianji_joint.JOINT_STATE_DIM,), dtype=np.float32)}
                    if tianji_joint.joint_mode() or self.joint_actions else {}
                ),
            }
        )
        if self._car_deployment:
            if len(self.subtask_prompts) != 15 or not self.joint_actions:
                raise ValueError("1118v1 requires joint actions and 15 numbered prompts")
            self.observation_space.spaces["car_stage_meta"] = gym.spaces.Box(
                0, np.iinfo(np.int64).max, (6,), dtype=np.int64
            )
        self._shadow_schema = bool(shadow.ROOT and config.hil_mode == "training")
        if self._shadow_schema:
            bounds = np.iinfo(np.int64)
            self.observation_space.spaces["shadow_meta"] = gym.spaces.Box(
                bounds.min, bounds.max, (shadow.SIZE,), dtype=np.int64
            )
        self.backend: TianjiBackend = self._make_backend(config)
        if config.tactile_enabled:
            from rlinf.utils.xense_tactile import TACTILE_IMAGE_SHAPE, TACTILE_ROLES
            n = len(TACTILE_ROLES)
            self.observation_space.spaces.update({
                "tactile_images": gym.spaces.Box(0, 255, TACTILE_IMAGE_SHAPE, dtype=np.uint8),
                "tactile_valid": gym.spaces.Box(
                    np.zeros(n, dtype=np.bool_), np.ones(n, dtype=np.bool_), dtype=np.bool_),
                "tactile_age_ms": gym.spaces.Box(0, 100, (n,), dtype=np.float32),
            })
        self._recorder = (
            TianjiTransitionRecorder(
                config.recorder_dir,
                source_jpeg=config.recorder_source_jpeg,
                image_roles=config.contract.image_roles,
                # 11-06: with hil_raw the archive's steps.jsonl is the only row copy.
                write_rows=not _hil_raw_lean(config),
            )
            if config.recorder_dir
            else None
        )
        self.episode_archive = None
        if config.episode_archive and self._recorder is not None and (
            config.hil_mode == "training" or self._car_deployment
        ):
            try:
                from .tianji_episode_archive import TianjiEpisodeArchive

                self.episode_archive = TianjiEpisodeArchive(
                    config.recorder_dir,
                    server_url=config.server_url if config.backend.lower() == "loopinfra" else None,
                    token=config.server_token,
                    contract=self.contract,
                    control_hz=config.step_frequency,
                    image_roles=self.contract.image_roles,
                    prompts=self.subtask_prompts,
                    image_hz=config.episode_archive_image_hz,
                    action_contract=self.action_contract,
                )
            except Exception as exc:  # noqa: BLE001 -- recording never stops training
                import logging

                if self._car_deployment:
                    raise RuntimeError("1118v1 could not initialize HIL recording") from exc
                logging.getLogger(__name__).warning("episode archive disabled: %s", exc)
        self._last_observation = None
        self._last_raw_obs = None
        self._last_state = np.zeros(STATE_DIM, dtype=np.float32)
        self._last_time = time.monotonic()
        self._step = 0
        self._closed = False
        self._episode_done = False
        self._skip_chunk = False
        self._unreachable_times = []
        self._chunk_context = None
        self._chunk_revision = 0
        self._observation_revision = 0
        self._sample_revision = 0
        self._next_window_steps = None
        self._window_remaining = None
        self._chunk_steps_left = None
        # rt fork: this chunk's actions and the row index of the next step;
        # the next chunk offered early by the env worker, once appended.
        self._chunk_actions = None
        self._chunk_row = 0
        self._offered_next = None
        self._next_appended = False
        # First-chunk ramp: armed by reset, bound when that chunk's actions arrive.
        self._ramp_next_chunk = False
        self._ramp = None
        # Resume ramp: whether the last prefetched request was read under policy
        # (None: no prefetch since), and whether the chunk now beginning ramps.
        self._prefetch_policy = None
        self._ramp_resume_chunk = False
        # Due time of the last paced (held/takeover) step.
        self._hold_due = None
        self._last_frames = None
        # True after a step whose policy segment the owner executed.
        self._streaming_step = False
        self._prefetch_reader = None
        self._prefetch_pool = None
        self._prefetch_revision = None
        self.hil = (
            TianjiHILControl(
                mode=config.hil_mode, subtask_prompts=self.subtask_prompts,
                stage_switch_replan=self._car_deployment,
                event_path=(
                    Path(config.recorder_dir) / "subtask_events.jsonl"
                    if self._car_deployment and config.recorder_dir else None
                ),
            )
            if config.hil_control
            else None
        )
        if isinstance(self.backend, LoopInfraPolicyBackend):
            self.backend.reference_cancelled = self._reference_cancelled
            self.backend.current_subtask = lambda: self.subtask  # experiment B row time scale
        if self.hil is not None:
            self.hil.start_server(config.hil_port)

    @staticmethod
    def _make_backend(config: TianjiConfig) -> TianjiBackend:
        backend = config.backend.lower()
        if backend in {"dummy", "playback"}:
            return DummyTianjiBackend(
                image_hw=config.image_hw,
                max_steps=config.max_num_steps,
                contract=config.contract,
                joint_actions=config.action_space == "joint",
            )
        if backend in {"loopinfra", "http"}:
            return LoopInfraPolicyBackend(
                config.server_url,
                config.server_token,
                config.request_timeout_s,
                config.contract,
                read_only=config.read_only,
                control_profile=config.control_profile,
                reference_windows=config.reference_windows,
                # rt fork: the collector server admits each chunk row at its time.
                rt_schedule=os.environ.get("TIANJI_RT_SCHEDULE") == "1",
                joint_actions=config.action_space == "joint",
                tactile_enabled=config.tactile_enabled,
                tactile_raw=bool(
                    config.tactile_raw_archive and config.episode_archive and config.recorder_dir
                    and config.hil_mode == "training"
                    # 11-06: hil_raw records every Xense frame at 30 Hz.
                    and not _hil_raw_lean(config)
                ),
            )
        raise ValueError(
            f"unknown Tianji backend {config.backend!r}; use dummy or loopinfra"
        )

    @property
    def subtask(self) -> int:
        """Current subtask, 1-based; 0 when no subtask prompts are configured."""
        if self.hil is not None:
            return int(self.hil.snapshot()["subtask"])
        return self._subtask

    @property
    def task_description(self) -> str:
        subtask = self.subtask
        if subtask:
            return self.subtask_prompts[subtask - 1]
        return self.config.task_description

    @property
    def car_control_revision(self):
        return self.hil.snapshot()["revision"] if self.hil else 0

    @property
    def rlt_switch_flags(self):
        """Stage number for Stage2 heads (A1 semantics); None without subtasks."""
        return self.subtask or None

    def set_subtask(self, index: int):
        """Panel-less subtask switch with the panel's rules (tests, scripted sims)."""
        if self.hil is not None:
            return self.hil.select_subtask(index)
        if not self.subtask_prompts or not 1 <= index <= len(self.subtask_prompts):
            raise ValueError("subtask outside the configured prompts")
        if self.config.hil_mode == "training":
            # 11-07: a subtask already reached this episode can be re-entered directly (no reward).
            if index > max(self._subtask + 1, self._subtask_reached) or index == self._subtask:
                raise ValueError("training advances one subtask at a time")
            if index > self._subtask_reached:  # 11-06: going back allowed; a milestone counts once
                self._milestones += 1
                self._subtask_reached = index
        self._subtask = index
        return None

    def _consume_milestones(self) -> int:
        if self.config.hil_mode != "training" or not self.subtask_prompts:
            return 0
        if self.hil is not None:
            return self.hil.consume_milestones()
        count, self._milestones = self._milestones, 0
        return count

    def mark_step_failure(self):
        """Panel-less "step failure" with the panel's rules (tests, scripted sims)."""
        if self.hil is not None:
            return self.hil.mark_step_failure()
        if self.config.hil_mode != "training":
            raise ValueError("step failure is a training reward")
        self._step_failure = True
        return None

    def _consume_step_failure(self) -> bool:
        if self.config.hil_mode != "training":
            return False
        if self.hil is not None:
            return self.hil.consume_step_failure()
        marked, self._step_failure = self._step_failure, False
        return marked

    def finalize_chunk_exec(self):
        """Executed-action labels of a whole human-takeover chunk (training).

        Called by the env wrapper once the chunk has executed; None keeps the
        per-step labels, as for every chunk with a policy or pause step. In a
        chunk whose real steps are all H-mode (2) or Pico (3) steps:

        - an H-mode hold with a fresh, aligned measurement (no Pico motion)
          uses that measured pose as its executed action;
        - up to HUMAN_FILL_MAX_GAP missing steps between two labelled steps
          are interpolated in capture time (positions linearly, rotations by
          slerp) when the two labels were captured within HUMAN_FILL_MAX_SPAN_S
          and the control context, subtask, Pico grip and both gripper commands
          are unchanged;
        - if both arms and grippers stay still from the pose before the first
          step on, the chunk is no BC target, and no Q sample unless it carries
          a reward.

        Holds and filled steps serve Q only: BC keeps the measured Pico steps.
        The episode event lists each filled gap as [first step, steps, capture
        span in ms].
        """
        records = self._chunk_records
        if self.config.hil_mode != "training" or not records:
            return None
        real = [i for i, r in enumerate(records) if not r["padding"]]
        if not real or any(records[i]["iv_kind"] not in (2, 3) for i in real):
            return None
        labels = [r["exec"].copy() for r in records]
        # rtj joint Stage2: the same labels in joint space, kept step for step.
        labels_joint = [r["exec_joint"].copy() for r in records] if self.joint_actions else None
        valid = [r["available"] for r in records]
        human_bc = [r["human_bc"] for r in records]
        holds = 0
        fills = []
        for i in real:
            if not valid[i] and records[i]["hold"] is not None:
                labels[i], valid[i] = records[i]["hold"].copy(), True
                if labels_joint is not None and records[i].get("hold_joint") is not None:
                    labels_joint[i] = records[i]["hold_joint"].copy()
                holds += 1
        i = 0
        while i < len(real):
            if valid[real[i]]:
                i += 1
                continue
            j = i
            while j < len(real) and not valid[real[j]]:
                j += 1
            gap = real[i:j]
            if 0 < i and j < len(real) and len(gap) <= HUMAN_FILL_MAX_GAP:
                before, after = real[i - 1], real[j]
                span = self._fill_gap(records, labels, valid, before, gap, after, labels_joint)
                if span is not None:
                    fills.append([gap[0], len(gap), round(span * 1000.0, 1)])
            i = j
        # From the pose before the first step: motion in that step counts.
        still = _still([records[real[0]]["before_state"]] + [records[i]["state"] for i in real])
        rewarded = any(records[i]["reward"] != 0 for i in real)
        if still:
            human_bc = [False] * len(records)
            if not rewarded:
                valid = [False] * len(records)
        missing = sum(not valid[i] for i in real)
        if self._recorder is not None:
            self._recorder.record_event(
                {
                    "type": "human_chunk_labels",
                    "episode": self.hil.snapshot()["episode"] if self.hil else None,
                    "subtask": records[real[0]]["stage"],
                    "steps": len(real),
                    "holds": holds,
                    "filled": sum(steps for _, steps, _ in fills),
                    "fill_gaps": fills,
                    "missing": missing,
                    "still": still,
                    "rewarded": rewarded,
                    "q_sample": missing == 0,
                }
            )
        result = {
            "exec_action": np.stack(labels).astype(np.float32),
            "exec_action_available": np.asarray(valid, dtype=bool),
            "intervene_action_available": np.asarray(human_bc, dtype=bool),
        }
        if labels_joint is not None:
            result["exec_action_joint"] = np.stack(labels_joint).astype(np.float32)
        return result

    def _fill_gap(self, records, labels, valid, before, gap, after, labels_joint=None):
        """Interpolate the missing steps ``gap`` between two labelled steps.

        Returns the two labels' capture span in seconds, or None if unfilled.
        ``labels_joint`` (rtj joint Stage2) is filled at the same times, linearly
        in joint space (the joint RL frame is affine in the joints).
        """
        first, last = records[before], records[after]
        if any(records[k]["stage"] != first["stage"] for k in (*gap, after)):
            return None
        if first["iv_kind"] == last["iv_kind"] == 3 and first["lease"] != last["lease"]:
            return None
        # Never across a reset, fault or reconnect (session, generation, epoch).
        if first["context"] is None or first["context"] != last["context"]:
            return None
        if first["t"] is None or last["t"] is None:
            return None  # no capture time to bound the gap
        span = last["t"] - first["t"]
        if not 0 < span <= HUMAN_FILL_MAX_SPAN_S:
            return None  # e.g. a pause inside the gap: the arm may have moved
        try:
            start = self.contract.decode_action(labels[before])
            end = self.contract.decode_action(labels[after])
        except ValueError:
            return None  # a label that is no pose: leave the gap missing
        if np.abs(start[[9, 19]] - end[[9, 19]]).max() > 1e-4:
            return None  # a gripper changed inside the gap: its timing is unknown
        times = _gap_times(first["t"], last["t"], [records[k]["t"] for k in gap])
        for k, at in zip(gap, times):
            t = (at - first["t"]) / span
            pose = start.copy()
            for o in (0, 10):
                pose[o : o + 3] = (1 - t) * start[o : o + 3] + t * end[o : o + 3]
                pose[o + 3 : o + 9] = rotation_to_6d(
                    _slerp(
                        rotation_from_6d(start[o + 3 : o + 9]),
                        rotation_from_6d(end[o + 3 : o + 9]),
                        t,
                    )
                )
            labels[k] = np.clip(self.contract.encode_action(pose), -1, 1)
            if labels_joint is not None:
                labels_joint[k] = ((1 - t) * labels_joint[before] + t * labels_joint[after]).astype(np.float32)
            valid[k] = True
        return span

    @staticmethod
    def _joint_physical(observation, commanded=False):
        """(16D joints + commanded closures, source) of an observation, or (None, None).

        Measured joints (``tianji_stage1.joints``), or with ``commanded`` the
        owner's last commanded joints when it published them (LoopInfra_rtj live
        captures: the teleop IK output while Pico drives, the joint VLA's action
        semantics). The closures are the state's commanded closures.
        """
        payload = (observation.raw or {}).get("tianji_stage1") or {}
        options = [("measured_joints", payload.get("joints"))]
        if commanded:
            options.insert(0, ("commanded_joints", payload.get("commanded_joints")))
        state = np.asarray(observation.state, dtype=np.float64)
        for source, joints in options:
            if joints is None:
                continue
            try:
                return joint_physical(joints, (state[9], state[19])), source
            except (ValueError, TypeError):
                continue
        return None, None

    def _joint_label(self, observation, commanded=False):
        """Joint RL-frame label (clipped into [-1, 1]) of an observation, and its source."""
        physical, source = self._joint_physical(observation, commanded)
        if physical is None:
            return None, None
        return JOINT_CONTRACT.encode_clipped(physical), source

    def _raw_obs(self, observation, frames=None, shadow_panel=None):
        """Side-effect free conversion, also used by the prefetch reader thread."""
        payload = observation.raw.get("tianji_stage1", {})
        qvel = np.asarray(
            payload.get("joint_velocities", np.zeros(14)), dtype=np.float32
        )
        if qvel.shape != (14,) or not np.isfinite(qvel).all():
            raise ValueError("invalid aligned joint velocities")
        if frames is None:
            from .tianji_backend import parallel_read_pool

            pool = parallel_read_pool()
            if pool is not None:
                fitted = {role: pool.submit(self._fit_frame, image, role)
                          for role, image in observation.frames.items()}
                frames = {role: future.result() for role, future in fitted.items()}
            else:
                frames = {
                    role: self._fit_frame(image, role)
                    for role, image in observation.frames.items()
                }
        raw = {
            "state": {"qpos": observation.state.astype(np.float32, copy=True)},
            "frames": frames,
            "rlt_qvel": qvel,
        }
        if self._car_deployment:
            panel = shadow_panel if shadow_panel is not None else self.hil.snapshot() if self.hil else None
            sync = observation.raw.get("sync") or {}
            capture_ns = sync.get("reference_monotonic_ns")
            if sync.get("sync_valid") is not True or type(capture_ns) is not int or capture_ns <= 0:
                capture_ns = 0
            raw["car_stage_meta"] = np.array([
                capture_ns,
                panel["episode"] if panel else 0,
                panel["subtask"] if panel else self.subtask,
                panel["subtask_revision"] if panel else 0,
                panel["revision"] if panel else 0,
                int(bool(panel and panel["phase"] == "running" and not panel["human_latched"])),
            ], dtype=np.int64)
        if self._shadow_schema:
            read_ns = int(observation.captured_monotonic * 1e9) if observation.captured_monotonic else time.monotonic_ns()
            raw["shadow_meta"] = shadow.metadata(observation, shadow_panel, read_ns)
        if tianji_joint.joint_mode() or self.joint_actions:
            raw["rlt_joint_state"] = tianji_joint.joint_state(observation.raw, observation.state)
        if self.config.tactile_enabled:
            from rlinf.utils.xense_tactile import empty_tactile
            # Live execution metadata has no Rectify pixels. Its explicit
            # invalid tensors never reuse an older inference snapshot.
            tactile = getattr(observation, "tactile", None)
            raw.update({k: v.copy() for k, v in (tactile if tactile is not None else empty_tactile()).items()})
        return raw

    def _to_raw_obs(self, observation, shadow_panel=None):
        self._last_time = time.monotonic()
        self._last_state = observation.state.copy()
        raw = self._raw_obs(observation, shadow_panel=shadow_panel)
        if raw["frames"]:
            self._last_frames = raw["frames"]
        elif self.config.stream_chunks and self._last_frames is not None:
            # Metadata-only step. Its frames are never an inference input in
            # stream mode (the prefetch reader supplies those); repeating the
            # last real frames only keeps the observation layout fixed.
            raw["frames"] = self._last_frames
        return raw

    def _fit_frame(self, image: np.ndarray, role=None) -> np.ndarray:
        from .tianji_images import fit_frame

        return fit_frame(image, role, self.config.image_hw)

    def refresh_observation(self):
        """Read a new inference input after learner waits, without a step/reset."""
        if self._closed:
            raise RuntimeError("TianjiEnv is closed")
        if self._episode_done:
            return self._last_raw_obs
        if self.hil is not None:
            panel = self.hil.snapshot()
            self._observation_revision = panel["revision"]
            self._sample_revision = panel["collection_revision"]
        observation = self.backend.observe() if self._car_deployment else self._observe_step()
        if observation.captured_monotonic is None:
            observation.captured_monotonic = time.monotonic()
        if self._car_deployment:
            self._prefetch_revision = None
            self._prefetch_policy = False
            self._cancel_schedule()
        if self.episode_archive is not None:
            self.episode_archive.policy_input(observation, "refresh")
        self._last_observation = observation
        self._last_raw_obs = self._to_raw_obs(observation, shadow_panel=panel if self.hil else None)
        return self._last_raw_obs

    def _observe_step(self):
        """Control/reset reads metadata; the Stage1 request worker owns RGB reads."""
        if self.config.sparse_execution_observation:
            return self.backend.observe_execution()
        return self.backend.observe()

    def _hardware_saved_reset(self):
        return (
            isinstance(self.backend, LoopInfraPolicyBackend)
            and not self.backend.read_only
        )

    def _reset_saved_pose(self, cancelled, trigger):
        """Saved-pose Reset once both arms have stopped.

        After the last streamed step the owner's window keeps coasting until
        its input deadline, and LoopInfra rejects a Reset with
        ``wait_until_both_arms_stationary`` while an arm moves. Only that
        explicit rejection (no motion was admitted) is retried.
        """
        from .tianji_reset import reset_saved_pose

        wait_closed = getattr(self.backend, "_wait_owner_window_closed", None)
        if callable(wait_closed):
            wait_closed(timeout_s=RESET_SETTLE_S)
        deadline = time.monotonic() + RESET_SETTLE_S
        while True:
            try:
                return reset_saved_pose(
                    self.backend, cancelled=cancelled, trigger=trigger
                )
            except RuntimeError as exc:
                if (
                    "wait_until_both_arms_stationary" not in str(exc)
                    or cancelled()
                    or time.monotonic() >= deadline
                ):
                    raise
            time.sleep(0.1)

    def _ends_episode_on_fault(self):
        """Stage2 training/inference outlive a failed step: the operator is present."""
        return self.hil is not None and self.config.hil_mode in {
            "training",
            "inference",
        }

    def _on_unreachable(self, receipt, observation):
        """Pause when targets keep being unreachable; never move toward them.

        A single rejection (nothing moved) only drops the rest of its chunk
        and the next one is planned from a fresh observation. Repeated ones
        mean the arm stands at a limit: the panel pauses with the joints,
        and the operator turns the wrist back (H, Pico) or ends the episode.
        Without a panel the event is only recorded.
        """
        now = time.monotonic()
        self._unreachable_times = [
            t for t in self._unreachable_times if now - t <= UNREACHABLE_PAUSE_WINDOW_S
        ] + [now]
        joints = (observation.raw.get("tianji_stage1") or {}).get("joints") if observation else None
        joints_deg = (
            {arm: [round(float(np.degrees(q)), 1) for q in joints[k : k + 7]] for arm, k in (("A", 0), ("B", 7))}
            if joints is not None and len(joints) == 14
            else None
        )
        event = {
            "type": "target_unreachable",
            "reason": receipt.get("reason"),
            "joints_deg": joints_deg,
            "recent": len(self._unreachable_times),
        }
        if self.hil is not None and len(self._unreachable_times) >= UNREACHABLE_PAUSE_COUNT:
            where = (
                f"A 臂关节 {joints_deg['A']}°，B 臂关节 {joints_deg['B']}°。" if joints_deg else ""
            )
            message = (
                f"policy 已暂停：{len(self._unreachable_times)} 次目标无法执行（{receipt.get('reason')}，"
                "通常是关节到达限位，例如 B 臂 J5）。" + where
                + "按 H 用 Pico 把腕部转回一些后按 P 交回；或点“恢复 policy”重试；"
                "或点“失败，结束本条”返回默认位姿。"
            )
            if self.hil.pause_for_limit(message):
                event["paused"] = True
                self._unreachable_times = []
        if self._recorder is not None:
            self._recorder.record_event(event)

    def _joint_step_fields(
        self, previous, observation, normalized, receipt, policy_executed, human_sample, intervene_tcp
    ):
        """Both-space fields of one step (rtj joint Stage2).

        - exec_action_joint: a policy row's commanded joint step (receipt), a
          takeover's joint label, else the measured joints after the step;
        - intervene_joint (popped by step): a takeover's joints at the step's end,
          the owner's commanded teleop joints if published, else measured;
        - action_tcp, intervene_action_tcp, state_joint, next_state_joint.
        """
        human_joint, human_source = (
            self._joint_label(observation, commanded=True) if human_sample else (None, None)
        )
        if policy_executed and receipt.get("executed_action_joint") is not None:
            exec_joint = np.asarray(receipt["executed_action_joint"], dtype=np.float32)
            exec_source = receipt.get("executed_action_joint_source")
        elif human_sample:
            exec_joint, exec_source = human_joint, human_source
        else:
            exec_joint, exec_source = self._joint_label(observation)
        return {
            "row_semantics": JOINT_ROW_SEMANTICS,
            "state_joint": self._joint_physical(previous)[0],
            "next_state_joint": self._joint_physical(observation)[0],
            "action_tcp": joint_rl_to_tcp_rl_np(normalized),
            "exec_action_joint": exec_joint if exec_joint is not None else np.zeros(JOINT_DIM, np.float32),
            "exec_action_joint_source": exec_source,
            "intervene_action_tcp": intervene_tcp,
            "intervene_action_joint_source": human_source,
            "intervene_joint": human_joint,
        }

    def _unlabelled_joint_fields(self):
        """Joint-space fields of a step without labels (padding, fault), joint Stage2 only."""
        if not self.joint_actions:
            return {}
        return {
            "exec_action_joint": np.zeros(JOINT_DIM, dtype=np.float32),
            "intervene_action_tcp": None,
            "row_semantics": JOINT_ROW_SEMANTICS,
        }

    def _end_episode_after_fault(self, receipt, started, error=None):
        """End the episode at an incomplete step instead of the whole session.

        An owner fault, a lost receipt or a missing post-action capture leaves
        no label, so the step is padding and the episode is truncated, never
        terminal. The panel shows the reason; the next Start resets the arms
        once the operator has recovered Position/EEF in the collector.
        """
        self._episode_done = True
        self._streaming_step = False
        try:
            self.backend.hold()
        except Exception:  # noqa: BLE001 -- a faulted owner may refuse a hold
            pass
        reason = error or " ".join(
            str(part) for part in (receipt.get("status"), receipt.get("reason")) if part
        )
        self.hil.update(control_mode="fault", episode_error=reason)
        self.hil.end_episode()
        if self._recorder is not None:
            self._recorder.record_event(
                {
                    "type": "episode_fault",
                    "episode": self.hil.snapshot()["episode"],
                    "reason": reason,
                    "receipt": receipt,
                }
            )
        if self.episode_archive is not None:
            self.episode_archive.end_episode("fault", reason)
        info = {
            "receipt": receipt,
            "observation_id": self._last_observation.observation_id,
            "intervene_flag": False,
            "intervene_action": None,
            "intervene_action_available": False,
            "iv_kind": 1,
            "exec_action": np.zeros(ACTION_DIM, dtype=np.float32),
            "exec_action_available": False,
            "transition_padding": True,
            "target_grid_matches_plan": True,
            "episode_discarded": False,
            "episode_fault": reason,
            "step_started_monotonic": started,
            **self._unlabelled_joint_fields(),
        }
        if self.subtask_prompts:
            info.update(rlt_switch_flags=self.subtask, stage=self.subtask, milestone=0)
        return self._last_raw_obs, 0.0, False, True, info

    def _return_after_episode(self):
        """Return only after an explicit End, outside recorded RL transitions."""
        panel = self.hil.begin_return()
        if panel is None:
            self.backend.hold()
            return {"status": "cancelled"}
        revision = panel["revision"]

        def cancelled():
            live = self.hil.snapshot()
            return (
                self._closed
                or live["phase"] != "returning"
                or live["revision"] != revision
                or live["return_cancelled"]
            )

        result = {"status": "completed", "trigger": "end"}
        try:
            self._reset_saved_pose(cancelled, "end")
        except Exception as exc:
            result.update(
                status="failed",
                error=str(exc),
                reset_trace=getattr(exc, "reset_trace", []),
            )
            self.backend.hold()
        self.hil.finish_return(result.get("error"))
        if self._recorder is not None:
            self._recorder.record_event(
                {
                    "type": "episode_return",
                    "episode": panel["episode"],
                    "outcome": panel["outcome"],
                    **result,
                }
            )
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        auto_reset = self.config.reset_to_saved_pose and self._hardware_saved_reset()
        # The saved-pose request itself performs the owner-side hold and
        # generation handshake. Sending a separate hold immediately before it
        # would leave the client waiting for a fresh generation and make the
        # reset request fail spuriously.
        if not auto_reset:
            self.backend.reset()
        if self.hil is not None:
            self.hil.begin_episode()
        while True:
            if self.hil is not None:
                self._observation_revision = self.hil.wait_until_active()["revision"]
                self._sample_revision = self.hil.snapshot()["collection_revision"]
            if not auto_reset:
                break
            if self.hil is None:
                self._reset_saved_pose(lambda: False, "start")
                break
            revision = self._observation_revision

            def cancelled():
                panel = self.hil.snapshot()
                return (
                    self._closed
                    or panel["phase"] != "running"
                    or panel["revision"] != revision
                    or panel["human_latched"]
                )

            self.hil.update(control_mode="resetting", reset_error=None)
            try:
                self._reset_saved_pose(cancelled, "start")
                if cancelled():
                    raise RuntimeError("Episode Start cancelled during Reset")
            except Exception as exc:
                self.hil.update(control_mode="reset_failed", reset_error=str(exc))
                if self._recorder is not None:
                    self._recorder.record_event(
                        {
                            "type": "episode_start_reset",
                            "status": "failed",
                            "episode": self.hil.snapshot()["episode"],
                            "error": str(exc),
                            "reset_trace": getattr(exc, "reset_trace", []),
                        }
                    )
                if self._closed or not self._ends_episode_on_fault():
                    raise
                # Stage2 keeps its session: fix the cause, then press Start again.
                self.hil.retry_start(str(exc))
                continue
            self.hil.update(
                control_mode="hold",
                last_return_error=None,
                reset_error=None,
                episode_error=None,
            )
            break
        # Capture after the operator starts, not before scene preparation.
        observation = self._observe_step()
        if observation.captured_monotonic is None:
            observation.captured_monotonic = time.monotonic()
        self._last_observation = observation
        self._last_state = observation.state.copy()
        self._last_time = time.monotonic()
        self._step = 0
        self._episode_done = False
        self._streaming_step = False
        self._unreachable_times = []
        if not self.hil and self.subtask_prompts:
            self._subtask, self._milestones = 1, 0
            self._subtask_reached = 1
        if not self.hil:
            self._step_failure = False
        if self.episode_archive is not None:
            self.episode_archive.begin_episode(
                self.hil.snapshot()["episode"] if self.hil else None,
                observation,
                subtask=self.subtask,
            )
        self.on_action_chunk_begin()
        self._ramp_next_chunk = True
        self._last_raw_obs = self._to_raw_obs(
            observation, shadow_panel=self.hil.snapshot() if (self._shadow_schema or self._car_deployment) and self.hil else None
        )
        return self._last_raw_obs, {"observation_id": observation.observation_id}

    def create_stage1_reader(self):
        """Return a worker-owned read-only input reader for Stage1."""
        factory = getattr(self.backend, "create_stage1_reader", None)
        if factory is None:
            raise RuntimeError("Tianji backend has no Stage1 observation reader")
        return factory()

    @staticmethod
    def _control_context(observation):
        if observation is None:
            return None
        raw = observation.raw
        return (
            raw.get("session_id"),
            raw.get("control_generation"),
            raw.get("robot_control", {}).get("stream_epoch"),
        )

    def _reference_cancelled(self):
        if self.hil is None:
            return False
        panel = self.hil.snapshot()
        return bool(
            panel["revision"] != self._chunk_revision
            or panel["phase"] != "running"
            or panel.get("human_latched")
        )

    def set_action_window_steps(self, steps: int):
        """Declare the executed slice independently of the model horizon."""
        horizon = self.contract.action_chunk_length
        if type(steps) is not int or not 1 <= steps <= horizon:
            raise ValueError(f"policy window must contain 1..{horizon} steps")
        self._next_window_steps = steps

    def on_action_chunk_end(self):
        """Record a passive endpoint check outside Stage1 transitions."""
        if self.hil is not None:
            panel = self.hil.snapshot()
            if panel["phase"] != "running" or panel.get("human_latched"):
                self.backend.hold()
                return None
        finish = getattr(self.backend, "finish_policy_window", None)
        result = finish() if finish is not None else None
        if result is not None and self._recorder is not None:
            self._recorder.record_event({"type": "policy_window_end", **result})
        return result

    def streams_action_chunks(self) -> bool:
        """True when the SDK owner can continue one window across chunks."""
        streaming = getattr(self.backend, "streaming", None)
        return bool(streaming is not None and streaming())

    def on_action_chunk_begin(self, *, continue_window: bool | None = None):
        """Discard a whole remaining chunk after a takeover or operator pause.

        ``continue_window`` hands the open streaming reference window to the
        next chunk instead of waiting for it to stop and converge. Unset, a
        stream-mode env continues whenever the previous step was a policy
        segment of a stream-capable owner; a late chunk is still safe because
        the owner then closes the window and the next target restarts it.
        """
        if continue_window is None:
            continue_window = bool(
                self.config.stream_chunks
                and self._streaming_step
                and self.streams_action_chunks()
            )
        if not continue_window:
            self.on_action_chunk_end()
        self._window_remaining = self._next_window_steps
        # Steps the caller will submit in this chunk (executed or not).
        self._chunk_steps_left = self._next_window_steps
        self._next_window_steps = None
        self._skip_chunk = False
        self._chunk_actions = None
        self._chunk_row = 0
        self._offered_next = None
        self._next_appended = False
        self._ramp = None
        self._chunk_records = []
        self._chunk_context = self._control_context(self._last_observation)
        self._chunk_observation = self._last_observation
        # Bind to the observation sent for inference. Reading the live panel
        # revision here would miss pause/resume while inference was in flight.
        # In stream mode that is the latest prefetch capture.
        revision = self._prefetch_revision if self.config.stream_chunks else None
        self._prefetch_revision = None
        self._chunk_revision = (
            revision if revision is not None else self._observation_revision
        )
        # A request read while the arm held was planned without RTC: its chunk
        # ramps in from the held joints (resume_chunk_ramp_steps).
        self._ramp_resume_chunk = bool(self.config.stream_chunks and self._prefetch_policy is False)
        self._prefetch_policy = None

    def set_chunk_actions(self, actions):
        """rt fork: the whole chunk, so the server can run its rows on a fixed grid
        (rtj joint Stage2: 16D joint rows, each sent to the owner as joint targets)."""
        self._chunk_actions = np.stack(
            [
                self.action_contract.validate_action(np.asarray(a, dtype=np.float32).reshape(-1))
                for a in np.asarray(actions, dtype=np.float32)
            ]
        )
        self._chunk_row = 0
        if self._ramp_next_chunk or self._ramp_resume_chunk:
            first = self._ramp_next_chunk
            self._ramp_next_chunk = self._ramp_resume_chunk = False
            self._ramp = (
                self._first_chunk_ramp()
                if first
                else self._measured_ramp(self.config.resume_chunk_ramp_steps)
            )
            if self._ramp is not None:
                self._chunk_actions = np.stack(
                    [self._ramped(row, a) for row, a in enumerate(self._chunk_actions)]
                )
        forget = getattr(self.backend, "forget_schedule", None)
        if forget is not None:
            forget()

    def _first_chunk_ramp(self):
        """(measured joints in the RL frame, rows) for the episode's first chunk, or None."""
        return self._measured_ramp(self.config.first_chunk_ramp_steps)

    def _measured_ramp(self, steps):
        """(measured joints in the RL frame, rows) for a chunk that starts from a held arm, or None.

        At most the chunk's rows: the next request's RTC prefix is the plan's own
        rows, so the ramp has to reach the plan by this chunk's last row.
        """
        steps = int(steps or 0)
        if self._chunk_actions is not None:
            steps = min(steps, len(self._chunk_actions))
        if steps <= 0 or not self.joint_actions or self._last_observation is None:
            return None
        start, _ = self._joint_label(self._last_observation)
        if start is None:
            return None
        return np.asarray(start, dtype=np.float32), steps

    def _ramped(self, row, action):
        """Blend row ``row`` of a ramped chunk from the measured joints to the plan."""
        if self._ramp is None or row >= self._ramp[1]:
            return action
        start, steps = self._ramp
        u = (row + 1) / steps
        weight = u * u * (3.0 - 2.0 * u)
        out = np.array(action, dtype=np.float32, copy=True)
        arm = list(JOINT_DIMS)
        out[arm] = start[arm] + weight * (out[arm] - start[arm])
        return out

    def offer_next_chunk(self, actions):
        """rt fork: the next chunk as the env worker will hand it over (main thread).

        Received while this chunk runs; the env thread queues it behind this
        chunk on the owner's grid at its next policy step.
        """
        if self.hil is not None and self.hil.monitor_owns_boundaries():
            return
        self._offered_next = np.stack(
            [
                self.action_contract.validate_action(np.asarray(a, dtype=np.float32).reshape(-1))
                for a in np.asarray(actions, dtype=np.float32)
            ]
        )

    def offer_dense_rows(self, actions, dense):
        """rtj dense rows (11-05): every model row of a chunk, k per env row, beside the env rows
        the chunk is handed as (set_chunk_actions / offer_next_chunk); the backend submits them
        as that chunk's schedule. Nothing else of the env changes."""
        offer = getattr(self.backend, "offer_dense_rows", None)
        if offer is None:
            return
        validate = self.action_contract.validate_action
        offer(
            np.stack([validate(np.asarray(a, dtype=np.float32).reshape(-1)) for a in np.asarray(actions, dtype=np.float32)]),
            np.stack([validate(np.asarray(a, dtype=np.float32).reshape(-1)) for a in np.asarray(dense, dtype=np.float32)]),
        )

    def _append_offered_chunk(self, receipt, observation):
        """Queue the offered next chunk once, while this chunk runs as policy."""
        offered = self._offered_next
        if (
            offered is None
            or self._next_appended
            or self._skip_chunk
            or receipt.get("status") not in ("executed_hardware", "executed_fake")
        ):
            return
        self._next_appended = True
        append = getattr(self.backend, "append_scheduled_chunk", None)
        if append is not None:
            append(offered, observation)

    def _scheduled_row(self, row, normalized) -> bool:
        """True when the server runs this chunk and ``row`` is its matching action."""
        schedules = getattr(self.backend, "schedules_rows", None)
        return bool(
            self.config.stream_chunks
            and self._chunk_actions is not None
            and row < len(self._chunk_actions)
            and np.array_equal(self._chunk_actions[row], normalized)
            and callable(schedules)
            and schedules()
        )

    def _cancel_schedule(self):
        cancel = getattr(self.backend, "cancel_schedule", None)
        if cancel is not None:
            cancel()

    def capture_observation_async(self, remaining_actions=None, deliver=None):
        """Read the next request's RGB input without blocking the control loop.

        An independent read-only client reads it on a worker thread, which
        also runs ``deliver(raw, extras)`` before the returned future
        completes. The result carries the actions of this chunk that still
        execute while the request is in flight; they are an RTC target only
        while this chunk is policy-executed (not skipped, held, paused or
        under human control).
        Outside stream mode this returns None and the caller reads inline.
        After the episode ended in this chunk it returns False: the request
        then waits for the chunk-end (auto-reset) observation, so no chunk
        planned from the old episode is ever executed in the next one.
        """
        if not self.config.stream_chunks:
            return None
        if self._episode_done:
            return False
        if self._prefetch_pool is None:
            self._prefetch_reader = self.create_stage1_reader()
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="tianji-prefetch"
            )
        panel = self.hil.snapshot() if self.hil is not None else None
        if panel is not None and (
            self._car_deployment or self.hil.monitor_owns_boundaries()
        ) and panel["revision"] != self._chunk_revision:
            self._skip_chunk = True
        # Bind language and stage before the RGB read can overlap a panel change.
        subtask = int(panel["subtask"]) if panel is not None else self._subtask
        task_description = (
            self.subtask_prompts[subtask - 1] if subtask else self.config.task_description
        )
        self._prefetch_revision = (
            panel["revision"] if panel is not None else self._observation_revision
        )
        policy = bool(
            self._streaming_step
            and not self._skip_chunk
            and not self._episode_done
            and (panel is None or (panel["phase"] == "running" and not panel["human_latched"]))
        )
        # Bound to the chunk this request plans (on_action_chunk_begin).
        self._prefetch_policy = policy
        # In the action space: joint Stage2's RTC guides in joints.
        action_dim = self.action_contract.action_dim
        remaining = np.zeros((0, action_dim), dtype=np.float32)
        if remaining_actions is not None:
            remaining = np.asarray(remaining_actions, dtype=np.float32).reshape(-1, action_dim)
        extras = {
            "rtc_prev_actions": remaining,
            "rtc_prev_valid": self.config.rtc_guidance and policy and len(remaining) > 0,
            "rlt_switch_flags": subtask or None,
            "task_description": task_description,
            "intervene_flag": not policy,
        }
        reader = self._prefetch_reader

        def read():
            observation = reader.observe()
            if self.episode_archive is not None:
                self.episode_archive.policy_input(observation, "prefetch")
            raw = self._raw_obs(observation, shadow_panel=panel)
            if not raw["frames"]:
                raise RuntimeError("prefetch observation has no RGB frames")
            self._last_frames = raw["frames"]
            if deliver is not None:
                deliver(raw, extras)
            return raw, extras

        return self._prefetch_pool.submit(read)

    def step(self, action):
        if self._closed:
            raise RuntimeError("TianjiEnv is closed")
        started = time.monotonic()
        normalized = self.action_contract.validate_action(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        row, self._chunk_row = self._chunk_row, self._chunk_row + 1
        # The first chunk's ramp (set_chunk_actions applied it to the server's copy).
        normalized = self._ramped(row, normalized)
        if self._episode_done:
            # Virtual trailing steps preserve the terminal observation. Scene
            # preparation after a result must not enter this transition.
            padding = {
                "receipt": {
                    "status": "terminal_padding",
                    "hardware_executed": False,
                },
                "observation_id": self._last_observation.observation_id,
                "intervene_flag": False,
                "intervene_action": None,
                "intervene_action_available": False,
                "iv_kind": 1,
                "exec_action": np.zeros(ACTION_DIM, dtype=np.float32),
                "exec_action_available": False,
                "transition_padding": True,
                "target_grid_matches_plan": True,
                "step_failure": False,
                "episode_discarded": False,
                **self._unlabelled_joint_fields(),
            }
            if self.subtask_prompts:
                padding.update(rlt_switch_flags=self.subtask, stage=self.subtask, milestone=0)
            if self.config.hil_mode == "training":
                self._chunk_records.append({"padding": True, "iv_kind": 1, "available": False, "human_bc": False, "exec": padding["exec_action"],
                                            "exec_joint": padding.get("exec_action_joint")})
            return self._last_raw_obs, 0.0, False, False, padding
        previous = (
            self._last_observation
            if self._last_observation is not None
            else self._observe_step()
        )
        outcome = None
        human_latched = False
        if self.hil is not None and not self._episode_done:
            state = self.hil.snapshot()
            if state["phase"] == "paused":
                self.backend.hold()
                self.hil.update(control_mode="paused")
                state = self.hil.wait_until_active()
                self._skip_chunk = True
            if state["revision"] != self._chunk_revision:
                self._skip_chunk = True
            human_latched = bool(state.get("human_latched", False))
            outcome = state["outcome"]
        if self._control_context(previous) != self._chunk_context:
            self._skip_chunk = True
        # Recheck the panel immediately before a policy submission. H/P may
        # be pressed while inference or observation processing is in flight.
        if self.hil is not None and not self._episode_done:
            live_state = self.hil.snapshot()
            if live_state["revision"] != self._chunk_revision:
                self._skip_chunk = True
            human_latched = bool(live_state.get("human_latched", human_latched))
        if human_latched:
            # H latches the environment in human mode. Pico's lease still
            # decides whether the operator is physically moving the robot;
            # without that lease this remains a no-command hold step.
            self.backend.hold()
            receipt = {"status": "human_latched_hold", "hardware_executed": False}
            self._skip_chunk = True
        elif self._episode_done or outcome is not None or self._skip_chunk:
            # rt fork: no server row may run on after a held step.
            self._cancel_schedule()
            receipt = {"status": "hil_hold", "hardware_executed": False}
        else:
            try:
                self.action_contract.decode_action(normalized)
            except ValueError as exc:
                self._cancel_schedule()
                receipt = {
                    "status": "invalid_target",
                    "reason": str(exc),
                    "hardware_executed": False,
                }
            else:
                configure = getattr(self.backend, "set_reference_final", None)
                if configure is not None:
                    # A stream continues into the next chunk; only an
                    # interruption (hold) or a late chunk ends its window.
                    configure(
                        not self.config.stream_chunks and self._window_remaining == 1
                    )
                latest = (
                    {"latest": previous}
                    if getattr(self.backend, "accepts_latest_observation", False)
                    else {}
                )
                if self._scheduled_row(row, normalized):
                    receipt = self.backend.send_scheduled_action(
                        normalized,
                        row,
                        self._chunk_actions,
                        self._chunk_observation or previous,
                    )
                    self._append_offered_chunk(receipt, self._chunk_observation or previous)
                else:
                    self._cancel_schedule()
                    receipt = self.backend.send_action(
                        normalized, self._chunk_observation or previous, **latest
                    )
                if (
                    receipt.get("status") in ("executed_hardware", "executed_fake")
                    and self._window_remaining is not None
                ):
                    self._window_remaining -= 1
                if unreachable_target(receipt):
                    self._on_unreachable(receipt, previous)
        # rt fork: the capture at the next row's time, read with this row.
        take = getattr(self.backend, "take_scheduled_observation", None)
        scheduled_observation = take() if take is not None else None
        # A fault can invalidate commanded gripper state and the next capture.
        # Preserve the terminal receipt before any observation conversion can
        # mask it with HTTP 409. No incomplete step becomes a training label.
        if receipt.get("status") in {
            "aborted_hardware",
            "aborted_fake",
            "receipt_timeout",
            "invalid_terminal_receipt",
        }:
            self._skip_chunk = True
            if self.hil is not None:
                self.hil.update(
                    last_receipt=receipt.get("status"),
                    last_receipt_reason=receipt.get("reason"),
                )
            if self._ends_episode_on_fault():
                return self._end_episode_after_fault(receipt, started)
            raise TianjiStepError("Tianji action was not completed", receipt)
        last_of_chunk = False
        if self._chunk_steps_left is not None:
            self._chunk_steps_left -= 1
            last_of_chunk = self._chunk_steps_left <= 0
        # Keep manual/held steps at the configured rate instead of spinning.
        # A streamed chunk's last step skips it: the next chunk's first target
        # waits for the env worker's chunk turnover anyway, and its segment
        # must arrive before this one stops coasting (45 ms after the receipt).
        # A takeover step (H mode or an active Pico) keeps it: no policy
        # segment coasts, and its action is measured against the next capture,
        # which right away would still be the previous frame.
        takeover_step = human_latched or receipt.get("status") == "human_active"
        # rt fork: a scheduled row already returned at the next row's time;
        # the server keeps the period, so no sleep here.
        if scheduled_observation is None and not (
            last_of_chunk and self.config.stream_chunks and not takeover_step
        ):
            # Paced on absolute time: the observation and bookkeeping below no
            # longer add to each period (takeover steps ran ~40 ms, stretching
            # each human chunk ~20% against the 30 Hz timeline). A late step
            # restarts the grid instead of catching up.
            period = 1.0 / self.config.step_frequency
            now = time.monotonic()
            due = started + period if self._hold_due is None else self._hold_due + period
            due = max(due, now) if self._hold_due is not None else due
            time.sleep(max(0.0, due - now))
            self._hold_due = due
        else:
            self._hold_due = None
        self._observation_revision = self.hil.snapshot()["revision"] if self.hil else 0
        try:
            if scheduled_observation is not None:
                observation = scheduled_observation
            elif self.config.sparse_execution_observation or self.config.stream_chunks:
                observe_execution = getattr(self.backend, "observe_execution", None)
                if observe_execution is None:
                    raise RuntimeError(
                        "sparse execution requires metadata-only observation support"
                    )
                observation = observe_execution()
            else:
                observation = self.backend.observe()
        except Exception as exc:
            self._skip_chunk = True
            if self._ends_episode_on_fault():
                return self._end_episode_after_fault(
                    receipt, started, f"post-action observation unavailable ({exc})"
                )
            raise TianjiStepError(
                f"Post-action observation unavailable ({exc})", receipt, str(exc)
            ) from exc
        if observation.captured_monotonic is None:
            observation.captured_monotonic = time.monotonic()
        if not self._episode_done:
            self._step += 1
        # Commit only after the last action in the current chunk returned.
        # A prefetched old prompt retains its original motion revision and is held.
        if last_of_chunk and self.hil is not None and self.hil.apply_monitor_boundary():
            self._observation_revision = self.hil.snapshot()["revision"]
            self._cancel_schedule()
        car_panel = self.hil.snapshot() if self._car_deployment and self.hil else None
        if car_panel is not None:
            self._observation_revision = car_panel["revision"]
        raw_obs = self._to_raw_obs(observation, shadow_panel=car_panel)
        manual = receipt.get("status") == "human_active" or self._manual_status(
            observation.status
        )
        sample_revision = self.hil.snapshot()["collection_revision"] if self.hil else 0
        collection_active = sample_revision == self._sample_revision
        self._sample_revision = sample_revision
        sample = (
            observed_action(
                previous,
                observation,
                contract=self.contract,
                max_interval_s=self.config.human_action_max_interval_s,
            )
            if collection_active
            else {"available": False, "reason": "episode_paused_or_revised"}
        )
        human_sample = bool(sample.get("available") and sample.get("human"))
        admission_retry = retryable_admission(receipt)
        manual = manual or human_sample
        intervene_action_available = human_sample
        intervene_action = (
            np.asarray(sample["bounded_action"], dtype=np.float32)
            if human_sample
            else None
        )
        executed = receipt.get("executed_action")
        # rtj dense rows (11-05): an executed dense row without an executed label (its gripper
        # command had not gone out before the next row ended it) is still the policy's step: it
        # neither holds the rest of the chunk nor ends the stream; it has no executed label.
        dense_unlabelled = bool(executed is None and receipt.get("dense_row_unlabelled"))
        policy_executed = bool((executed is not None or dense_unlabelled) and not manual)
        exec_available = policy_executed and collection_active and not dense_unlabelled
        if (
            not policy_executed
            and (not admission_retry or human_sample)
            and sample.get("available")
            and (human_sample or (not manual and not human_latched))
        ):
            # Keep the unprojected execution separate from the bounded BC
            # target. Out-of-range human motion is BC-only, never a fake Q label.
            executed = sample["normalized_action"]
            exec_available = bool(sample["exec_action_available"])
        joint_fields = {}
        if self.joint_actions:
            # rtj joint Stage2: the critic keeps the measured TCP labels above;
            # the actor's space (and its human BC target) is joints.
            joint_fields = self._joint_step_fields(
                previous, observation, normalized, receipt, policy_executed, human_sample, intervene_action
            )
            intervene_action = joint_fields.pop("intervene_joint")
            if human_sample and intervene_action is None:
                intervene_action_available = False  # no joints: no joint BC target
        if manual or (not policy_executed and not admission_retry):
            self._skip_chunk = True
        if self._control_context(observation) != self._chunk_context:
            self._skip_chunk = True
        # 2 marks H-mode steps without a Pico sample, distinct from pause or
        # terminal holds (1), so a takeover penalty never counts a pause.
        iv_kind = (
            3
            if human_sample
            else 0
            if policy_executed
            else 2
            if manual or human_latched
            else 1
        )
        if self.hil is not None and not self._episode_done:
            # A result entered while HTTP execution was pending belongs to
            # this transition, and must stop the rest of this chunk.
            outcome = self.hil.snapshot()["outcome"]
        reward = float(outcome == "success" and not self._episode_done)
        milestones = self._consume_milestones()
        reward += MILESTONE_REWARD * milestones
        step_failure = self._consume_step_failure()
        reward += STEP_FAILURE_REWARD * step_failure
        terminated = not self._episode_done and (
            outcome is not None or bool(receipt.get("success", False))
        )
        truncated = (
            not self._episode_done
            and not self.config.manual_episode_control_only
            and self._step >= self.config.max_num_steps
            and not terminated
        )
        return_after_end = bool(
            terminated
            and outcome is not None
            and self.hil is not None
            and self.config.reset_after_episode
            and self._hardware_saved_reset()
        )
        if terminated or truncated:
            self._episode_done = True
            self._cancel_schedule()
            if not return_after_end:
                self.backend.hold()
            if self.hil is not None:
                self.hil.end_episode()
        # The panel's Discard ends the episode like Failure; the env worker
        # keeps every chunk of it from training (env.train.discard_episodes).
        episode_discarded = bool(terminated and outcome == "discard")
        if episode_discarded and self._recorder is not None:
            self._recorder.record_event(
                {
                    "type": "episode_discarded",
                    "episode": self.hil.snapshot()["episode"] if self.hil else None,
                    "steps": self._step,
                }
            )
        info = {
            "receipt": receipt,
            "observation_id": observation.observation_id,
            # Mark the whole latched interval as intervention-controlled so
            # rollout workers do not keep producing policy actions. The BC
            # label remains separately gated by intervene_action_available,
            # which stays false while Pico is released.
            "intervene_flag": bool(manual or human_latched),
            "intervene_action": intervene_action,
            "intervene_action_available": intervene_action_available,
            "intervene_state": receipt.get("intervene_state"),
            "iv_kind": iv_kind,
            "exec_action": np.asarray(
                executed if executed is not None else np.zeros(ACTION_DIM),
                dtype=np.float32,
            ),
            "exec_action_available": exec_available,
            "transition_padding": False,
            # Startup/resume ramps and a dense fallback change the commanded
            # knots. Do not label those chunks as execution of the saved plan.
            "target_grid_matches_plan": bool(
                not policy_executed or (
                    self._ramp is None and (
                        not getattr(self.backend, "dense_rows", 0)
                        or receipt.get("dense_rows") == self.backend.dense_rows
                    )
                )
            ),
            "observed_action": sample,
            "gripper_actual_available": False,
            "control_mode": "human"
            if manual
            else "human_latched"
            if human_latched
            else "policy"
            if policy_executed
            else "hold",
            "human_latched": human_latched,
            "hil_outcome": outcome,
            "step_failure": step_failure,
            "episode_discarded": episode_discarded,
            "observation_images_available": bool(previous.frames),
            "next_observation_images_available": bool(observation.frames),
            # Monotonic step start: recorded so chunk-boundary gaps are visible.
            "step_started_monotonic": started,
            **joint_fields,
        }
        if self.subtask_prompts:
            # A1 semantics: this step's stage includes a switch made during it.
            info.update(
                rlt_switch_flags=self.subtask,
                stage=self.subtask,
                milestone=milestones,
                task_description=self.task_description,
            )
        self._streaming_step = bool(
            policy_executed and not self._skip_chunk and not (terminated or truncated)
        )
        if self.config.hil_mode == "training":
            # An H-mode hold measured without Pico motion: a candidate executed
            # action for a takeover chunk's Q label (finalize_chunk_exec).
            hold = (
                np.asarray(sample["normalized_action"], dtype=np.float32)
                if iv_kind == 2
                and sample.get("available")
                and not sample.get("human")
                and not sample.get("clipped")
                else None
            )
            self._chunk_records.append(
                {
                    "padding": False,
                    "iv_kind": iv_kind,
                    "available": bool(exec_available),
                    "human_bc": bool(intervene_action_available),
                    "exec": info["exec_action"],
                    "exec_joint": info.get("exec_action_joint"),
                    "hold": hold,
                    # The same measured hold in joint space (joint Stage2).
                    "hold_joint": self._joint_label(observation)[0]
                    if hold is not None and self.joint_actions
                    else None,
                    "lease": sample.get("lease_id"),
                    "stage": self.subtask,
                    "state": np.asarray(observation.state, dtype=np.float32),
                    "before_state": np.asarray(previous.state, dtype=np.float32),
                    "t": _capture_time(observation),
                    "context": self._control_context(observation),
                    "reward": float(reward),
                }
            )
        if self.hil is not None:
            self.hil.update(
                step=self._step,
                control_mode=info["control_mode"],
                last_receipt=receipt.get("status"),
                last_receipt_reason=receipt.get("reason"),
                observation_id=observation.observation_id,
            )
        recording_started = time.monotonic()
        info["step_before_record_seconds"] = recording_started - started
        if self._recorder is not None:
            line = self._recorder.append(
                observation=previous,
                action=normalized,
                next_observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            if self.episode_archive is not None:
                self.episode_archive.add_step(line, previous, observation, info)
                if terminated or truncated:
                    # Before the return motion, which is outside the episode.
                    self.episode_archive.end_episode(
                        outcome or ("terminated" if terminated else "truncated")
                    )
        info["record_seconds"] = time.monotonic() - recording_started
        info["step_seconds"] = time.monotonic() - started
        if self.hil is not None:
            self.hil.update(
                last_step_seconds=info["step_seconds"],
                last_record_seconds=info["record_seconds"],
            )
        self._last_observation = observation
        self._last_raw_obs = raw_obs
        # Preserve terminal state/reward and the recorder row BEFORE the return
        # motion. Padding in a Stage2 chunk must retain this terminal snapshot.
        if return_after_end:
            info["episode_return"] = self._return_after_episode()
        return raw_obs, reward, terminated, truncated, info

    @staticmethod
    def _manual_status(status: Mapping[str, Any]) -> bool:
        manual = status.get("manual") if isinstance(status, Mapping) else None
        if isinstance(manual, Mapping):
            return True
        control = status.get("robot_control") if isinstance(status, Mapping) else None
        if isinstance(control, Mapping):
            teleop = control.get("teleop")
            return bool(
                control.get("mode") == "drag"
                or (isinstance(teleop, Mapping) and teleop.get("active"))
            )
        return False

    def step_hold(self):
        """Collect one transition without submitting a policy command."""
        self._skip_chunk = True
        return self.step(self.get_hold_action())

    def get_hold_action(self, fallback=None):
        del fallback
        if self.joint_actions:
            # rtj joint Stage2: hold = the measured joints and commanded closures.
            label = (
                self._joint_label(self._last_observation)[0]
                if self._last_observation is not None
                else None
            )
            if label is None:
                raise RuntimeError("a joint hold needs an observation with measured joints")
            return label
        state = (
            self._last_observation.state
            if self._last_observation is not None
            else identity_state()
        )
        return self.contract.encode_action(state)

    def close(self):
        if not self._closed:
            self._closed = True
            if self.hil is not None:
                self.hil.close()
            if self._prefetch_pool is not None:
                # A late RGB read is discarded; it owns no command state.
                self._prefetch_pool.shutdown(wait=False, cancel_futures=True)
                self._prefetch_reader.close()
            try:
                self.backend.close()
            finally:
                if self.episode_archive is not None:
                    self.episode_archive.close()
                if self._recorder is not None:
                    self._recorder.close()
