# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Single-flight Stage1 receding-horizon execution.

The execution thread owns the environment, action ids, HIL state and receipt
matching. A worker owns one read-only observation reader and the frozen GPU
predictor. This separation lets a slow RGB request or model call run while an
already-valid action window continues, without sharing command ownership or
relabelling a prediction with a newer observation.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rlinf.utils.rtc import RTCChunkSchedule

from ..tianji_backend import retryable_admission, unreachable_target
from ..tianji_contract import DEFAULT_CONTRACT, TASK_PROMPT, prefetch_schedule
from .contract import validate_response

# There is deliberately no earlier ("about two seconds") trigger.

# Measured on tianji2: about 0.3 s (3 steps at 10 Hz) executes while one
# request is in flight; the schedule then tracks the observed delays.
INITIAL_DELAY_STEPS = DEFAULT_CONTRACT.initial_delay_steps()


@dataclass
class InferenceInput:
    chunk: int
    epoch: int
    revision: int
    submitted_monotonic: float
    trigger: str
    completed_before_request: int
    attempts_before_request: int
    episode: int | None = None
    observation: object | None = None
    captured_monotonic: float | None = None
    discard_reason: str | None = None
    # RTC: global completed-step count and active chunk index at request time,
    # the predicted delay and the active chunk's physical absolute targets.
    completed_global: int = 0
    executed_horizon: int = 0
    delay_steps: int = 0
    prev_physical: np.ndarray | None = None
    # The language input: the global prompt or the current subtask's prompt.
    prompt: str = TASK_PROMPT
    subtask: int = 0
    subtask_revision: int = 0
    rtc_hard_only: bool = False


@dataclass
class RTCGuidanceInput:
    """Previous chunk for RLinf RTC guidance, in the physical absolute frame.

    Model-space actions are relative to the inference state, so the predictor
    maps these targets into the new request's model space itself.
    ``hard_only`` keeps only the in-flight rows (after a subtask switch).
    """

    prev_physical: np.ndarray
    executed_horizon: int
    delay_steps: int
    hard_only: bool = False


def subtask_state(env) -> tuple[int, int]:
    """(current subtask, its revision); (0, 0) without subtask prompts."""
    if env.hil is not None:
        panel = env.hil.snapshot()
        return int(panel.get("subtask", 0)), int(panel.get("subtask_revision", 0))
    subtask = int(getattr(env, "subtask", 0) or 0)
    return subtask, subtask


def source_age(env, request: InferenceInput) -> float:
    """Return age on the executor host, never subtracting remote clocks."""
    if request.captured_monotonic is None:
        return max(0.0, time.monotonic() - request.submitted_monotonic)
    return max(0.0, time.monotonic() - request.captured_monotonic)


def _source_identity(request: InferenceInput) -> dict:
    observation = request.observation
    if observation is None:
        return {"observation_id": None, "episode": request.epoch}
    raw = observation.raw
    control = raw.get("robot_control") or {}
    grippers = raw.get("collection_grippers") or {}
    return {
        "observation_id": observation.observation_id,
        "source_reference_time_ns": observation.timestamp_ns,
        "session_id": raw.get("session_id"),
        "control_generation": raw.get("control_generation"),
        "stream_epoch": control.get("stream_epoch"),
        "pose_config_id": control.get("pose_config_id"),
        "teleop_config_id": control.get("teleop_config_id"),
        "gripper_config_id": grippers.get("config_id"),
        "episode": (
            observation.raw.get("episode", request.episode)
            if observation is not None
            else request.episode
        ),
        "hil_revision": request.revision,
    }


def invalid_reason(env, request: InferenceInput, epoch: int) -> str | None:
    """Invalidate on identity/HIL changes; wall time never shortens a chunk."""
    if request.discard_reason:
        return request.discard_reason
    if request.epoch != epoch:
        return "episode_changed"
    if env.hil is not None:
        live_episode = env.hil.snapshot().get("episode")
        if request.episode is not None and live_episode != request.episode:
            return "episode_changed"
    if request.observation is None:
        return "input_unavailable"
    if env._control_context(request.observation) != env._control_context(
        env._last_observation
    ):
        return "control_context_changed"
    panel = env.hil.snapshot() if env.hil else None
    if panel and (
        panel["revision"] != request.revision
        or panel["phase"] != "running"
        or panel.get("human_latched")
    ):
        return "hil_changed"
    if env._manual_status(env._last_observation.status):
        return "human_active"
    return None


def bind_input(env, request: InferenceInput) -> None:
    """Bind commands to this request's source identity, never the latest one."""
    if request.observation is None:
        raise RuntimeError("cannot bind an inference without an observation")
    env._skip_chunk = False
    env._chunk_observation = request.observation
    env._chunk_context = env._control_context(request.observation)
    env._chunk_revision = request.revision


def predict(predictor, state, frames, execute_steps, rtc_context=None, prompt=TASK_PROMPT, observation=None):
    """Run one frozen model request and validate the complete model chunk.

    Predictors with ``predict_frames_rtc`` accept RLinf RTC guidance from the
    previous chunk and return a guidance summary.
    """
    started = time.monotonic()
    rtc = None
    # Joint fork: the joint predictor reads the aligned joints of its source capture.
    extra = {"observation": observation} if getattr(predictor, "joint", False) else {}
    if hasattr(predictor, "predict_frames_rtc"):
        physical, rtc = predictor.predict_frames_rtc(
            state, frames, prompt, rtc_context, **extra
        )
    else:
        physical = predictor.predict_frames(state, frames, prompt)
    physical = validate_response(physical)
    encoded = DEFAULT_CONTRACT.encode_action(physical)
    # Validate the complete model output before exposing any execution slice.
    all_actions = [DEFAULT_CONTRACT.validate_action(a) for a in encoded]
    actions = all_actions[:execute_steps]
    return physical, actions, rtc, time.monotonic() - started


def read_and_predict(reader, predictor, execute_steps, request: InferenceInput):
    """Read RGB/state and infer in the worker, preserving capture timing."""
    read_started = time.monotonic()
    observation = reader.observe()
    captured = observation.captured_monotonic or time.monotonic()
    request.observation = observation
    request.captured_monotonic = captured
    read_seconds = time.monotonic() - read_started
    state = observation.state.copy()
    frames = {key: value.copy() for key, value in observation.frames.items()}
    rtc_context = None
    if request.prev_physical is not None:
        rtc_context = RTCGuidanceInput(
            prev_physical=request.prev_physical,
            executed_horizon=request.executed_horizon,
            delay_steps=request.delay_steps,
            hard_only=request.rtc_hard_only,
        )
    physical, actions, rtc, inference_seconds = predict(
        predictor, state, frames, execute_steps, rtc_context, request.prompt, observation
    )
    return {
        "observation": observation,
        "physical": physical,
        "actions": actions,
        "rtc": rtc,
        "read_seconds": read_seconds,
        "inference_seconds": inference_seconds,
        "processing_seconds": max(
            0.0,
            time.monotonic() - read_started - read_seconds - inference_seconds,
        ),
    }


def _completed_receipt(info: dict) -> bool:
    receipt = info.get("receipt") or {}
    return receipt.get("status") in {
        "executed_hardware",
        "executed_fake",
        "executed_dummy",
    }


def _source_expired_receipt(info: dict) -> bool:
    """Legacy diagnostic helper; capture expiry is handled as admission retry."""
    return (info.get("receipt") or {}).get("reason") == "source_observation_expired"


def _control_interruption_receipt(info: dict) -> bool:
    receipt = info.get("receipt") or {}
    return receipt.get("status") in {
        "human_active",
        "human_latched_hold",
        "hil_hold",
    }


def _save_inference_input(
    request: InferenceInput, directory: Path | None
) -> str | None:
    """Save only this request's real input; never reuse an older frame."""
    observation = request.observation
    if directory is None or observation is None or not observation.frames:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{request.chunk:08d}.npz"
    arrays = {}
    jpeg = observation.images_jpeg or {}
    use_jpeg = bool(jpeg) and set(jpeg) == set(observation.frames)
    for role, frame in observation.frames.items():
        arrays[role] = np.frombuffer(jpeg[role], dtype=np.uint8) if use_jpeg else frame
    arrays["frame_encoding"] = np.asarray("jpeg" if use_jpeg else "rgb")
    np.savez(path, **arrays)
    return str(path)


def run_stage1_prefetch(
    env,
    predictor,
    *,
    execute_steps=None,
    prefetch_step=None,
    max_chunks=0,
    predictions=None,
    inference_frames_dir=None,
    rtc_guidance=True,
):
    """Execute valid actions while one fresh input/model request is in flight.

    ``prefetch_step`` counts matched terminal execution receipts. Source age is
    diagnostic only. Capture admission failures retry the same index; HIL or
    identity changes invalidate the chunk immediately. A subtask switch
    requests a new chunk with the new prompt at once; the active chunk keeps
    running until it arrives.
    """
    horizon = DEFAULT_CONTRACT.action_chunk_length
    if execute_steps is None:
        execute_steps = horizon
    if prefetch_step is None:
        prefetch_step = prefetch_schedule(
            horizon, DEFAULT_CONTRACT.control_hz
        ).request_after_steps
    if not 1 <= prefetch_step < execute_steps <= horizon or max_chunks < 0:
        raise ValueError(
            f"prefetch requires 1 <= prefetch_step < execute_steps <= {horizon}"
        )
    schedule = RTCChunkSchedule(prefetch_step, initial_delay_steps=INITIAL_DELAY_STEPS)
    env.reset()
    reader_factory = getattr(env, "create_stage1_reader", None)
    if reader_factory is None:
        raise RuntimeError("Stage1 async mode requires a read-only observation reader")
    reader = reader_factory()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage1-input")
    pending = None
    active = None
    waiting_for_fresh = False
    requests = attempts = completed_steps = episodes = epoch = accepted = discarded = 0
    inference_seconds = []
    request_events = []
    frame_dir = Path(inference_frames_dir) if inference_frames_dir else None
    last_finished_counts = [0, 0]

    def log(value):
        print(json.dumps(value, default=str), flush=True)

    def finish(reason, *, continue_window=False):
        nonlocal active
        if active is None:
            return
        request = active["request"]
        convergence = None
        if not continue_window and reason in {
            "new_prediction_ready",
            "window_completed",
            "source_observation_expired",
        }:
            end = getattr(env, "on_action_chunk_end", None)
            convergence = end() if end is not None else None
        last_finished_counts[0] = active["completed"]
        last_finished_counts[1] = active["attempted"]
        log(
            {
                "stage1_chunk": request.chunk,
                "window_end": convergence,
                **_source_identity(request),
                "inference_seconds": active.get("inference_seconds"),
                "read_seconds": active.get("read_seconds"),
                "processing_seconds": active.get("processing_seconds"),
                "start_index": active["start_index"],
                "end_index": active["index"],
                "window_continued": continue_window,
                "completed_steps": active["completed"],
                "attempted_steps": active["attempted"],
                "requested_execute_steps": execute_steps,
                "end_reason": reason,
                "receipts": active["statuses"],
                "receipt_reasons": active["reasons"],
                "step_timings": active["timings"],
                "source_age_s": source_age(env, request),
            }
        )
        active = None

    def invalidate_pending(reason):
        if pending is not None:
            pending[0].discard_reason = reason

    def start_request(trigger, completed_before_request, attempts_before_request):
        nonlocal pending, requests
        if max_chunks and requests >= max_chunks:
            return False
        requests += 1
        subtask, subtask_revision = subtask_state(env)
        request = InferenceInput(
            chunk=requests,
            epoch=epoch,
            revision=env._observation_revision,
            submitted_monotonic=time.monotonic(),
            trigger=trigger,
            completed_before_request=completed_before_request,
            attempts_before_request=attempts_before_request,
            episode=(env.hil.snapshot().get("episode") if env.hil else epoch),
            completed_global=completed_steps,
            prompt=env.task_description,
            subtask=subtask,
            subtask_revision=subtask_revision,
        )
        if active is not None:
            # RTC: the new chunk's index 0 aligns with this active index.
            request.executed_horizon = active["index"]
            request.delay_steps = schedule.predicted_delay_steps
            if rtc_guidance:
                request.prev_physical = active.get("physical")
                # The old subtask's plan must not pull the new one beyond
                # the steps that execute anyway while this request runs.
                request.rtc_hard_only = (
                    active["request"].subtask_revision != subtask_revision
                )
        pending = (
            request,
            executor.submit(
                read_and_predict, reader, predictor, execute_steps, request
            ),
        )
        event = {
            "stage1_prefetch_request": requests,
            "trigger": trigger,
            "completed_steps": completed_before_request,
            "attempted_steps": attempts_before_request,
            "epoch": epoch,
            "subtask": subtask,
        }
        request_events.append(event)
        log(event)
        return True

    def prediction_row(request, result, reason, frame_path):
        return {
            "chunk": request.chunk,
            **_source_identity(request),
            # Keep the long-standing field names in the JSONL stream while
            # retaining the structured identity fields above.
            "source_observation_id": (
                request.observation.observation_id if request.observation else None
            ),
            "source_reference_time_ns": (
                request.observation.timestamp_ns if request.observation else None
            ),
            "source_captured_monotonic_local": request.captured_monotonic,
            "source_age_s_at_ready": source_age(env, request),
            "read_seconds": result.get("read_seconds"),
            "processing_seconds": result.get("processing_seconds"),
            "inference_seconds": result.get("inference_seconds"),
            "absolute_targets": result["physical"].tolist(),
            "requested_execute_steps": execute_steps,
            "prefetch_step": prefetch_step,
            "prefetch_trigger": request.trigger,
            "requested_after_completed_steps": request.completed_before_request,
            "requested_after_attempts": request.attempts_before_request,
            "requested_after_step": request.completed_before_request,
            "rtc_executed_horizon": request.executed_horizon,
            "rtc_predicted_delay_steps": request.delay_steps,
            "rtc_guidance": result.get("rtc"),
            "prompt": request.prompt,
            "subtask": request.subtask,
            "disposition": "discarded" if reason else "accepted",
            "discard_reason": reason,
            "frame_roles": sorted((request.observation.frames or {}).keys()),
            "inference_input_path": frame_path,
            "model": "frozen_stage1",
        }

    def consume_pending():
        nonlocal pending, active, accepted, discarded, waiting_for_fresh
        if pending is None or not pending[1].done():
            return False
        request, future = pending
        pending = None
        result = future.result()
        reason = invalid_reason(env, request, epoch)
        frame_path = _save_inference_input(request, frame_dir)
        row = prediction_row(request, result, reason, frame_path)
        if predictions is not None:
            predictions.write(json.dumps(row, default=str) + "\n")
            predictions.flush()
        if reason:
            discarded += 1
            log(
                {
                    "stage1_prediction_discarded": request.chunk,
                    "reason": reason,
                    **_source_identity(request),
                }
            )
            if reason == "source_observation_expired":
                waiting_for_fresh = True
            return True
        # RTC delay compensation: steps executed while this request was in
        # flight already happened; start the new chunk at that time index.
        delay = max(0, completed_steps - request.completed_global)
        schedule.record_delay(delay)
        start_index = min(delay, len(result["actions"]) - 1)
        continue_window = bool(
            active is not None
            and getattr(env, "streams_action_chunks", lambda: False)()
        )
        if active is not None:
            finish("new_prediction_ready", continue_window=continue_window)
        configure = getattr(env, "set_action_window_steps", None)
        if configure is not None:
            configure(len(result["actions"]) - start_index)
            if continue_window:
                env.on_action_chunk_begin(continue_window=True)
            else:
                env.on_action_chunk_begin()
        bind_input(env, request)
        accepted += 1
        waiting_for_fresh = False
        active = {
            "request": request,
            "actions": result["actions"],
            "physical": result["physical"],
            "start_index": start_index,
            "index": start_index,
            "completed": 0,
            "attempted": 0,
            "statuses": [],
            "reasons": [],
            "timings": [],
            "read_seconds": result["read_seconds"],
            "processing_seconds": result["processing_seconds"],
            "inference_seconds": result["inference_seconds"],
        }
        inference_seconds.append(result["inference_seconds"])
        log(
            {
                "stage1_prediction_activated": request.chunk,
                "start_index": start_index,
                "window_continued": continue_window,
                **_source_identity(request),
            }
        )
        return True

    def after_step(terminated, truncated):
        nonlocal epoch, episodes, waiting_for_fresh
        if terminated or truncated:
            finish("episode_ended")
            invalidate_pending("episode_ended")
            waiting_for_fresh = False
            episodes += 1
            epoch += 1
            if not max_chunks or requests < max_chunks:
                env.reset()
            return True
        return False

    try:
        while True:
            panel = env.hil.snapshot() if env.hil else None
            blocked = env._manual_status(env._last_observation.status) or (
                panel and (panel["phase"] != "running" or panel.get("human_latched"))
            )
            if blocked:
                finish("hil_or_manual_changed")
                invalidate_pending("hil_or_manual_changed")
                configure = getattr(env, "set_action_window_steps", None)
                if configure is not None:
                    configure(execute_steps)
                env.on_action_chunk_begin()
                _, _, terminated, truncated, _ = env.step_hold()
                attempts += 1
                if (
                    after_step(terminated, truncated)
                    and max_chunks
                    and requests >= max_chunks
                ):
                    break
                continue

            # Consume a ready result before deciding whether another old action
            # may be submitted. Switching happens only at this boundary.
            consume_pending()

            if active is not None:
                reason = invalid_reason(env, active["request"], epoch)
                if reason is not None:
                    finish(reason)
                    invalidate_pending(reason)
                    if reason not in {
                        "human_active",
                        "hil_changed",
                        "control_context_changed",
                    }:
                        raise RuntimeError(f"Stage1 source invalidated: {reason}")

            can_request = not max_chunks or requests < max_chunks
            if pending is None and can_request:
                if active is None:
                    start_request(
                        "source_age" if waiting_for_fresh else "initial_or_replan",
                        last_finished_counts[0],
                        last_finished_counts[1],
                    )
                    waiting_for_fresh = False
                elif active["index"] >= len(active["actions"]):
                    finish("window_completed")
                elif schedule.request_due(active["completed"]):
                    start_request("step", active["completed"], active["attempted"])
                elif subtask_state(env)[1] != active["request"].subtask_revision:
                    # A new subtask prompt needs a new chunk now, not at the
                    # next prefetch step; the active chunk keeps executing.
                    start_request(
                        "subtask_switch", active["completed"], active["attempted"]
                    )

            # A fast worker can finish between the scheduling check and the
            # action boundary; consume it without submitting another old action.
            if consume_pending():
                continue

            if active is not None:
                reason = invalid_reason(env, active["request"], epoch)
                if reason is not None:
                    if reason in {
                        "human_active",
                        "hil_changed",
                        "control_context_changed",
                    }:
                        # A HIL/control change can occur after the first
                        # check but before the command boundary.  Drop
                        # this chunk and let the next loop obtain a fresh
                        # source under the new control context.
                        finish(reason)
                        invalidate_pending(reason)
                        continue
                    raise RuntimeError(f"Stage1 source invalidated: {reason}")
                # A ready prediction can replace the active chunk between the
                # scheduling check and this boundary.  Recheck the finite
                # action window before indexing so a completed 30-step window
                # is never indexed once more during that race.
                if active["index"] >= len(active["actions"]):
                    finish("window_completed")
                    continue
                index = active["index"]
                _, _, terminated, truncated, info = env.step(active["actions"][index])
                active["attempted"] += 1
                attempts += 1
                receipt = info.get("receipt") or {}
                active["statuses"].append(receipt.get("status"))
                active["reasons"].append(receipt.get("reason"))
                active["timings"].append(
                    {k: info.get(k) for k in ("step_seconds", "record_seconds")}
                )
                if _completed_receipt(info) and not env._skip_chunk:
                    active["index"] += 1
                    active["completed"] += 1
                    completed_steps += 1
                elif retryable_admission(receipt) and not env._skip_chunk:
                    # Keep this exact unexecuted index. Only successful new
                    # actions advance the shared RTC request horizon.
                    pass
                elif _control_interruption_receipt(info) or (
                    env._skip_chunk and _completed_receipt(info)
                ):
                    finish("execution_interrupted")
                    invalidate_pending("execution_interrupted")
                elif unreachable_target(receipt):
                    # Nothing moved (typically B J5 at its limit). Replan from
                    # the current observation; the env pauses the episode when
                    # targets keep being unreachable.
                    finish("target_unreachable")
                    invalidate_pending("target_unreachable")
                else:
                    finish("execution_rejected")
                    invalidate_pending("execution_rejected")
                    error = RuntimeError(
                        f"Stage1 action was not completed: {receipt.get('status')} {receipt.get('reason', '')}"
                    )
                    error.receipt = receipt
                    raise error
                ended = after_step(terminated, truncated)
                if ended and max_chunks and requests >= max_chunks:
                    break
            if pending is not None:
                # Source expiry waiting is intentionally a true hold: do not
                # submit a duplicate last target and do not call HOLD, because
                # HOLD would change the control generation.
                time.sleep(0.005)
            elif not can_request and active is None:
                break
    finally:
        finish("runner_stopped")
        invalidate_pending("runner_stopped")
        try:
            reader.close()
        finally:
            # The worker owns no SDK command state. A late CUDA result is
            # discarded; normal process shutdown may still wait for its kernel.
            executor.shutdown(wait=False, cancel_futures=True)

    return {
        "mode": "frozen_stage1",
        "chunks": requests,
        "steps": attempts,
        "attempted_steps": attempts,
        "completed_steps": completed_steps,
        "episodes": episodes,
        "action_shape": [horizon, 20],
        "control_hz": DEFAULT_CONTRACT.control_hz,
        "subtask_prompts": list(getattr(env, "subtask_prompts", ()) or ()),
        "inference_seconds": inference_seconds,
        "stage2_training": False,
        "prefetch_step": prefetch_step,
        "prefetch_age_s": None,
        "source_budget_s": None,
        "accepted_predictions": accepted,
        "discarded_predictions": discarded,
        "request_events": request_events,
    }
