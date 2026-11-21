# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage1-only receding-horizon execution using the existing Tianji environment."""

import json
import time

from ..tianji_contract import DEFAULT_CONTRACT
from .contract import validate_response


def run_stage1(
    env,
    predictor,
    *,
    execute_steps=3,
    prefetch_step=0,
    max_chunks=0,
    predictions=None,
    inference_frames_dir=None,
    rtc_guidance=True,
):
    """Run frozen actions, preserving HIL boundaries and inference source identity.

    The caller owns env.close(), including on exceptions. No learner or replay
    consumer is constructed. Episode results are logs only.
    """
    if prefetch_step:
        from .async_runner import run_stage1_prefetch

        return run_stage1_prefetch(
            env,
            predictor,
            execute_steps=execute_steps,
            prefetch_step=prefetch_step,
            max_chunks=max_chunks,
            predictions=predictions,
            inference_frames_dir=inference_frames_dir,
            rtc_guidance=rtc_guidance,
        )
    horizon = DEFAULT_CONTRACT.action_chunk_length
    if not 1 <= execute_steps <= horizon or max_chunks < 0:
        raise ValueError(f"execute_steps must be 1..{horizon}; max_chunks must be >=0")
    obs, _ = env.reset()
    chunks, steps, episodes = 0, 0, 0
    inference_seconds = []
    while not max_chunks or chunks < max_chunks:
        configure = getattr(env, "set_action_window_steps", None)
        if configure is not None:
            configure(execute_steps)
        env.on_action_chunk_begin()
        panel = env.hil.snapshot() if env.hil else None
        manual = env._manual_status(env._last_observation.status)
        if manual or (
            panel and (panel["phase"] != "running" or panel.get("human_latched"))
        ):
            # Do not infer during a manual lease or send a stale hold target
            # if the lease is released during this iteration.
            obs, _, terminated, truncated, info = env.step_hold()
            steps += 1
        else:
            source_id = env._last_observation.observation_id
            started = time.monotonic()
            physical = validate_response(
                predictor.predict_frames(
                    obs["state"]["qpos"], obs["frames"], env.task_description
                )
            )
            elapsed = time.monotonic() - started
            encoded = DEFAULT_CONTRACT.encode_action(physical)
            # Validate the complete execution slice before issuing any command.
            actions = [
                DEFAULT_CONTRACT.validate_action(a) for a in encoded[:execute_steps]
            ]
            chunks += 1
            inference_seconds.append(elapsed)
            if predictions is not None:
                predictions.write(
                    json.dumps(
                        {
                            "chunk": chunks,
                            "source_observation_id": source_id,
                            "source_reference_time_ns": env._last_observation.timestamp_ns,
                            "inference_seconds": elapsed,
                            "absolute_targets": physical.tolist(),
                            "requested_execute_steps": execute_steps,
                            "model": "frozen_stage1",
                        }
                    )
                    + "\n"
                )
                predictions.flush()
            statuses = []
            reasons = []
            timings = []
            for action in actions:
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                statuses.append(info["receipt"]["status"])
                reasons.append(info["receipt"].get("reason"))
                timings.append(
                    {k: info.get(k) for k in ("step_seconds", "record_seconds")}
                )
                if terminated or truncated or env._skip_chunk:
                    break
            end = getattr(env, "on_action_chunk_end", None)
            convergence = end() if end is not None else None
            print(
                json.dumps(
                    {
                        "stage1_chunk": chunks,
                        "window_end": convergence,
                        "inference_seconds": round(elapsed, 4),
                        "source_observation_id": source_id,
                        "receipts": statuses,
                        "receipt_reasons": reasons,
                        "step_timings": timings,
                    }
                ),
                flush=True,
            )
        if terminated or truncated:
            episodes += 1
            if max_chunks and chunks >= max_chunks:
                break
            obs, _ = env.reset()
    return {
        "mode": "frozen_stage1",
        "chunks": chunks,
        "steps": steps,
        "episodes": episodes,
        "action_shape": [horizon, 20],
        "inference_seconds": inference_seconds,
        "stage2_training": False,
    }
