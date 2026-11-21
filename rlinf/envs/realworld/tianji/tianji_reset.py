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

"""Run the loaded LoopInfra saved-pose Reset through its existing owner."""

import time
import urllib.error
from collections import deque

import numpy as np


def reset_saved_pose(
    backend, *, cancelled=lambda: False, timeout_s=120.0, trigger="start"
):
    """Wait for a matched Reset completion; never enable or invent a target.

    The caller must obtain episode Start or explicit End before calling. Renewals retain the
    server's 500ms deadman; faults, cancellation and loss of ownership fail the
    episode instead of allowing policy inference from an unfinished reset.
    """
    if backend.read_only:
        return
    if trigger not in {"start", "end"}:
        raise ValueError("Reset trigger must be start or end")
    status = backend._request("/api/status")
    validate_profile = getattr(backend, "validate_control_profile", None)
    if callable(validate_profile):
        validate_profile(status)
    control = status.get("robot_control") or {}
    pose = control.get("reset_pose") or {}
    if (
        pose.get("schema_version") != "loopinfra.reset_pose.v1"
        or pose.get("robot_model") != "M6S-CCS-696-V4.0"
        or pose.get("backend") != "hardware"
        or pose.get("units") != "rad"
        or not isinstance(pose.get("id"), str)
        or not pose["id"]
    ):
        raise RuntimeError(
            "Saved Tianji Reset pose is unavailable; episode not started"
        )
    joints = pose.get("position_rad") or {}
    if set(joints) != {"A", "B"} or any(
        np.asarray(joints[a]).shape != (7,)
        or not np.isfinite(np.asarray(joints[a], dtype=float)).all()
        for a in ("A", "B")
    ):
        raise RuntimeError("Invalid saved Reset joints; episode not started")
    identity = status.get("identity") or {}
    session, generation = identity.get("session_id"), identity.get("control_generation")
    if not session or type(generation) is not int:
        raise RuntimeError("Reset requires a connected LoopInfra session")
    if (
        not status.get("connected")
        or status.get("stale", True)
        or backend._manual_active(status)
        or status.get("pending")
        or status.get("reset_lease")
        or control.get("mode") != "eef_position"
        or control.get("transition")
        or control.get("last_error")
        or status.get("awaiting_generation") is not None
        or cancelled()
    ):
        raise RuntimeError(
            "Reset requires idle Position/EEF, released Pico and active "
            + ("Start" if trigger == "start" else "End")
        )

    lease = None
    complete = False
    observed_generation = generation
    visual_handoff_deadline = None
    trace = deque(maxlen=40)
    try:
        # No automatic retries: a lost POST response may already have queued
        # motion. Without renewals the existing server lease expires.
        # settle_assist: a contact-assist Reset that stalls just outside its
        # 0.5 deg tolerance (joint friction against the weak null-space
        # stiffness) gets a bounded push instead of a convergence-timeout
        # fault; LoopInfra's own reset buttons do not send it.
        queued = backend._request(
            "/api/robot",
            {
                "operation": "reset",
                "target": "saved",
                "reset_pose_id": pose["id"],
                "settle_assist": True,
                # Episode Start opens both grippers after the Reset (training
                # starts open; a fresh collector has no gripper command yet and
                # withholds the Stage1 state). The end-of-episode return does not.
                "open_grippers": trigger == "start",
            },
        )
        lease = queued.get("lease_id")
        sequence = queued.get("sequence")
        if not lease or type(sequence) is not int:
            raise RuntimeError("Reset did not return a lease and sequence")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            poll_started = time.monotonic()
            if cancelled():
                raise RuntimeError("Episode Reset cancelled by operator")
            status = backend._request("/api/status")
            current = status.get("identity") or {}
            control = status.get("robot_control") or {}
            active = status.get("reset_lease") or {}
            ack = (status.get("metrics") or {}).get("gauges", {}).get(
                "last_control_ack"
            ) or {}
            operation = control.get("last_operation") or {}
            trace.append(
                {
                    "monotonic_s": time.monotonic(),
                    "identity": current,
                    "connected": status.get("connected"),
                    "stale": status.get("stale"),
                    "observation_present": status.get("observation") is not None,
                    "observation_age_ms": status.get("observation_age_at_server_ms"),
                    "awaiting_generation": status.get("awaiting_generation"),
                    "reset_lease": active,
                    "expected_lease": lease,
                    "expected_sequence": sequence,
                    "ack": ack,
                    "control_generation": control.get("control_generation"),
                    "mode": control.get("mode"),
                    "last_error": control.get("last_error"),
                    "last_operation": operation,
                    "reset_progress": (control.get("reset") or {}).get("progress"),
                }
            )
            if current.get("session_id") != session or backend._manual_active(status):
                raise RuntimeError(
                    "Reset interrupted by session change or Pico takeover"
                )
            if not status.get("connected"):
                raise RuntimeError("Reset feedback disconnected or stale")
            if ack.get("sequence") == sequence and ack.get("status") == "rejected":
                raise RuntimeError("Reset rejected: " + str(ack.get("reason")))
            if control.get("last_error") or control.get("mode") in {
                "fault",
                "disable",
                "drag",
            }:
                raise RuntimeError(
                    "Reset interrupted: "
                    + str(control.get("last_error") or control.get("mode"))
                )
            if (control.get("reset_pose") or {}).get("id") != pose["id"]:
                raise RuntimeError("Saved Reset pose changed during motion")
            if status.get("stale", True):
                # LoopInfra clears RGB observations on a generation change.
                # At 10 Hz the next image may arrive after the reset ack.
                # Wait briefly ONLY for a matched reset generation handoff;
                # do not renew motion or accept completion without fresh data.
                # The owner's 500 ms deadman and hardware feedback gates remain
                # active throughout this wait.
                reset = control.get("reset") or {}
                matched_reset = (
                    active.get("id") == lease
                    and reset.get("lease_id") == lease
                    and operation.get("operation") == "reset"
                    and operation.get("status") == "admitted"
                    and control.get("mode") == "joint_reset"
                )
                matched_completion = (
                    not active
                    and operation.get("operation") == "reset"
                    and operation.get("status") == "completed"
                    and control.get("reset") is None
                    and control.get("mode") == "eef_position"
                )
                new_generation = current.get("control_generation")
                # ACKs and teleop windows advance the broker identity before
                # the separate robot_control state message arrives. That old
                # state may still say EEF / set_mode completed. Correlate this
                # gap with our lease and the single reset admission generation.
                matched_state_pending = (
                    active.get("id") == lease
                    and active.get("requested_generation") == generation
                    and new_generation == generation + 1
                    and control.get("session_id") == session
                    and control.get("control_generation") == generation
                    and control.get("mode") == "eef_position"
                    and control.get("reset") is None
                    and not control.get("transition")
                )
                missing_new_image = (
                    status.get("observation") is None
                    and status.get("observation_age_at_server_ms") is None
                    and status.get("awaiting_generation") is None
                    and type(new_generation) is int
                    and new_generation > generation
                    and (matched_reset or matched_completion or matched_state_pending)
                )
                if missing_new_image:
                    if new_generation != observed_generation:
                        observed_generation = new_generation
                        visual_handoff_deadline = time.monotonic() + 0.2
                    if (
                        visual_handoff_deadline is not None
                        and time.monotonic() < visual_handoff_deadline
                    ):
                        time.sleep(0.02)
                        continue
                raise RuntimeError("Reset feedback disconnected or stale")
            observed_generation = current.get("control_generation", generation)
            visual_handoff_deadline = None
            if not active:
                # Old completion messages cannot acknowledge this request.
                complete = (
                    current.get("control_generation", generation) > generation
                    and operation.get("operation") == "reset"
                    and operation.get("status") == "completed"
                    and control.get("reset") is None
                    and control.get("mode") == "eef_position"
                    and not control.get("transition")
                    and status.get("awaiting_generation") is None
                )
                if complete:
                    return
                raise RuntimeError("Reset lease ended without matched completion")
            if active.get("id") != lease:
                raise RuntimeError("Reset lease replaced")
            # Recheck after the network read: operator cancellation during GET
            # must not be followed by one more motion renewal.
            if cancelled():
                raise RuntimeError("Episode Reset cancelled by operator")
            try:
                backend._request(
                    "/api/robot", {"operation": "reset_renew", "lease_id": lease}
                )
            except urllib.error.HTTPError as exc:
                # Completion can race the renewal. Verify fresh status on the
                # next iteration; a missing lease alone never means success.
                if exc.code != 409:
                    raise
            trace[-1]["poll_and_renew_s"] = time.monotonic() - poll_started
            # Account for both HTTP trips instead of adding 100 ms after them.
            # The owner's 500 ms deadman remains unchanged.
            time.sleep(max(0.0, 0.05 - (time.monotonic() - poll_started)))
        raise TimeoutError("Saved-pose Reset timed out; episode not started")
    except Exception as exc:
        # Keep the pre-hold evidence; cleanup changes the very state we need
        # to diagnose. Stage1 persists this bounded trace in error.json.
        exc.reset_trace = list(trace)
        raise
    finally:
        if lease and not complete:
            try:
                current = backend._request("/api/status")
                if (
                    (current.get("identity") or {}).get("session_id") == session
                    and (current.get("reset_lease") or {}).get("id") == lease
                    and not backend._manual_active(current)
                ):
                    backend._request("/api/hold", {})
            except (OSError, ValueError):
                # Do not renew: server/client deadman stops a disconnected reset.
                pass
