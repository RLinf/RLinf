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

"""Observation-linked absolute TCP targets for human BC and execution labels."""

from __future__ import annotations

import math

import numpy as np

from .tianji_contract import DEFAULT_CONTRACT, SEMANTICS, rotation_from_6d


def _rotation_log(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-9:
        return skew / 2
    if math.pi - angle < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1) / 2, 0))
        index = int(np.argmax(axis))
        for other in range(3):
            if other != index:
                axis[other] = (rotation[index, other] + rotation[other, index]) / (
                    4 * axis[index]
                )
        return axis * angle
    return skew * (angle / (2 * math.sin(angle)))


LIVE_PROTOCOL = "loopinfra.policy_execution_observation.v1"


def _sample(observation):
    raw = observation.raw
    control = raw["robot_control"]
    live = raw.get("execution_observation_protocol") == LIVE_PROTOCOL
    if live:
        # A measured control capture carries no mode or lease; take them from
        # the compact status read together with it. Its fresh SDK feedback
        # and Position/EEF mode are already required by the capture itself.
        status = observation.status or {}
        control = {
            **(status.get("robot_control") or {}),
            **control,
            "manual": status.get("manual"),
        }
    sync = raw["sync"]
    if sync.get("sync_valid") is not True:
        raise ValueError("incompatible_feedback")
    identity = (raw["session_id"], raw["control_generation"], control["stream_epoch"])
    if not identity[0] or any(type(v) is not int for v in identity[1:]):
        raise ValueError("invalid_control_identity")
    if any(control.get(k) != raw[k] for k in ("session_id", "control_generation")):
        raise ValueError("unaligned_control_identity")
    if (
        control.get("mode") != "eef_position"
        or control.get("transition")
        or control.get("reset")
    ):
        raise ValueError("control_not_ready")
    stamp = sync["reference_monotonic_ns"]
    if type(stamp) is not int or stamp <= 0:
        raise ValueError("invalid_capture_time")
    grippers = raw["collection_grippers"]
    if grippers.get("available") is not True or grippers["sample_monotonic_ns"] > stamp:
        raise ValueError("unaligned_gripper_commands")
    config = (
        control["pose_config_id"],
        control["teleop_config_id"],
        grippers["config_id"],
    )
    if not all(isinstance(v, str) and v for v in config):
        raise ValueError("missing_config_identity")
    payload = raw["tianji_stage1"]
    if payload["semantics"] != SEMANTICS or payload["reference_monotonic_ns"] != stamp:
        raise ValueError("unaligned_stage1_state")
    state = DEFAULT_CONTRACT.validate_state(observation.state)
    poses = []
    feedback_stamps = []
    for arm, offset in (("A", 0), ("B", 10)):
        feedback = sync["latest_arms"][arm]
        if not live and (
            feedback["state"].get("err_code") != 0
            or feedback["state"].get("cur_state") != 1
        ):
            raise ValueError("incompatible_feedback")
        poses.append(
            (
                state[offset : offset + 3],
                rotation_from_6d(state[offset + 3 : offset + 9]),
                float(state[offset + 9]),
            )
        )
        feedback_stamps.append(stamp)
    return identity, config, stamp, control, poses, feedback_stamps


def observed_action(
    previous, following, *, contract=DEFAULT_CONTRACT, max_interval_s=0.25
):
    """Use the next measured absolute pose as the action label, with lease checks."""
    result = {
        "available": False,
        "source": "image_aligned_absolute_tcp_and_commanded_closure",
    }
    try:
        if previous.observation_id == following.observation_id:
            raise ValueError("duplicate_observation")
        identity0, config0, t0, control0, poses0, feedback0 = _sample(previous)
        identity1, config1, t1, control1, poses1, feedback1 = _sample(following)
        if identity0 != identity1 or config0 != config1:
            raise ValueError("control_context_changed")
        dt = (t1 - t0) / 1e9
        if not 0 < dt <= max_interval_s:
            raise ValueError("capture_interval_out_of_range")
        if any(
            not 0 < (end - start) / 1e9 <= max_interval_s
            for start, end in zip(feedback0, feedback1)
        ):
            raise ValueError("feedback_interval_out_of_range")
        manual0, manual1 = control0.get("manual"), control1.get("manual")
        human = isinstance(manual0, dict) and isinstance(manual1, dict)
        if human:
            lease = manual0.get("lease_id")
            if (
                not isinstance(lease, str)
                or not lease
                or lease != manual1.get("lease_id")
                or manual0.get("source") != "pico"
                or manual1.get("source") != "pico"
            ):
                raise ValueError("pico_lease_changed_or_unavailable")
            result["lease_id"] = lease
        elif manual0 is not None or manual1 is not None:
            raise ValueError("takeover_boundary")
        physical = contract.validate_state(following.state)
        normalized = contract.encode_action(physical)
        if not np.isfinite(normalized).all():
            raise ValueError("nonfinite_observed_delta")
        in_range = bool((np.abs(normalized) <= 1.000001).all())
        result.update(
            available=True,
            human=human,
            interval_s=dt,
            feedback_interval_s=[(b - a) / 1e9 for a, b in zip(feedback0, feedback1)],
            feedback_capture_skew_ms=[
                [(v - t) / 1e6 for v in stamps]
                for t, stamps in ((t0, feedback0), (t1, feedback1))
            ],
            observation_id=previous.observation_id,
            next_observation_id=following.observation_id,
            physical_action=physical.tolist(),
            normalized_action=normalized.tolist(),
            bounded_action=np.clip(normalized, -1, 1).tolist(),
            clipped=not in_range,
            exec_action_available=in_range,
        )
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        result["reason"] = str(exc)
    return result
