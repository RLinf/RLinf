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

"""Backends for dummy/playback and the existing LoopInfra policy API."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import struct
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

import numpy as np

from .tianji_contract import (
    ACTION_DIM,
    DEFAULT_CONTRACT,
    MIN_SCALED_DURATION_S,
    SEMANTICS,
    default_loopinfra_root,
    dense_rows_from_env,
    identity_state,
    project_pose,
    stage_time_scale_from_env,
)
from .tianji_tracking import controller_targets, executed_absolute, target_deltas
from . import tianji_joint
from .tianji_kinematics import JOINT_CONTRACT, JOINT_DIMS, tcp20_from_joint16_np

LOOPINFRA_SCHEMA_VERSION = "loopinfra.sdk.v1"
LOOPINFRA_MAX_HEADER = 256 * 1024
LOOPINFRA_POLICY_PROTOCOL = "loopinfra.policy_pose.v1"
REFERENCE_PROTOCOL = "loopinfra.policy_reference_window.v1"
# Owner continues each segment from the previous reference velocity and
# receipts it early, so one window spans consecutive chunks without stopping.
STREAM_PROTOCOL = "loopinfra.policy_reference_stream.v1"
# Segment length at 10 Hz; TianjiContract.stream_interval_s() scales it.
STREAM_INTERVAL_S = 0.12
# A single > 30 ms SDK feedback gap rejects an admission without executing it.
FEEDBACK_GAP_RETRIES = 5
FEEDBACK_GAP_RETRY_S = 0.02
# Reuse a measured capture younger than this for the next admission.
EXECUTION_REUSE_S = 0.02
# Receipt polling; a 30 Hz step leaves a few milliseconds for detection.
RECEIPT_POLL_S = 0.0015
# rt fork: the server admits the rows of a chunk on a fixed grid
# (LoopInfra server/policy_schedule.py). One long-poll per row.
SCHEDULE_VERSION = 1
SCHEDULE_ROW_WAIT_MS = 50
# Unexecuted rows after which the server keeps running the chunk (as RLinf does).
SCHEDULE_CONTINUES = frozenset(
    {
        "executed_hardware",
        "executed_fake",
        "execution_observation_expired",
        "source_observation_expired",
    }
)
# rtj joint Stage2: chunk rows that are only joint targets + closures
# (LoopInfra_rtj policy_action.JOINT_ROWS_PROTOCOL).
JOINT_ROWS_PROTOCOL = "loopinfra.policy_chunk_joint_rows.v1"


def joint_physical(joints14, closures):
    """[A J1..J7, A closure, B J1..J7, B closure] from 14 joints and two closures."""
    joints14 = np.asarray(joints14, dtype=np.float64).reshape(-1)
    if joints14.shape != (14,) or not np.isfinite(joints14).all():
        raise ValueError("joint label needs 14 finite joints")
    return np.concatenate([joints14[:7], [closures[0]], joints14[7:], [closures[1]]])


def executed_joint_label(receipt: Mapping[str, Any], tcp_physical) -> tuple[list | None, str | None]:
    """16D joint RL label of an executed row: the commanded joint step the owner
    sent (planning.target_joints), else the joints it measured at the receipt;
    closures are the TCP label's commanded closures."""
    closures = (float(tcp_physical[9]), float(tcp_physical[19]))
    for source, value in (
        ("commanded_joint_target", (receipt.get("planning") or {}).get("target_joints")),
        ("measured_joints_at_receipt", receipt.get("executed_joints")),
    ):
        if isinstance(value, Mapping) and set(value) == {"A", "B"}:
            try:
                physical = joint_physical([*value["A"], *value["B"]], closures)
            except (ValueError, TypeError):
                continue
            return JOINT_CONTRACT.encode_clipped(physical).tolist(), source
    return None, None


def retryable_admission(receipt: Mapping[str, Any]) -> bool:
    """A rejected execution capture is not a consumed model action."""
    return (
        receipt.get("hardware_executed") is False
        and not receipt.get("execution_may_be_partial")
        and (
            receipt.get("status") == "execution_observation_expired"
            or (
                receipt.get("status") == "source_observation_expired"
                and receipt.get("reason")
                in {
                    "source_observation_expired",
                    "policy_input_window_stale",
                    "observation_stale_or_unavailable",
                    "execution_observation_wait_timeout",
                }
            )
        )
    )


# LoopInfra found no safe joint increment toward the target (a joint limit,
# J6/J7 interference, or an IK failure on the way): nothing moved and the
# hardware is fine. Seen on B J5 at its -170 deg limit while the model turns
# the B gripper about its own axis close to the table.
UNREACHABLE_REASONS = frozenset(
    {
        "jog_no_safe_increment",
        "M6_V4_joint_limit",
        "M6_V4_J6_J7_interference",
        "native_IK_failed_singular_or_limit",
        # Joint fork: a joint target that would put an elbow inside the
        # shoulder plane, or a step LoopInfra cannot bound to its lookahead.
        "elbow_inside_shoulder",
        "policy_joint_target_lookahead",
    }
)


def unreachable_target(receipt: Mapping[str, Any]) -> bool:
    """A planner rejection of an unreachable target; nothing was executed."""
    return (
        receipt.get("status") == "rejected"
        and receipt.get("hardware_executed") is False
        and not receipt.get("execution_may_be_partial")
        and receipt.get("reason") in UNREACHABLE_REASONS
    )


@dataclass
class TianjiObservation:
    """One synchronized LoopInfra observation in model-facing form."""

    state: np.ndarray
    frames: dict[str, np.ndarray]
    observation_id: str
    timestamp_ns: int
    status: dict[str, Any]
    raw: dict[str, Any]
    images_jpeg: dict[str, bytes]
    # Monotonic time on the process that received this observation.  This is
    # deliberately local; LoopInfra's capture monotonic clock may belong to a
    # different host and must never be subtracted from this value.
    captured_monotonic: float | None = None
    # Independent Rectify snapshot; never mixed into camera RGB or force fields.
    tactile: dict[str, np.ndarray] | None = None
    # The same snapshot's selected roles at full resolution, for the episode
    # archive's xense/<role>.mp4 + .jsonl (the collector's own layout); never a
    # model input. {"error": ...} when the read failed.
    tactile_raw: dict | None = None


class TianjiBackend(Protocol):
    def observe(self) -> TianjiObservation: ...

    def observe_execution(self) -> TianjiObservation: ...

    def create_stage1_reader(self): ...

    def send_action(
        self, action: np.ndarray, observation: TianjiObservation
    ) -> dict: ...

    def reset(self) -> None: ...

    def hold(self) -> None: ...

    def close(self) -> None: ...


_READ_POOL = None


def parallel_read_pool():
    """Shared worker threads for observation reads; None when TIANJI_PARALLEL_READ=0."""
    global _READ_POOL
    if os.environ.get("TIANJI_PARALLEL_READ", "1") == "0":
        return None
    if _READ_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _READ_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tianji-read")
    return _READ_POOL


def _decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG without requiring OpenCV in the protocol smoke test."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except ImportError:
        import cv2

        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid JPEG image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise ValueError(f"invalid JPEG image: {exc}") from exc


def _state_from_observation(observation, status, previous):
    del status, previous
    payload = observation.get("tianji_stage1", {})
    if payload.get("semantics") != SEMANTICS:
        raise ValueError(
            "LoopInfra Stage1 v2 aligned FK is unavailable; restart updated server"
        )
    if (
        payload.get("reference_monotonic_ns")
        != observation["sync"]["reference_monotonic_ns"]
    ):
        raise ValueError("Stage1 state is not aligned to image capture")
    return DEFAULT_CONTRACT.validate_state(payload["state"])


def _unpack_loopinfra_packet(packet: bytes) -> tuple[dict, dict[str, bytes]]:
    """Decode LoopInfra's LIO1 packet without importing its package."""
    if len(packet) < 8 or packet[:4] != b"LIO1":
        raise ValueError("invalid LoopInfra observation envelope")
    size = struct.unpack("!I", packet[4:8])[0]
    if size <= 0 or size > LOOPINFRA_MAX_HEADER or size > len(packet) - 8:
        raise ValueError("invalid LoopInfra metadata length")

    def reject_constant(value):
        raise ValueError(f"non-finite LoopInfra metadata value: {value}")

    metadata = json.loads(
        packet[8 : 8 + size].decode("utf-8"), parse_constant=reject_constant
    )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != LOOPINFRA_SCHEMA_VERSION
    ):
        raise ValueError("LoopInfra schema mismatch")
    body = memoryview(packet)[8 + size :]
    images: dict[str, bytes] = {}
    offset = 0
    entries = metadata.get("images")
    if not isinstance(entries, list) or not entries:
        raise ValueError("LoopInfra packet has no images")
    for info in entries:
        if not isinstance(info, Mapping):
            raise ValueError("invalid LoopInfra image metadata")
        camera_id = info.get("camera_id")
        length = info.get("length")
        if (
            not isinstance(camera_id, str)
            or camera_id in images
            or not isinstance(length, int)
            or info.get("observation_id") != metadata.get("observation_id")
        ):
            raise ValueError("invalid LoopInfra image metadata")
        if info.get("offset") != offset or length <= 0 or offset + length > len(body):
            raise ValueError("misaligned LoopInfra image payload")
        data = bytes(body[offset : offset + length])
        if hashlib.sha256(data).hexdigest() != info.get("sha256"):
            raise ValueError("LoopInfra image checksum mismatch")
        images[camera_id] = data
        offset += length
    if offset != len(body):
        raise ValueError("trailing bytes in LoopInfra observation packet")
    required = metadata.get("required_camera_ids")
    if (
        not isinstance(required, list)
        or len(set(required)) != len(required)
        or any(not isinstance(camera_id, str) for camera_id in required)
        or set(required) != set(images)
    ):
        raise ValueError("LoopInfra observation is missing a required camera")
    return metadata, images


class DummyTianjiBackend:
    """Deterministic playback backend used before GPU and hardware access."""

    # rtj joint Stage2: a reachable pose of both arms (rad) for the dummy's joints.
    DUMMY_JOINTS = (0.0, -0.5, 0.0, -1.2, 0.0, 0.3, 0.0)

    def __init__(
        self, *, image_hw=(64, 64), max_steps=100, seed=0, contract=DEFAULT_CONTRACT,
        joint_actions=False,
    ):
        self.contract = contract
        self.joint_actions = bool(joint_actions)
        self.action_contract = JOINT_CONTRACT if self.joint_actions else contract
        self.image_hw = tuple(int(x) for x in image_hw)
        if len(self.image_hw) != 2 or min(self.image_hw) < 2:
            raise ValueError("image_hw must be two dimensions >= 2")
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.joints = None
        self.reset()
        self._observation_counter = 0

    def _observation(self) -> TianjiObservation:
        self._observation_counter += 1
        h, w = self.image_hw
        base = np.uint8(self._observation_counter % 255)
        image = np.full((h, w, 3), base, dtype=np.uint8)
        frames = {role: image.copy() for role in self.contract.image_roles}
        raw = {}
        if self.joints is not None:
            raw["tianji_stage1"] = {
                "joints": [float(self.joints[d]) for d in JOINT_DIMS],
                "joint_velocities": [0.0] * 14,
                "state": self.state.tolist(),
                "commanded_joints": [float(self.joints[d]) for d in JOINT_DIMS],
            }
        return TianjiObservation(
            state=self.state.copy(),
            frames=frames,
            observation_id=f"dummy-{self._observation_counter}",
            timestamp_ns=time.time_ns(),
            status={"mode": "simulate", "manual": None, "robot_control": {}},
            raw=raw,
            images_jpeg={},
            captured_monotonic=time.monotonic(),
        )

    def observe(self) -> TianjiObservation:
        return self._observation()

    def observe_execution(self) -> TianjiObservation:
        """Dummy equivalent of the metadata-only execution observation."""
        return self._observation()

    def create_stage1_reader(self):
        return _DummyStage1ObservationReader(self)

    def send_action(self, action: np.ndarray, observation: TianjiObservation) -> dict:
        del observation
        normalized = self.action_contract.validate_action(action)
        decoded = self.action_contract.decode_action(normalized)
        self.last_action = normalized.copy()
        self.step_count += 1
        receipt = {
            "status": "executed_dummy",
            "hardware_executed": False,
            "rotation_policy_applied": True,
            "step": self.step_count,
        }
        if self.joint_actions:
            self.joints = decoded.astype(np.float64)
            self.state = tcp20_from_joint16_np(self.joints)
            receipt["executed_action_joint"] = JOINT_CONTRACT.encode_clipped(self.joints)
        else:
            self.state = decoded.copy()
        receipt["executed_action"] = self.contract.encode_action(self.state)
        return receipt

    def reset(self) -> None:
        self.step_count = 0
        self.last_action = np.zeros(self.action_contract.action_dim, dtype=np.float32)
        if self.joint_actions:
            self.joints = np.asarray([*self.DUMMY_JOINTS, 0.0] * 2, dtype=np.float64)
            self.state = tcp20_from_joint16_np(self.joints)
        else:
            self.state = identity_state()

    def hold(self) -> None:
        self.last_action.fill(0.0)

    def close(self) -> None:
        return None


class LoopInfraPolicyBackend:
    """HTTP policy adapter for a running LoopInfra server.

    It reads the exact synchronized LIO1 packet and submits the supported
    policy action (`eef_pose_delta`). LoopInfra's SDK controller keeps
    ownership of physical motion; when it reports an active manual lease this
    backend records the step as human controlled and does not submit a policy
    command.  This prevents a second writer from bypassing the SDK owner.
    """

    def __init__(
        self,
        url="http://127.0.0.1:8780",
        token=None,
        timeout=2.0,
        contract=DEFAULT_CONTRACT,
        *,
        read_only=True,
        control_profile=None,
        reference_windows=False,
        rt_schedule=False,
        joint_actions=False,
        tactile_enabled=False,
        tactile_raw=False,
    ):
        self.url = str(url).rstrip("/")
        self.tactile_enabled = bool(tactile_enabled)
        self.tactile_raw = bool(tactile_enabled and tactile_raw)
        self.token = token
        self.timeout = float(timeout)
        self.contract = contract
        # rtj joint Stage2: actions are 16D joint targets, sent to the owner as
        # joint targets only (never a TCP target, so never IK); labels in both spaces.
        self.joint_actions = bool(joint_actions)
        self.action_contract = JOINT_CONTRACT if self.joint_actions else contract
        self.read_only = bool(read_only)
        if control_profile not in (None, "loopinfra", "wuji_compat"):
            raise ValueError("Tianji policy requires loopinfra or wuji_compat Position")
        self.control_profile = control_profile
        self.reference_windows = bool(reference_windows)
        self._reference_window_id = None
        self._reference_active = False
        self._reference_context = None
        self._reference_final = False
        self.reference_cancelled = None
        # 11-04 experiment B: per-stage row time scale; the env sets current_subtask.
        self.stage_time_scale = stage_time_scale_from_env()
        self.current_subtask = None
        if self.stage_time_scale:
            print(f"TIANJI_STAGE_TIME_SCALE {self.stage_time_scale}", flush=True)
        self._verified_control_config = None
        self._closure = identity_state()
        self._last_state = identity_state()
        self._last_status: dict[str, Any] = {}
        self._owns_pending = False
        self._policy_action_id: str | None = None
        self.rt_schedule = bool(rt_schedule)
        self._schedule: dict[str, Any] | None = None
        # The next chunk queued behind the current one, and (after the chunk
        # turnover) that queued chunk awaiting the env's actual actions.
        self._next_schedule: dict[str, Any] | None = None
        self._candidate: dict[str, Any] | None = None
        # True while the owner's newest action may be one of our rows.
        self._schedule_owner = False
        self._scheduled_post: TianjiObservation | None = None
        self._scheduled_status_at = None
        # 11-05 dense rows (TIANJI_DENSE_ROWS=1): k model rows per env row, submitted as the chunk's
        # schedule at 1/(control_hz*k); the env's rows (every k-th) are what results are read for.
        self.dense_rows = dense_rows_from_env()
        self._dense_rows: list[tuple[np.ndarray, np.ndarray]] = []
        self._dense_missing_logged = False
        if self.dense_rows:
            print(f"TIANJI_DENSE_ROWS k={self.dense_rows}", flush=True)

    def streaming(self, status: Mapping[str, Any] | None = None) -> bool:
        """True when the owner continues reference windows across chunks."""
        control = (status if status is not None else self._last_status).get(
            "robot_control"
        ) or {}
        return bool(
            self.reference_windows
            and control.get("policy_stream_protocol") == STREAM_PROTOCOL
        )

    def validate_control_profile(self, status: Mapping[str, Any]) -> None:
        """Check the configured SDK owner before Reset or policy writes.

        The compact execution API omits backend policies. Read the collector's
        dual-endpoint summary once per session, generation and pose config;
        those identities are already checked again by the SDK on admission.
        """
        if not self.read_only and self.reference_windows:
            if (
                status.get("policy_reference_version") != 1
                or (status.get("robot_control") or {}).get("policy_reference_protocol")
                != REFERENCE_PROTOCOL
            ):
                raise RuntimeError(
                    "Restart both updated LoopInfra peers: bounded policy reference protocol unavailable; no motion sent"
                )
        if self.read_only or self.control_profile is None:
            return
        identity = status.get("identity") or {}
        control = status.get("robot_control") or {}
        key = (
            identity.get("session_id"),
            identity.get("control_generation"),
            control.get("pose_config_id"),
        )
        if not key[0] or type(key[1]) is not int or not key[2]:
            raise RuntimeError("LoopInfra control identity unavailable; no motion sent")
        if key == self._verified_control_config:
            return
        self._verified_control_config = None
        strategy = self._request("/api/collection").get("control_strategy") or {}
        policies = strategy.get("client_policies") or {}
        contact = policies.get("contact_assist")
        contact_supported = not contact or (
            self.control_profile == "wuji_compat"
            and contact == "sdk_cartesian_impedance_position_v7"
            and policies.get("policy_pose_contact") == "finite_cartesian_impedance_v1"
        )
        if not (
            strategy.get("configuration_matched") is True
            and strategy.get("backend_profile") == self.control_profile
            and strategy.get("client_backend_profile") == self.control_profile
            and strategy.get("sdk_control_mode") == "joint_position"
            and strategy.get("server_pose_config_id") == key[2]
            and strategy.get("client_pose_config_id") == key[2]
            and contact_supported
        ):
            raise RuntimeError(
                f"LoopInfra requires matching {self.control_profile}/joint_position "
                "on client and server and verified finite policy support for contact-assist; "
                "no motion sent. Restart the updated LoopInfra client/server with the RLT "
                "run_tianji_wuji_collector.sh entry for wuji_compat. "
                f"Reported server={strategy.get('backend_profile')}, "
                f"client={strategy.get('client_backend_profile')}, "
                f"mode={strategy.get('sdk_control_mode')}"
            )
        self._verified_control_config = key

    def _request(self, path: str, body: dict | None = None) -> Any:
        control_status = path == "/api/status" and body is None
        if control_status:
            query = {"control": "1"}
            if self._policy_action_id:
                query["action_id"] = self._policy_action_id
            path += "?" + urlencode(query)
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body, separators=(",", ":")).encode()
            if body is not None
            else None,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode())
        if control_status and result.get("control_status_version") != 1:
            raise RuntimeError(
                "Restart the updated LoopInfra collector: compact control status unavailable"
            )
        return result

    def _execution_observation(self, *, after: str | None = None) -> TianjiObservation:
        """Fetch real execution metadata without redundant RGB or status trips."""
        path = "/api/tianji/execution?current=1"
        if after is not None:
            path += "&" + urlencode({"after": after})
        result = self._request(path)
        if result.get("execution_metadata_version") != 2:
            raise RuntimeError(
                "Restart the updated LoopInfra server: fast execution metadata v2 unavailable"
            )
        metadata, status = result["observation"], result["status"]
        if status.get("control_status_version") != 1:
            raise RuntimeError("LoopInfra execution metadata protocol unavailable")
        self._last_status = status
        return self._execution_from_metadata(metadata, status)

    def _execution_from_metadata(self, metadata, status) -> TianjiObservation:
        return TianjiObservation(
            state=_state_from_observation(metadata, status, self._closure),
            frames={},
            images_jpeg={},
            observation_id=str(metadata["observation_id"]),
            timestamp_ns=int(metadata["reference_time_ns"]),
            status=status,
            raw=metadata,
            captured_monotonic=time.monotonic(),
        )

    def observe_execution(self) -> TianjiObservation:
        """Wait briefly for a valid capture; never replay an executed command."""
        deadline = time.monotonic() + self.timeout
        transient = {
            "waiting_for_control_generation",
            "observation_stale_or_unavailable",
            "execution_observation_wait_timeout",
        }
        while True:
            try:
                return self._execution_observation()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                try:
                    reason = json.loads(detail).get("error")
                except (ValueError, AttributeError):
                    reason = None
                if (
                    exc.code != 409
                    or not isinstance(reason, str)
                    or reason not in transient
                    or time.monotonic() >= deadline
                ):
                    raise RuntimeError(
                        f"LoopInfra execution observation unavailable ({exc.code}): {detail}"
                    ) from exc
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def create_stage1_reader(self):
        # The Stage1 worker gets an independent read-only client.  It has no
        # action id, pending command, or ownership state in common with this
        # execution backend.
        return LoopInfraStage1ObservationReader(
            self.url, self.token, self.timeout, self.contract,
            tactile_enabled=self.tactile_enabled, tactile_raw=self.tactile_raw,
        )

    def _packet(self) -> tuple[dict, dict[str, bytes]]:
        headers = {"Authorization": "Bearer " + self.token} if self.token else {}
        request = urllib.request.Request(
            self.url + "/api/packet?tianji_stage1=1", headers=headers
        )
        deadline = time.monotonic() + self.timeout
        gripper_note = 0.0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return _unpack_loopinfra_packet(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if exc.code == 409 and "stage1_invalid_commanded_closure" in detail:
                    # No gripper command since the collector started: LoopInfra
                    # withholds the Stage1 state. The episode-Start Reset opens
                    # both grippers (enable delay first); wait instead of failing.
                    now = time.monotonic()
                    if now >= gripper_note:
                        print(
                            "TIANJI_WAIT_GRIPPER: 等待夹爪的第一条命令"
                            "(stage1_invalid_commanded_closure);"
                            "若一直停在这里,在 Pico 上把左右扳机各按一下",
                            flush=True,
                        )
                        gripper_note = now + 10.0
                    time.sleep(0.05)
                    deadline = time.monotonic() + self.timeout
                    continue
                awaiting_frame = any(
                    reason in detail
                    for reason in (
                        "waiting_for_control_generation",
                        "observation_stale_or_unavailable",
                    )
                )
                if (
                    exc.code != 409
                    or not awaiting_frame
                    or time.monotonic() >= deadline
                ):
                    raise RuntimeError(
                        f"LoopInfra observation unavailable ({exc.code}): {detail}"
                    ) from exc
                # HOLD/Pico release invalidates the old observation; wait for
                # the next real capture instead of relabelling an old frame.
                time.sleep(0.01)

    @staticmethod
    def _manual_active(status: Mapping[str, Any]) -> bool:
        manual = status.get("manual")
        if isinstance(manual, Mapping):
            return True
        control = status.get("robot_control")
        if isinstance(control, Mapping):
            teleop = control.get("teleop")
            return bool(
                control.get("mode") == "drag"
                or (isinstance(teleop, Mapping) and teleop.get("active"))
            )
        return False

    def _read_tactile(self, metadata: Mapping) -> dict[str, np.ndarray] | None:
        if not self.tactile_enabled:
            return None
        from rlinf.utils.xense_tactile import MAX_TACTILE_PACKET, decode_tactile_packet

        descriptor = metadata.get("xense_tactile")
        if not isinstance(descriptor, Mapping) or descriptor.get("observation_id") != metadata["observation_id"]:
            raise RuntimeError("Enable Xense on the updated LoopInfra server before tactile inference")
        headers = {"Authorization": "Bearer " + self.token} if self.token else {}
        query = urlencode({"observation_id": metadata["observation_id"]})
        request = urllib.request.Request(self.url + "/api/tianji/tactile?" + query, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            packet = response.read(MAX_TACTILE_PACKET + 1)
        return decode_tactile_packet(packet, descriptor)

    def _read_tactile_raw(self, metadata: Mapping) -> dict | None:
        """Full-resolution twin of _read_tactile for the archive; a failure is recorded, not raised."""
        if not self.tactile_raw:
            return None
        from rlinf.utils.xense_tactile import MAX_TACTILE_RAW_PACKET, decode_tactile_raw_packet

        try:
            headers = {"Authorization": "Bearer " + self.token} if self.token else {}
            query = urlencode({"observation_id": metadata["observation_id"]})
            request = urllib.request.Request(self.url + "/api/tianji/tactile_raw?" + query, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                packet = response.read(MAX_TACTILE_RAW_PACKET + 1)
            return decode_tactile_raw_packet(packet, str(metadata["observation_id"]))
        except Exception as exc:  # noqa: BLE001 -- archive data never stops the policy
            return {"error": f"{type(exc).__name__}: {exc}"}

    def observe(self) -> TianjiObservation:
        metadata, images = self._packet()
        pool = parallel_read_pool()
        if pool is not None:
            # Same requests and decodes as below, overlapped (11-04 latency).
            status_future = pool.submit(self._request, "/api/status")
            tactile_future = pool.submit(self._read_tactile, metadata)
            tactile_raw_future = pool.submit(self._read_tactile_raw, metadata)
            decoded = {camera_id: pool.submit(_decode_image, data) for camera_id, data in images.items()}
            status = status_future.result()
        else:
            status = self._request("/api/status")
        state = _state_from_observation(metadata, status, self._closure)
        role_by_id = {}
        seen_roles = set()
        for info in metadata.get("images", []):
            calibration = info.get("calibration")
            role = (
                info.get("physical_role")
                or info.get("role")
                or (
                    calibration.get("physical_role")
                    if isinstance(calibration, Mapping)
                    else None
                )
                or info.get("camera_id")
            )
            if role in seen_roles:
                raise ValueError(f"LoopInfra camera role is duplicated: {role!r}")
            seen_roles.add(role)
            role_by_id[info["camera_id"]] = role
        if pool is not None:
            frames = {role_by_id[camera_id]: future.result() for camera_id, future in decoded.items()}
        else:
            frames = {
                role_by_id[camera_id]: _decode_image(data)
                for camera_id, data in images.items()
            }
        missing = set(self.contract.image_roles) - set(frames)
        if missing:
            raise ValueError(
                f"LoopInfra observation missing camera roles: {sorted(missing)}"
            )
        self._last_state = state
        self._closure = state.copy()
        self._last_status = copy.deepcopy(status)
        return TianjiObservation(
            state=state,
            frames={role: frames[role] for role in self.contract.image_roles},
            observation_id=str(metadata["observation_id"]),
            timestamp_ns=int(metadata.get("reference_time_ns", time.time_ns())),
            status=status,
            raw=metadata,
            images_jpeg={role_by_id[k]: v for k, v in images.items()},
            captured_monotonic=time.monotonic(),
            tactile=tactile_future.result() if pool is not None else self._read_tactile(metadata),
            tactile_raw=tactile_raw_future.result() if pool is not None else self._read_tactile_raw(metadata),
        )

    @staticmethod
    def _local_translation_in_base(
        observation: TianjiObservation, arm: str, delta_local: np.ndarray
    ) -> np.ndarray:
        """Convert unified-episode local TCP translation to LoopInfra base axes."""
        try:
            sample = observation.raw["sync"]["latest_arms"][arm]
            rotation = np.asarray(sample["eef"]["rotation_matrix"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "LoopInfra observation has no validated EEF rotation; refusing "
                "to send a local action as a base-frame command"
            ) from exc
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise RuntimeError("LoopInfra observation EEF rotation is invalid")
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-4
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise RuntimeError("LoopInfra EEF rotation is not a proper rotation matrix")
        delta = np.asarray(delta_local, dtype=np.float64)
        result = rotation @ delta
        if not np.isfinite(result).all():
            raise RuntimeError("local-to-base EEF action conversion is non-finite")
        return result.astype(np.float32)

    # The env passes its latest measured capture to send_action.
    accepts_latest_observation = True

    @staticmethod
    def _reusable_execution(latest):
        """A measured capture this fresh replaces the pre-admission reads.

        The previous step's post-action observation was read moments ago and
        carries the compact status of that read; at 30 Hz the two extra round
        trips would not fit in the control period. Admission still checks its
        identity and age on the owner.
        """
        if (
            latest is None
            or latest.raw.get("execution_observation_protocol")
            != "loopinfra.policy_execution_observation.v1"
            or latest.captured_monotonic is None
            or time.monotonic() - latest.captured_monotonic > EXECUTION_REUSE_S
            or (latest.status or {}).get("control_status_version") != 1
        ):
            return None
        return latest

    def send_action(
        self,
        action: np.ndarray,
        observation: TianjiObservation,
        latest: TianjiObservation | None = None,
    ) -> dict:
        normalized = self.action_contract.validate_action(action)
        reuse = self._reusable_execution(latest)
        status = reuse.status if reuse is not None else self._request("/api/status")
        refused = self._refuse_admission(status, observation)
        if refused is not None:
            return refused
        capabilities = observation.raw.get("capabilities", {})
        if capabilities.get("policy_pose_protocol") == LOOPINFRA_POLICY_PROTOCOL:
            # Configuration discovery may be slow. Finish it before acquiring
            # the short-lived execution capture, then verify its identity below.
            self.validate_control_profile(status)
            expired_id = None
            admission_attempts = []
            total_fetch_seconds = 0.0
            window_restarts = 0
            admission_failures = 0
            feedback_retries = 0
            stationary_started = None
            stationary_rejections = 0

            def stationary_timeout():
                receipt.update(
                    status="rejected",
                    reason="policy_stationary_start_timeout",
                    last_rejection_reason="policy_requires_stationary_start",
                    stationary_start_rejections=stationary_rejections,
                    stationary_wait_seconds=time.monotonic() - stationary_started,
                )
                return receipt

            while True:
                if (
                    self.reference_windows
                    and callable(self.reference_cancelled)
                    and self.reference_cancelled()
                ):
                    self.hold()
                    return {"status": "hil_hold", "hardware_executed": False}
                if stationary_started is not None:
                    if time.monotonic() - stationary_started >= 0.5:
                        return stationary_timeout()
                fetch_started = time.monotonic()
                try:
                    if reuse is not None and expired_id is None:
                        execution, reuse = reuse, None
                    else:
                        execution = (
                            self._execution_observation(after=expired_id)
                            if expired_id is not None
                            else self._execution_observation()
                        )
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode(errors="replace")
                    try:
                        reason = json.loads(detail).get("error")
                    except (ValueError, AttributeError):
                        reason = None
                    if exc.code == 409 and reason in {
                        "waiting_for_control_generation",
                        "observation_stale_or_unavailable",
                        "execution_observation_wait_timeout",
                    }:
                        return {
                            "status": (
                                "observation_invalidated"
                                if reason == "waiting_for_control_generation"
                                else "source_observation_expired"
                            ),
                            "reason": reason,
                            "hardware_executed": False,
                        }
                    raise RuntimeError(
                        f"LoopInfra execution observation unavailable ({exc.code}): {detail}"
                    ) from exc
                fetch_seconds = time.monotonic() - fetch_started
                total_fetch_seconds += fetch_seconds
                keys = ("session_id", "control_generation")
                old, new = observation.raw, execution.raw
                config_keys = ("stream_epoch", "pose_config_id", "teleop_config_id")
                age = (
                    new["sync"]["reference_monotonic_ns"]
                    - old["sync"]["reference_monotonic_ns"]
                ) / 1e9
                if (
                    any(old[k] != new[k] for k in keys)
                    or any(
                        old["robot_control"][k] != new["robot_control"][k]
                        for k in config_keys
                    )
                    or old["collection_grippers"]["config_id"]
                    != new["collection_grippers"]["config_id"]
                ):
                    return {
                        "status": "observation_invalidated",
                        "hardware_executed": False,
                    }
                if age < 0:
                    return {
                        "status": "observation_invalidated",
                        "reason": "capture_order_changed",
                        "hardware_executed": False,
                    }
                if observation.captured_monotonic is not None:
                    age = max(age, time.monotonic() - observation.captured_monotonic)
                if self._manual_active(execution.status):
                    return {"status": "human_active", "hardware_executed": False}
                execution_identity = execution.status.get("identity") or {}
                if (
                    not execution.status.get("connected")
                    or (
                        execution.status.get("stale", True)
                        and not (
                            new.get("execution_observation_protocol")
                            == "loopinfra.policy_execution_observation.v1"
                            and execution.status.get("execution_feedback_fresh") is True
                        )
                    )
                    or execution.status.get("awaiting_generation") is not None
                    or any(execution_identity.get(k) != new[k] for k in keys)
                ):
                    return {
                        "status": "observation_invalidated",
                        "hardware_executed": False,
                    }
                # A metadata request may itself wait for the next capture.
                # Do not submit once the independent stationary wait has expired.
                if (
                    stationary_started is not None
                    and time.monotonic() - stationary_started >= 0.5
                ):
                    return stationary_timeout()
                receipt = self._send_pose_action(normalized, execution)
                admission_attempts.append(
                    {
                        "observation_id": execution.observation_id,
                        "status": receipt.get("status"),
                        "reason": receipt.get("reason"),
                        "fetch_seconds": fetch_seconds,
                    }
                )
                receipt.update(
                    source_observation_id=observation.observation_id,
                    execution_observation_id=execution.observation_id,
                    execution_capture_kind=(
                        "measured_control_feedback"
                        if execution.raw.get("execution_observation_protocol")
                        == "loopinfra.policy_execution_observation.v1"
                        else "synchronized_rgb_capture"
                    ),
                    source_age_s=age,
                    execution_fetch_seconds=total_fetch_seconds,
                    execution_admission_attempts=admission_attempts,
                    execution_admission_retries=len(admission_attempts) - 1,
                    reference_window_restarts=window_restarts,
                    feedback_gap_retries=feedback_retries,
                    stationary_start_rejections=stationary_rejections,
                    stationary_wait_seconds=(
                        time.monotonic() - stationary_started
                        if stationary_started is not None
                        else 0.0
                    ),
                )
                stationary_rejection = (
                    self.reference_windows
                    and receipt.get("status") == "rejected"
                    and receipt.get("hardware_executed") is False
                    and receipt.get("reason") == "policy_requires_stationary_start"
                )
                if stationary_rejection:
                    # Only a confirmed unexecuted admission can be retried.
                    # The SDK still decides stationary from fresh live feedback;
                    # never weaken that gate or resend an executed interval.
                    if stationary_started is None:
                        stationary_started = time.monotonic()
                    stationary_rejections += 1
                    receipt["stationary_start_rejections"] = stationary_rejections
                    remaining = 0.5 - (time.monotonic() - stationary_started)
                    if remaining <= 0 or stationary_rejections >= 10:
                        return stationary_timeout()
                    expired_id = execution.observation_id
                    time.sleep(min(0.05, remaining))
                    continue
                window_expired = bool(
                    self.reference_windows
                    and receipt.get("status") == "rejected"
                    and receipt.get("hardware_executed") is False
                    and receipt.get("reason")
                    in ("policy_window_closed", "policy_window_expired_requires_end")
                )
                if window_expired and admission_failures < 2:
                    # The previous zero-speed endpoint may have auto-closed
                    # during an observation/recording delay. Drain it first,
                    # then admit the SAME unexecuted target in a new window.
                    # All identities are rechecked next loop; source age is logged.
                    drained = self.finish_policy_window()
                    if not drained or drained.get("status") != "window_converged":
                        return {
                            "status": "observation_invalidated",
                            "reason": "control_context_changed",
                            "hardware_executed": False,
                        }
                    window_restarts += 1
                    admission_failures += 1
                    expired_id = execution.observation_id
                    continue
                window_busy = bool(
                    self.reference_windows
                    and receipt.get("status") == "rejected"
                    and receipt.get("hardware_executed") is False
                    and receipt.get("reason") == "policy_window_identity"
                    and not self._reference_active
                )
                if (
                    window_busy
                    and admission_failures < 2
                    and self._wait_owner_window_closed()
                ):
                    # A stream window left by the previous episode (or a hold)
                    # coasts to rest and closes at its input deadline. Admit
                    # the SAME unexecuted target in the new window afterwards.
                    window_restarts += 1
                    admission_failures += 1
                    expired_id = execution.observation_id
                    continue
                expired_admission = (
                    receipt.get("status") == "rejected"
                    and receipt.get("hardware_executed") is False
                    and receipt.get("reason")
                    in {
                        "expired_observation_or_action",
                        "policy_expired_during_planning",
                        "policy_input_window_expired",
                        "insufficient_observation_ttl_for_trajectory",
                    }
                )
                if expired_admission and admission_failures < 2:
                    admission_failures += 1
                    expired_id = execution.observation_id
                    continue  # Fresh metadata; same absolute target and original model source.
                feedback_gap = (
                    receipt.get("status") == "rejected"
                    and receipt.get("hardware_executed") is False
                    and receipt.get("reason") == "stale_feedback"
                )
                if feedback_gap and feedback_retries < FEEDBACK_GAP_RETRIES:
                    # The SDK refused before writing anything. Wait for a newer
                    # live capture and admit the SAME target; the 30 ms gate stays.
                    feedback_retries += 1
                    expired_id = execution.observation_id
                    time.sleep(FEEDBACK_GAP_RETRY_S)
                    continue
                if expired_admission or window_expired:
                    receipt["status"] = "execution_observation_expired"
                return receipt
        raise RuntimeError(
            "LoopInfra policy_pose.v1 is required for absolute TCP tracking"
        )

    def _refuse_admission(self, status, observation: TianjiObservation) -> dict | None:
        """A receipt when no policy action may be admitted for ``observation``."""
        self._last_status = status
        if self._manual_active(status):
            return {
                "status": "human_active",
                "hardware_executed": False,
                # The HTTP status endpoint does not contain a separately
                # sampled Pico command.  Do not substitute state for a measured action label. Preserve the state for diagnostics.
                "intervene_action": None,
                "intervene_action_available": False,
                "intervene_state": self._last_state.copy(),
            }
        if self.read_only:
            return {"status": "read_only", "hardware_executed": False}
        identity = status.get("identity") or {}
        control = status.get("robot_control") or {}
        observation_control = observation.raw.get("robot_control") or {}
        if (
            any(
                key in identity and identity[key] != observation.raw.get(key)
                for key in ("session_id", "control_generation")
            )
            or (
                "stream_epoch" in control
                and control["stream_epoch"] != observation_control.get("stream_epoch")
            )
            or status.get("awaiting_generation") is not None
        ):
            # A release/reset occurred during inference. Never attach this
            # action to a new observation and silently replay the old chunk.
            return {"status": "observation_invalidated", "hardware_executed": False}
        return None

    def _send_pose_action(
        self,
        normalized: np.ndarray,
        observation: TianjiObservation,
        *,
        window_end=False,
    ) -> dict:
        """Track absolute targets through bounded local increments and matched receipts."""
        if (
            self.reference_windows
            and callable(self.reference_cancelled)
            and self.reference_cancelled()
        ):
            self.hold()
            return {
                "status": "hil_hold",
                "reason": "hil_changed",
                "hardware_executed": False,
            }
        if self.joint_actions:
            # rtj joint Stage2: the action is the absolute joint target itself;
            # zero TCP increment, the gripper delta toward its closure.
            decoded = self.action_contract.decode_action(normalized)
            grips = observation.raw["collection_grippers"]["arms"]
            deltas = {
                a: [0.0] * 6 + [float(decoded[g] - grips[a]["commanded_closure"])]
                for a, g in (("A", 7), ("B", 15))
            }
            scales = {"A": 1.0, "B": 1.0}
        else:
            decoded = self.contract.decode_action(normalized)
            deltas, scales = target_deltas(decoded, observation)
        control = observation.raw["robot_control"]
        command = {
            "source": "policy",
            "action_type": "eef_pose_delta",
            "policy_protocol": LOOPINFRA_POLICY_PROTOCOL,
            "observation_id": observation.observation_id,
            "session_id": observation.raw["session_id"],
            "control_generation": observation.raw["control_generation"],
            "stream_epoch": control["stream_epoch"],
            "pose_config_id": control["pose_config_id"],
            "teleop_config_id": control["teleop_config_id"],
            "gripper_config_id": control["grippers"]["config_id"],
            "frame": "tcp_local",
            "units": {
                "translation": "m",
                "rotation": "rad",
                "gripper": "closure_delta",
            },
            "deltas": deltas,
            "duration_s": self.contract.stream_interval_s()
            if self.streaming()
            else min(0.08, max(0.05, 1.0 / self.contract.control_hz)),
            "ttl_ms": 250,
        }
        if self.reference_windows:
            if self._reference_window_id is None:
                if window_end:
                    raise RuntimeError("No reference window to finish")
                self._reference_window_id = uuid.uuid4().hex
            command["reference_window"] = {
                "protocol": REFERENCE_PROTOCOL,
                "window_id": self._reference_window_id,
                "phase": "end"
                if window_end
                else "final"
                if self._reference_final
                else "segment",
            }
            if window_end:
                command["deltas"] = {a: [0.0] * 7 for a in ("A", "B")}
        if self.joint_actions and not window_end:
            capabilities = observation.raw.get("capabilities") or {}
            if capabilities.get("policy_joint_target_protocol") != tianji_joint.JOINT_TARGET_PROTOCOL:
                raise RuntimeError(
                    "LoopInfra joint targets unavailable; start the collector pair "
                    f"from LOOPINFRA_ROOT ({default_loopinfra_root()}). No motion sent."
                )
            command["joint_targets"] = {
                "A": [float(x) for x in decoded[0:7]],
                "B": [float(x) for x in decoded[8:15]],
            }
            command["joint_target_protocol"] = tianji_joint.JOINT_TARGET_PROTOCOL
        elif tianji_joint.joint_mode() and not window_end:
            # Joint fork: execute the model's own joint row for this step; the
            # TCP increment stays zero, the gripper delta keeps its meaning.
            try:
                joints = tianji_joint.CACHE.joints_for_encoded(normalized)
            except KeyError:
                # rtj fork: Stage2 rows reach the owner through the schedule; a
                # per-step action without a cached Stage1 joint row is tracked
                # as a TCP target like any other action.
                joints = None
            capabilities = observation.raw.get("capabilities") or {}
            if (joints is not None
                    and capabilities.get("policy_joint_target_protocol") != tianji_joint.JOINT_TARGET_PROTOCOL):
                raise RuntimeError(
                    "LoopInfra joint targets unavailable; start the collector pair "
                    f"from LOOPINFRA_ROOT ({default_loopinfra_root()}). No motion sent."
                )
            if joints is not None:
                command["deltas"] = {a: [0.0] * 6 + [deltas[a][6]] for a in ("A", "B")}
                command["joint_targets"] = {
                    "A": [float(x) for x in joints[0:7]],
                    "B": [float(x) for x in joints[8:15]],
                }
                command["joint_target_protocol"] = tianji_joint.JOINT_TARGET_PROTOCOL
        try:
            queued = self._request("/api/commands", command)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                reason = json.loads(detail).get("error")
            except (ValueError, AttributeError):
                reason = None
            # Pico can take over between GET status and POST commands. These
            # conflicts invalidate this chunk but are normal HIL transitions.
            transient = {
                "manual_control_active",
                "executor_busy",
                "unknown_observation_or_generation",
                "waiting_for_control_generation",
                "policy_input_window_unavailable",
                "policy_input_window_stale",
                "observation_stale_or_unavailable",
            }
            if exc.code == 409 and reason in transient:
                source_expired = reason in {
                    "policy_input_window_stale",
                    "observation_stale_or_unavailable",
                }
                return {
                    "status": "human_active"
                    if reason == "manual_control_active"
                    else "source_observation_expired"
                    if source_expired
                    else "observation_invalidated",
                    "reason": reason,
                    "hardware_executed": False,
                }
            raise RuntimeError(
                f"LoopInfra rejected policy action ({exc.code}): {detail}"
            ) from exc
        self._owns_pending = True
        self._policy_action_id = queued.get("action_id")
        receipt = self._wait_for_action(queued, observation.observation_id)
        return self._complete_receipt(
            receipt,
            queued.get("action_id"),
            observation,
            command.get("reference_window"),
            normalized=normalized,
            decoded=decoded,
            deltas=command["deltas"],
            scales=scales,
            window_end=window_end,
        )

    def _complete_receipt(
        self,
        receipt: dict,
        aid: str | None,
        observation: TianjiObservation,
        command_window: dict | None,
        *,
        normalized: np.ndarray,
        decoded: np.ndarray,
        deltas,
        scales,
        window_end: bool,
    ) -> dict:
        """Validate a matched terminal receipt and derive its executed label."""
        execution_label_available = receipt.get("exec_action_available") is True
        receipt.update(
            action_normalized=normalized.tolist(),
            requested_deltas=deltas,
            requested_absolute_target=decoded.tolist(),
            tracking_scales=scales,
            action_semantics=getattr(self.action_contract, "semantics", SEMANTICS),
            exec_action_available=False,
        )
        terminal = receipt.get("status")
        if self.reference_windows:
            if (
                terminal in ("hil_hold", "human_active")
                or self._reference_window_id is None
            ):
                self._reference_active = False
                self._reference_window_id = None
            else:
                self._reference_active = (
                    self._reference_active or terminal != "rejected"
                )
            self._reference_context = (
                observation.raw.get("session_id"),
                observation.raw.get("control_generation"),
            )
        completed = (
            terminal in ("executed_hardware", "executed_fake")
            and receipt.get("action_id") == aid
            and receipt.get("observation_id") == observation.observation_id
            and receipt.get("policy_protocol") == LOOPINFRA_POLICY_PROTOCOL
            and receipt.get("hardware_executed") is (terminal == "executed_hardware")
        )
        receipt["rotation_policy_applied"] = bool(
            completed and receipt.get("rotation_policy_applied")
        )
        receipt["gripper_policy_applied"] = bool(
            completed and receipt.get("gripper_policy_applied")
        )
        if self.reference_windows:
            expected_phase = command_window["phase"]
            completed = bool(
                completed
                and receipt.get("reference_window") == command_window
                and receipt.get("completion_kind")
                == (
                    "window_converged"
                    if expected_phase == "final"
                    else "reference_interval"
                )
                and receipt.get("endpoint_converged") is (expected_phase == "final")
            )
            end_checked = bool(
                window_end
                and terminal == "window_converged"
                and receipt.get("action_id") == aid
                and receipt.get("observation_id") == observation.observation_id
                and receipt.get("policy_protocol") == LOOPINFRA_POLICY_PROTOCOL
                and receipt.get("reference_window") == command_window
                and receipt.get("completion_kind") == "window_converged"
                and receipt.get("endpoint_converged") is True
                and receipt.get("observation_only") is True
                and receipt.get("hardware_executed") is False
            )
            if (completed and expected_phase == "final") or end_checked:
                self._reference_active = False
                self._reference_window_id = None
            if (
                terminal in ("executed_hardware", "executed_fake") and not completed
            ) or (terminal == "window_converged" and not end_checked):
                receipt.update(
                    status="invalid_terminal_receipt",
                    reason="policy_reference_receipt_contract",
                )
        if (
            completed
            and execution_label_available
            and receipt["rotation_policy_applied"]
            and receipt["gripper_policy_applied"]
        ):
            values = receipt.get("executed_deltas")
            if isinstance(values, Mapping) and set(values) == {"A", "B"}:
                try:
                    physical = executed_absolute(values, observation)
                    self.contract.validate_state(physical)
                    executed = self.contract.validate_action(
                        self.contract.encode_action(physical)
                    )
                except (ValueError, TypeError):
                    receipt["execution_label_error"] = (
                        "invalid_or_out_of_range_executed_absolute_pose"
                    )
                else:
                    receipt["executed_action"] = executed.tolist()
                    receipt["exec_action_available"] = True
                    if self.joint_actions:
                        joint, source = executed_joint_label(receipt, physical)
                        if joint is not None:
                            receipt["executed_action_joint"] = joint
                            receipt["executed_action_joint_source"] = source
        return receipt

    def _wait_for_action(self, queued: dict, observation_id: str) -> dict:
        """Match terminal history by ID; admission alone is never an action label."""
        aid = queued.get("action_id")
        if not isinstance(aid, str) or not aid:
            raise RuntimeError("LoopInfra did not return a valid action_id")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if (
                self.reference_windows
                and callable(self.reference_cancelled)
                and self.reference_cancelled()
            ):
                self.hold()
                return {
                    "status": "hil_hold",
                    "reason": "hil_changed",
                    "action_id": aid,
                    "hardware_executed": False,
                    "execution_may_be_partial": True,
                    "exec_action_available": False,
                }
            status = self._request("/api/status")
            self._last_status = status
            for entry in reversed(status.get("history", [])):
                action = entry.get("action", {})
                if (
                    action.get("action_id") != aid
                    or action.get("observation_id") != observation_id
                ):
                    continue
                self._owns_pending = False
                result = dict(entry.get("receipt") or {})
                if entry["status"] in ("executed_hardware", "executed_fake") and (
                    result.get("action_id") != aid
                    or result.get("status") != entry["status"]
                ):
                    return {
                        "status": "invalid_terminal_receipt",
                        "action_id": aid,
                        "hardware_executed": False,
                    }
                result.update(status=entry["status"], action_id=aid)
                if entry["status"].startswith("preempted_by_"):
                    result.update(status="human_active", hardware_executed=False)
                return result
            if self._manual_active(status):
                self._owns_pending = False
                return {
                    "status": "human_active",
                    "action_id": aid,
                    "hardware_executed": False,
                }
            time.sleep(RECEIPT_POLL_S)
        # Only stop our still-pending action, never a later human takeover.
        pending = self._last_status.get("pending") or {}
        if pending.get("action", {}).get("action_id") == aid:
            self.hold()
        self._owns_pending = False
        return {
            "status": "receipt_timeout",
            "action_id": aid,
            "hardware_executed": False,
        }

    # ---------- rt fork: the server runs a chunk's rows on a fixed grid ----------
    def schedules_rows(self) -> bool:
        """True when the LoopInfra server admits each row of a chunk at its time."""
        return bool(self.rt_schedule and not self.read_only and self.streaming())

    def forget_schedule(self) -> None:
        """A new chunk starts; the chunk queued behind the previous one becomes the candidate."""
        self._schedule = None
        self._candidate, self._next_schedule = self._next_schedule, None

    def append_scheduled_chunk(self, actions, observation: TianjiObservation) -> bool:
        """Queue the next chunk behind the current one; the owner starts it on the grid."""
        if self.read_only or self._schedule is None or self._next_schedule is not None:
            return False
        try:
            schedule = self._submit_schedule(
                actions, observation, 0, after=self._schedule["schedule_id"]
            )
        except (RuntimeError, urllib.error.URLError) as exc:
            print(f"TIANJI_APPEND_FAILED: {exc}", flush=True)
            return False
        self._next_schedule = schedule if schedule["forwarded"] else None
        return self._next_schedule is not None

    def take_scheduled_observation(self) -> TianjiObservation | None:
        """The capture at the next row's time, read with the last scheduled row."""
        observation, self._scheduled_post = self._scheduled_post, None
        return observation

    def cancel_schedule(self, *, hold: bool = False) -> None:
        """Drop the rows the server has not started; ``hold`` also stops our segment."""
        live = any(x is not None for x in (self._schedule, self._next_schedule, self._candidate))
        self._schedule = self._next_schedule = self._candidate = None
        if self.read_only or not self._schedule_owner or not (live or hold):
            return
        try:
            result = self._request(
                "/api/policy_schedule",
                {
                    "operation": "cancel",
                    "hold": hold,
                    "reason": "rlinf_hold" if hold else "rlinf_cancel",
                },
            )
        except (urllib.error.URLError, urllib.error.HTTPError):
            # Best effort like hold(): a stopped peer runs no rows.
            self._schedule_owner = False
            return
        if hold or result.get("held"):
            self._schedule_owner = False

    # ---------- 11-05 dense rows (TIANJI_DENSE_ROWS=1) ----------
    def offer_dense_rows(self, actions, dense) -> None:
        """Every model row of a chunk (k per env row) beside the env rows it is handed as; kept for
        that chunk's schedule (submitted at any row, or appended), matched by the env rows."""
        if not self.dense_rows:
            return
        k = self.dense_rows
        actions = np.asarray(actions, dtype=np.float32)
        dense = np.asarray(dense, dtype=np.float32)
        if k == int(k):
            holds = dense.shape == (k * len(actions), actions.shape[-1]) and np.array_equal(dense[::k], actions)
        else:
            # 11-06 1.5x: env row j is model row 1.5j = dense row 3j/2 (even j, exact) or the mean of
            # dense rows (3j-1)/2 and (3j+1)/2 (odd j); ceil(1.5*rows) dense rows.
            even, odd = np.arange(0, len(actions), 2), np.arange(1, len(actions), 2)
            holds = (
                dense.shape == (int(-(-k * len(actions) // 1)), actions.shape[-1])
                and np.array_equal(dense[3 * even // 2], actions[even])
                and np.allclose(
                    .5 * (dense[(3 * odd - 1) // 2] + dense[(3 * odd + 1) // 2]), actions[odd], rtol=1e-2, atol=1e-2
                )
            )
        if not holds:
            print(
                "TIANJI_DENSE_ROWS: offered dense rows do not hold the env rows at every k-th index; "
                "that chunk runs at the row period",
                flush=True,
            )
            return
        self._dense_rows = [(actions, dense), *self._dense_rows][:4]

    def _dense_schedule_rows(self, rows, first):
        """The validated dense rows from env row ``first`` of the offered chunk whose env rows end with
        ``rows``; None when dense rows are off or this chunk (a ramp or hold placeholder) has none."""
        if not self.dense_rows or not self.joint_actions or not rows:
            return None
        rest = np.stack(rows)
        for sparse, dense in self._dense_rows:
            if len(sparse) == first + len(rest) and np.array_equal(sparse[first:], rest):
                self._dense_missing_logged = False
                start = self.dense_rows * first
                if start != int(start):
                    # 11-06 1.5x: an odd env row starts between two dense rows.
                    print(f"TIANJI_DENSE_ROWS: chunk resubmitted from env row {first}; its rest runs at the row period", flush=True)
                    return None
                return [self.action_contract.validate_action(a) for a in dense[int(start) :]]
        if not self._dense_missing_logged:
            self._dense_missing_logged = True
            print("TIANJI_DENSE_ROWS: no dense rows for this chunk; its env rows run at the row period", flush=True)
        return None

    def _schedule_env_row(self, schedule, index):
        """One env row's result. In a dense schedule its k sub-rows are read in order: the first one
        that ends the schedule, else the last one (its capture is the next env row's)."""
        k = schedule.get("dense") or 1
        if k == int(k):
            k = int(k)
            subs = range(k * index, k * index + k)
        else:
            # 11-06 1.5x: env row r = first + index spans the chunk's dense rows [ceil(k*r), ceil(k*(r+1)));
            # the schedule starts at dense row k*first, an integer (_dense_schedule_rows).
            r, base = schedule["first"] + index, int(k * schedule["first"])
            subs = range(int(-(-k * r // 1)) - base, int(-(-k * (r + 1) // 1)) - base)
        result = None
        for sub in subs:
            result = self._schedule_row(schedule, sub)
            if result is None or result.get("status") not in SCHEDULE_CONTINUES:
                break
        return result

    def send_scheduled_action(
        self,
        action: np.ndarray,
        row: int,
        chunk_actions: np.ndarray,
        observation: TianjiObservation,
    ) -> dict:
        """Submit the rest of the chunk once, then read row ``row``'s receipt.

        The server admits each row at its own time against its newest measured
        capture, so the control rate no longer depends on this process's HTTP
        round trips. Labels are validated exactly as for a per-step admission.
        """
        normalized = self.action_contract.validate_action(action)
        self._scheduled_post = None
        if callable(self.reference_cancelled) and self.reference_cancelled():
            self.hold()
            return {"status": "hil_hold", "reason": "hil_changed", "hardware_executed": False}
        schedule = self._schedule
        if schedule is not None and not (
            0 <= row - schedule["first"] < len(schedule["actions"])
            and np.array_equal(schedule["actions"][row - schedule["first"]], normalized)
        ):
            # Not this chunk's schedule: submit anew (the server drops unstarted rows).
            schedule = None
        if schedule is None and self._candidate is not None:
            candidate, self._candidate = self._candidate, None
            rest = np.asarray(chunk_actions[row:], dtype=np.float32)
            if row == candidate["first"] and np.array_equal(np.stack(candidate["actions"]), rest):
                # The queued chunk is what the env executes: it already runs on the grid.
                schedule = self._schedule = candidate
            else:
                self._cancel_chunk(candidate["schedule_id"])
        if schedule is None:
            # The previous row's capture status is at most a period old; the
            # server checks identity and manual control again at every row.
            fresh = (
                self._scheduled_status_at is not None
                and time.monotonic() - self._scheduled_status_at < EXECUTION_REUSE_S * 5
            )
            status = self._last_status if fresh else self._request("/api/status")
            refused = self._refuse_admission(status, observation)
            if refused is not None:
                return refused
            self.validate_control_profile(status)
            schedule = self._schedule = self._submit_schedule(chunk_actions[row:], observation, row)
        result = self._schedule_env_row(schedule, row - schedule["first"])
        if result is None:
            return {"status": "hil_hold", "reason": "hil_changed", "hardware_executed": False}
        status = result["status"]
        if status == "admission_error" and result.get("reason") == "policy_chunk_predecessor_ended":
            # rtj fork (11-05): the queued next chunk reached the owner after its predecessor
            # ended early (a rejected row). None of it ran: handle it as a rejected row, so the
            # next row submits the rest of this chunk as a new schedule instead of ending Stage2.
            status = "rejected"
            result = {
                "status": status,
                "receipt": {"reason": "policy_chunk_predecessor_ended", "hardware_executed": False},
            }
        if status in ("admission_error", "execution_unavailable", "schedule_error"):
            self._schedule = None
            raise RuntimeError(
                f"LoopInfra policy schedule row failed ({status}): {result.get('reason')}"
            )
        if status not in SCHEDULE_CONTINUES:
            self._schedule = None  # The server dropped the rest of this chunk.
        post = result.get("post") or {}
        if post.get("observation") is not None:
            self._last_status = post["status"]
            self._scheduled_status_at = time.monotonic()
            self._scheduled_post = self._execution_from_metadata(
                post["observation"], post["status"]
            )
        action_sent = result.get("action") or {}
        if status in ("executed_hardware", "executed_fake") and action_sent.get(
            "reference_window"
        ):
            # The server may have reopened the window; only a window the owner
            # executed in can be drained at the chunk end (2026-10-27: a never
            # opened id made finish_policy_window time out).
            self._reference_window_id = action_sent["reference_window"]["window_id"]
        metadata = result.get("execution_observation")
        receipt = dict(result.get("receipt") or {})
        receipt.update(status=status, action_id=action_sent.get("action_id"))
        receipt["time_scale"] = schedule.get("time_scale", 1.0)  # experiment B
        if schedule.get("dense"):
            receipt["dense_rows"] = schedule["dense"]  # 11-05 dense rows: k model rows per env row
        if metadata is None or action_sent.get("reference_window") is None:
            # Nothing was admitted for this row.
            receipt["hardware_executed"] = bool(receipt.get("hardware_executed"))
            if status in ("hil_hold", "human_active"):
                self._reference_active = False
                self._reference_window_id = None
        else:
            execution = self._execution_from_metadata(metadata, {})
            receipt = self._complete_receipt(
                receipt,
                action_sent.get("action_id"),
                execution,
                action_sent["reference_window"],
                normalized=normalized,
                decoded=self.action_contract.decode_action(normalized),
                deltas=result.get("requested_deltas"),
                scales=result.get("tracking_scales"),
                window_end=False,
            )
            receipt.update(
                execution_observation_id=execution.observation_id,
                execution_capture_kind="measured_control_feedback",
            )
            if (
                schedule.get("dense")
                and receipt.get("status") in ("executed_hardware", "executed_fake")
                and not receipt.get("exec_action_available")
                and receipt.get("gripper_status") == "not_sent"
                and "execution_label_error" not in receipt
            ):
                # 11-05 dense rows: the row ran; only its gripper command had not gone out before
                # the next row ended it (DM period 20 ms > row period). Not a hold of the chunk.
                receipt["dense_row_unlabelled"] = True
        timing = result.get("timing") or {}
        stationary = result.get("stationary_wait_seconds")
        receipt.update(
            source_observation_id=observation.observation_id,
            source_age_s=result.get("source_age_s"),
            execution_fetch_seconds=0.0,
            execution_admission_attempts=result.get("attempts") or [],
            execution_admission_retries=max(0, len(result.get("attempts") or []) - 1),
            reference_window_restarts=result.get("window_restarts", 0),
            feedback_gap_retries=result.get("feedback_gap_retries", 0),
            stationary_start_rejections=result.get("stationary_start_rejections", 0),
            stationary_wait_seconds=stationary if stationary is not None else 0.0,
            schedule_row_id=result.get("row_id"),
            schedule_row_index=result.get("index"),
            schedule_late_ms=timing.get("late_ms"),
            # Server monotonic clock (same host): the row's actual start.
            schedule_started_ns=timing.get("started_ns"),
            # ms after the row's start: capture read, queued to the owner, receipt.
            schedule_timing_ms={
                k: timing[k]
                for k in ("captured_ms", "queued_ms", "receipt_ms")
                if k in timing
            },
        )
        return receipt

    def _cancel_chunk(self, schedule_id: str) -> None:
        try:
            self._request("/api/policy_schedule", {"operation": "cancel", "schedule_id": schedule_id})
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass  # A stopped peer runs no rows; a new chunk supersedes it anyway.

    def _submit_schedule(self, actions, observation: TianjiObservation, first: int, *, after=None):
        rows = [
            self.action_contract.validate_action(a)
            for a in np.asarray(actions, dtype=np.float32).reshape(-1, self.action_contract.action_dim)
        ]
        # 11-05 dense rows: the executor gets every model row of this chunk (k per env row) at
        # 1/(control_hz*k); ``rows`` stay the env rows whose results are read (the k-th dense rows).
        # Builtins only here (see below): the dense helpers are looked up, never assumed.
        dense_rows = getattr(self, "_dense_schedule_rows", None)
        dense = dense_rows(rows, first) if dense_rows is not None else None
        if self.joint_actions:
            # rtj joint Stage2: every row is its own joint target + closures.
            physical = np.stack([self.action_contract.decode_action(a) for a in (rows if dense is None else dense)])
            row_body = {
                "row_space": "joint",
                "joint_targets": [
                    {"A": [float(x) for x in p[0:7]], "B": [float(x) for x in p[8:15]]} for p in physical
                ],
                "closures": [{"A": float(p[7]), "B": float(p[15])} for p in physical],
            }
        else:
            # decode_action for the whole chunk at once (42 single-row calls cost ~8 ms).
            physical = project_pose(
                np.stack(rows) * self.contract.action_scale + self.contract.action_bias
            )
            tools = observation.raw["tianji_stage1"]["controller_tool_transforms"]
            row_body = {"targets": controller_targets(physical, tools)}
        if self._reference_window_id is None:
            self._reference_window_id = uuid.uuid4().hex
        control = observation.raw["robot_control"]
        # Experiment B: the current stage's row time scale (1.0 when off). Builtins
        # only: the joint Stage1 deploy test runs this function in a bare namespace.
        scales = getattr(self, "stage_time_scale", None)
        current = getattr(self, "current_subtask", None)
        scale = scales.get(int(current()), 1.0) if scales and callable(current) else 1.0
        body = {
            "operation": "submit",
            "session_id": observation.raw["session_id"],
            "control_generation": observation.raw["control_generation"],
            "stream_epoch": control["stream_epoch"],
            "pose_config_id": control["pose_config_id"],
            "teleop_config_id": control["teleop_config_id"],
            "gripper_config_id": observation.raw["collection_grippers"]["config_id"],
            "window_id": self._reference_window_id,
            "period_s": 1.0 / self.contract.control_hz / scale / (getattr(self, "dense_rows", 0) if dense is not None else 1),
            # Experiment B scales only the row period: a shorter segment sends its
            # early receipt (duration - 45 ms) before the gripper command, which
            # marks the row unexecuted and holds the rest of the chunk.
            "duration_s": self.contract.stream_interval_s(),
            "ttl_ms": 250,
            **row_body,
            "source_observation_id": observation.observation_id,
            "source_reference_monotonic_ns": observation.raw["sync"][
                "reference_monotonic_ns"
            ],
        }
        if after is not None:
            body.update(append=True, after_schedule_id=after)
        if dense is not None:
            body["dense"] = True  # 11-05 dense rows: the owner ends a segment at the next due row
        try:
            result = self._request("/api/policy_schedule", body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"LoopInfra policy schedule unavailable ({exc.code}): {detail}; "
                f"start the collector from LOOPINFRA_ROOT ({default_loopinfra_root()})"
            ) from exc
        if result.get("schedule_version") != SCHEDULE_VERSION:
            raise RuntimeError("LoopInfra policy schedule protocol mismatch")
        self._schedule_owner = True
        return {
            "schedule_id": result["schedule_id"],
            "row_ids": result["row_ids"],
            "actions": rows,
            "first": first,
            "forwarded": result.get("forwarded", True),
            "time_scale": scale,
            "dense": getattr(self, "dense_rows", 0) if dense is not None else 0,
        }

    def _schedule_row(self, schedule, index):
        """Long-poll one row; the HIL panel is checked between polls."""
        query = urlencode(
            {
                "schedule_id": schedule["schedule_id"],
                "row_id": schedule["row_ids"][index],
                "wait_ms": SCHEDULE_ROW_WAIT_MS,
            }
        )
        deadline = time.monotonic() + max(self.timeout, 3.0)
        while True:
            if callable(self.reference_cancelled) and self.reference_cancelled():
                self.hold()
                return None
            result = self._request("/api/policy_schedule/row?" + query)
            if result.get("status") != "pending":
                return result
            if time.monotonic() >= deadline:
                self.hold()
                return {
                    "status": "receipt_timeout",
                    "hardware_executed": False,
                    "receipt": {"hardware_executed": False},
                }

    def set_reference_final(self, final: bool) -> None:
        self._reference_final = bool(final)

    def _wait_owner_window_closed(self, timeout_s: float = 1.0) -> bool:
        """Wait for the owner to close an open window before a new one starts.

        Ending an episode or a hold forgets the stream window locally without
        an end command; on the owner the last segment coasts to rest and the
        window closes once its 250 ms input deadline passes.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            status = self._request("/api/status")
            if self._manual_active(status):
                return False
            if not (status.get("robot_control") or {}).get("policy_window"):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def finish_policy_window(self) -> dict | None:
        """Passively read the owner close ledger; never submit an end command."""
        if not self.reference_windows or not self._reference_active:
            self._reference_window_id = None
            return None
        window_id = self._reference_window_id
        deadline = time.monotonic() + self.timeout

        def cancelled(reason):
            self._reference_active = False
            self._reference_window_id = None
            return {
                "status": "window_cancelled",
                "reason": reason,
                "hardware_executed": False,
                "exec_action_available": False,
            }

        while True:
            if callable(self.reference_cancelled) and self.reference_cancelled():
                self.hold()
                return cancelled("hil_changed")
            status = self._request("/api/status")
            control = status.get("robot_control") or {}
            identity = status.get("identity") or {}
            context = (identity.get("session_id"), identity.get("control_generation"))
            if self._manual_active(status):
                return cancelled("manual_control_active")
            if context != self._reference_context:
                return cancelled("control_context_changed")
            previous = control.get("last_policy_window") or {}
            if control.get("last_error") or control.get("mode") != "eef_position":
                error = RuntimeError(
                    f"Policy window stopped: {control.get('last_error') or control.get('mode')}"
                )
                error.receipt = {
                    "status": "aborted_reference_window",
                    "reason": control.get("last_error"),
                    "window": previous,
                    "hardware_executed": False,
                }
                raise error
            if previous.get("window_id") == window_id:
                if previous.get("status") == "converged":
                    self._reference_active = False
                    self._reference_window_id = None
                    return {
                        "status": "window_converged",
                        "endpoint_converged": True,
                        "hardware_executed": False,
                        "exec_action_available": False,
                        "window_convergence": previous.get("convergence"),
                    }
                if previous.get("status") == "cancelled":
                    return cancelled(previous.get("reason"))
            if time.monotonic() >= deadline:
                error = RuntimeError("Policy window close receipt unavailable")
                error.receipt = {
                    "status": "reference_window_receipt_timeout",
                    "window_id": window_id,
                    "window": previous,
                    "hardware_executed": False,
                }
                raise error
            time.sleep(0.01)

    def reset(self) -> None:
        self.hold()

    def hold(self) -> None:
        self._reference_window_id = None
        self._reference_active = False
        self._reference_final = False
        self.cancel_schedule(hold=True)
        if self.read_only or not (self._owns_pending or self._policy_action_id):
            return
        try:
            status = self._request("/api/status")
            if self._manual_active(status):
                return
            if self._policy_action_id:
                pending = status.get("pending") or {}
                history = status.get("history") or []
                owner = pending or (history[-1] if history else {})
                if owner.get("action", {}).get("action_id") != self._policy_action_id:
                    return
            self._request("/api/hold", {})
        except (urllib.error.URLError, urllib.error.HTTPError):
            # close() and reset() must remain best effort when the peer has
            # already stopped; stale commands cannot be resumed locally.
            pass
        finally:
            self._owns_pending = False
            self._policy_action_id = None

    def close(self) -> None:
        self.hold()


class _DummyStage1ObservationReader:
    """Read-only dummy reader used by the asynchronous Stage1 scheduler."""

    def __init__(self, backend: DummyTianjiBackend):
        self._backend = backend

    def observe(self) -> TianjiObservation:
        return self._backend.observe()

    def close(self) -> None:
        return None


class LoopInfraStage1ObservationReader:
    """Independent RGB/state reader for Stage1 inference worker threads.

    This wrapper owns a separate read-only HTTP client.  It intentionally
    exposes no command, hold, reset, or action-id methods, so a slow image
    request cannot mutate the execution backend's ownership state.
    """

    def __init__(self, url, token, timeout, contract=DEFAULT_CONTRACT, *, tactile_enabled=False,
                 tactile_raw=False):
        self._backend = LoopInfraPolicyBackend(
            url, token, timeout, contract, read_only=True, tactile_enabled=tactile_enabled,
            tactile_raw=tactile_raw,
        )

    def observe(self) -> TianjiObservation:
        return self._backend.observe()

    def close(self) -> None:
        return None
