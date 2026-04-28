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

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

VERSION = 1
MSG_MODE = 1
MSG_POSE = 2
MSG_JOINT = 3

HEADER_FMT = "<BBIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MIN_FRAME_SIZE = HEADER_SIZE + 1
MAX_FRAME_SIZE = 4096


class FrameTimeoutError(TimeoutError):
    """Raised when a frame read times out before the payload is complete."""


@dataclass
class X2RobotTakeoverTCPConfig:
    host: str = "0.0.0.0"
    port: int = 8766
    socket_timeout_s: float = 0.1
    accept_timeout_s: float = 0.1
    reconnect_sleep_s: float = 0.1
    running_mode_source: str = "ros_param"
    running_mode_param: str = "/running_mode"
    normal_mode_value: int = 1
    takeover_mode_value: int = 2
    takeover_delay_s: float = 4.0
    slave_hold_settle_s: float = 0.0
    max_pose_age_s: float = 0.25
    debug_log: bool = False

    @classmethod
    def from_dict(
        cls, cfg: Mapping[str, Any] | None
    ) -> "X2RobotTakeoverTCPConfig":
        if cfg is None:
            return cls()
        cfg_dict = dict(cfg)
        control_mode = str(cfg_dict.pop("control_mode", "pose")).lower()
        if control_mode != "pose":
            raise ValueError(
                "master_takeover.control_mode must be 'pose' for this "
                f"takeover raw-collection path, got {control_mode!r}."
            )
        cfg_dict.pop("max_joint_age_s", None)
        return cls(**cfg_dict)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        try:
            data = sock.recv(size - received)
        except socket.timeout as exc:
            raise FrameTimeoutError from exc
        except OSError as exc:
            raise ConnectionError from exc
        if not data:
            raise ConnectionError("Socket closed while reading frame")
        chunks.append(data)
        received += len(data)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> dict[str, dict[str, Any]]:
    length_bytes = _read_exact(sock, 4)
    (length,) = struct.unpack("<I", length_bytes)
    if length < MIN_FRAME_SIZE or length > MAX_FRAME_SIZE:
        raise ValueError(f"Invalid frame length: {length}")
    body = _read_exact(sock, length)
    if len(body) < HEADER_SIZE:
        raise ValueError("Frame shorter than header")

    header_bytes = body[:HEADER_SIZE]
    payload_bytes = body[HEADER_SIZE:]
    version, msg_type, seq, timestamp_us = struct.unpack(HEADER_FMT, header_bytes)
    header = {
        "version": version,
        "msg_type": msg_type,
        "seq": seq,
        "timestamp_us": timestamp_us,
    }
    if version != VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")

    if msg_type == MSG_MODE:
        if len(payload_bytes) < 1:
            raise ValueError("MODE frame missing payload")
        (running_mode,) = struct.unpack("<b", payload_bytes[:1])
        payload = {"running_mode": int(running_mode)}
    elif msg_type == MSG_POSE:
        if len(payload_bytes) < 14 * 4:
            raise ValueError("POSE frame missing payload")
        values = struct.unpack("<" + "f" * 14, payload_bytes[: 14 * 4])
        payload = {
            "pose_left": list(values[:7]),
            "pose_right": list(values[7:]),
        }
    elif msg_type == MSG_JOINT:
        if len(payload_bytes) < 1:
            raise ValueError("JOINT frame missing payload")
        joint_count = payload_bytes[0]
        expected = 1 + joint_count * 4 * 2
        if len(payload_bytes) < expected:
            raise ValueError("JOINT frame shorter than declared payload")
        offset = 1
        joint_left = struct.unpack(
            "<" + "f" * joint_count,
            payload_bytes[offset : offset + 4 * joint_count],
        )
        offset += 4 * joint_count
        joint_right = struct.unpack(
            "<" + "f" * joint_count,
            payload_bytes[offset : offset + 4 * joint_count],
        )
        payload = {
            "joint_count": joint_count,
            "joint_left": list(joint_left),
            "joint_right": list(joint_right),
        }
    else:
        raise ValueError(f"Unknown message type: {msg_type}")

    return {"header": header, "payload": payload}


def send_frame(sock: socket.socket, data: dict[str, dict[str, Any]]) -> None:
    header = data.get("header") or {}
    payload = data.get("payload") or {}

    version = int(header.get("version", VERSION))
    msg_type = header.get("msg_type")
    seq = int(header.get("seq", 0))
    timestamp_us = int(header.get("timestamp_us", int(time.time() * 1e6)))

    if msg_type is None:
        raise ValueError("header.msg_type is required")

    if msg_type == MSG_MODE:
        running_mode = int(payload.get("running_mode", 0))
        payload_bytes = struct.pack("<b", running_mode)
    elif msg_type == MSG_POSE:
        pose_left = list(payload.get("pose_left") or [])
        pose_right = list(payload.get("pose_right") or [])
        if len(pose_left) != 7 or len(pose_right) != 7:
            raise ValueError("pose_left and pose_right must have 7 floats")
        payload_bytes = struct.pack("<" + "f" * 14, *(pose_left + pose_right))
    elif msg_type == MSG_JOINT:
        joint_left = list(payload.get("joint_left") or [])
        joint_right = list(payload.get("joint_right") or [])
        joint_count = int(
            payload.get("joint_count", min(len(joint_left), len(joint_right)))
        )
        if joint_count <= 0:
            raise ValueError("joint_count must be > 0")
        payload_bytes = struct.pack("<B", joint_count)
        payload_bytes += struct.pack("<" + "f" * joint_count, *joint_left[:joint_count])
        payload_bytes += struct.pack(
            "<" + "f" * joint_count, *joint_right[:joint_count]
        )
    else:
        raise ValueError(f"Unknown message type: {msg_type}")

    header_bytes = struct.pack(HEADER_FMT, version, msg_type, seq, timestamp_us)
    body = header_bytes + payload_bytes
    if len(body) < MIN_FRAME_SIZE or len(body) > MAX_FRAME_SIZE:
        raise ValueError("Frame length out of bounds")
    sock.sendall(struct.pack("<I", len(body)) + body)


class X2RobotTakeoverTCPServer:
    """Env-side TCP bridge preserving x2robot master/slave takeover semantics."""

    def __init__(
        self,
        config: X2RobotTakeoverTCPConfig,
        running_mode_getter: Callable[[], int],
        joint_snapshot_getter: Callable[[], np.ndarray],
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self._running_mode_getter = running_mode_getter
        self._joint_snapshot_getter = joint_snapshot_getter
        self._logger = logger or logging.getLogger(__name__)

        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._socket_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._io_thread: threading.Thread | None = None

        self._seq = 0
        self._current_mode = self.config.normal_mode_value
        self._mode_dirty = True
        self._snapshot_dirty = False

        self._master_pose_left: np.ndarray | None = None
        self._master_pose_right: np.ndarray | None = None
        self._master_pose_timestamp_us = 0
        self._master_pose_seq = 0
        self._master_pose_recv_time = 0.0
        self._follow_start_time: float | None = None
        self._min_pose_timestamp_us = 0
        self._last_pose_debug_log_time = 0.0

    def start(self) -> None:
        if self._io_thread is not None:
            return
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.config.host, self.config.port))
        self._server_socket.listen(1)
        self._server_socket.settimeout(self.config.accept_timeout_s)

        self._io_thread = threading.Thread(
            target=self._io_loop,
            name="X2RobotTakeoverTCPServer",
            daemon=True,
        )
        self._io_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._socket_lock:
            client_socket = self._client_socket
            server_socket = self._server_socket
            self._client_socket = None
            self._server_socket = None
        for sock in (client_socket, server_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if self._io_thread is not None:
            self._io_thread.join(timeout=1.0)
            self._io_thread = None

    def __del__(self):
        self.close()

    def poll(self) -> None:
        running_mode = int(self._running_mode_getter())
        with self._state_lock:
            if running_mode != self._current_mode:
                old_mode = self._current_mode
                self._current_mode = running_mode
                self._mode_dirty = True
                self._master_pose_left = None
                self._master_pose_right = None
                self._master_pose_timestamp_us = 0
                self._master_pose_seq = 0
                self._master_pose_recv_time = 0.0
                self._follow_start_time = None
                self._min_pose_timestamp_us = 0
                self._snapshot_dirty = running_mode == self.config.takeover_mode_value
                if self.config.debug_log:
                    self._logger.info(
                        "X2Robot takeover protocol mode change: %s -> %s, snapshot_dirty=%s",
                        old_mode,
                        running_mode,
                        self._snapshot_dirty,
                    )

    def sync_control_plane(self) -> None:
        self._sync_control_plane()

    def is_connected(self) -> bool:
        with self._socket_lock:
            return self._client_socket is not None

    def is_takeover_active(self) -> bool:
        with self._state_lock:
            return self._current_mode == self.config.takeover_mode_value

    def get_takeover_action(self) -> np.ndarray | None:
        with self._state_lock:
            now = time.time()
            if self._current_mode != self.config.takeover_mode_value:
                return None
            if self._follow_start_time is None or now < self._follow_start_time:
                if self.config.debug_log:
                    self._log_pose_gate_locked("delay", now)
                return None
            if (
                self._master_pose_left is None
                or self._master_pose_right is None
                or self._master_pose_recv_time <= self._follow_start_time
            ):
                if self.config.debug_log:
                    self._log_pose_gate_locked("missing_or_before_gate", now)
                return None
            pose_age_s = now - self._master_pose_recv_time
            if self.config.max_pose_age_s > 0 and pose_age_s > self.config.max_pose_age_s:
                if self.config.debug_log:
                    self._log_pose_gate_locked("stale", now, pose_age_s=pose_age_s)
                return None
            if self.config.debug_log:
                self._log_pose_gate_locked("fresh", now, pose_age_s=pose_age_s)
            return np.concatenate(
                [self._master_pose_left, self._master_pose_right], axis=0
            ).astype(np.float32)

    def _log_pose_gate_locked(
        self,
        reason: str,
        now: float,
        pose_age_s: float | None = None,
    ) -> None:
        if now - self._last_pose_debug_log_time < 1.0:
            return
        self._last_pose_debug_log_time = now
        follow_start = self._follow_start_time or 0.0
        if pose_age_s is None and self._master_pose_recv_time > 0:
            pose_age_s = now - self._master_pose_recv_time
        self._logger.info(
            "X2Robot takeover pose gate: reason=%s seq=%s pose_age_s=%s "
            "pose_ts_us=%s min_pose_ts_us=%s pose_recv=%.6f follow_start=%.6f now=%.6f",
            reason,
            self._master_pose_seq,
            "none" if pose_age_s is None else f"{pose_age_s:.4f}",
            self._master_pose_timestamp_us,
            self._min_pose_timestamp_us,
            self._master_pose_recv_time,
            follow_start,
            now,
        )

    def _io_loop(self) -> None:
        while not self._stop_event.is_set():
            client_socket = self._ensure_client_connection()
            if client_socket is None:
                time.sleep(self.config.reconnect_sleep_s)
                continue
            try:
                frame = recv_frame(client_socket)
            except FrameTimeoutError:
                continue
            except (ConnectionError, OSError, ValueError) as exc:
                self._logger.warning("Master takeover socket reset: %s", exc)
                self._drop_client_connection()
                continue
            header = frame["header"]
            payload = frame["payload"]
            if header["msg_type"] == MSG_POSE:
                pose_left = np.asarray(payload["pose_left"], dtype=np.float32)
                pose_right = np.asarray(payload["pose_right"], dtype=np.float32)
                with self._state_lock:
                    self._master_pose_left = pose_left
                    self._master_pose_right = pose_right
                    self._master_pose_timestamp_us = int(header["timestamp_us"])
                    self._master_pose_seq = int(header["seq"])
                    self._master_pose_recv_time = time.time()
                continue
    def _ensure_client_connection(self) -> socket.socket | None:
        with self._socket_lock:
            if self._client_socket is not None:
                return self._client_socket
            server_socket = self._server_socket

        if server_socket is None:
            return None
        try:
            client_socket, client_addr = server_socket.accept()
        except socket.timeout:
            return None
        except OSError:
            return None

        client_socket.settimeout(self.config.socket_timeout_s)
        with self._socket_lock:
            self._client_socket = client_socket
        with self._state_lock:
            self._mode_dirty = True
            if self._current_mode == self.config.takeover_mode_value:
                self._snapshot_dirty = True
                self._master_pose_left = None
                self._master_pose_right = None
                self._master_pose_timestamp_us = 0
                self._master_pose_seq = 0
                self._master_pose_recv_time = 0.0
                self._follow_start_time = None
                self._min_pose_timestamp_us = 0
        self._logger.info("Master takeover client connected from %s", client_addr)
        return client_socket

    def _drop_client_connection(self) -> None:
        with self._socket_lock:
            client_socket = self._client_socket
            self._client_socket = None
        if client_socket is not None:
            try:
                client_socket.close()
            except OSError:
                pass
        with self._state_lock:
            self._mode_dirty = True
            if self._current_mode == self.config.takeover_mode_value:
                self._snapshot_dirty = True
                self._master_pose_left = None
                self._master_pose_right = None
                self._master_pose_timestamp_us = 0
                self._master_pose_seq = 0
                self._master_pose_recv_time = 0.0
                self._follow_start_time = None
                self._min_pose_timestamp_us = 0

    def _next_header(self, msg_type: int) -> dict[str, int]:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return {
            "version": VERSION,
            "msg_type": msg_type,
            "seq": self._seq,
            "timestamp_us": int(time.time() * 1e6),
        }

    def _sync_control_plane(self) -> None:
        with self._socket_lock:
            client_socket = self._client_socket
        if client_socket is None:
            return

        with self._state_lock:
            current_mode = self._current_mode
            mode_dirty = self._mode_dirty
            snapshot_dirty = self._snapshot_dirty

        def send_mode() -> bool:
            try:
                if self.config.debug_log:
                    self._logger.info(
                        "X2Robot takeover protocol send MSG_MODE: mode=%s t=%.6f",
                        current_mode,
                        time.time(),
                    )
                send_frame(
                    client_socket,
                    {
                        "header": self._next_header(MSG_MODE),
                        "payload": {"running_mode": current_mode},
                    },
                )
            except (OSError, ValueError) as exc:
                self._logger.warning("Failed to send takeover mode: %s", exc)
                self._drop_client_connection()
                return False
            return True

        def send_joint_snapshot(phase: str) -> bool:
            joint_snapshot = np.asarray(self._joint_snapshot_getter(), dtype=np.float32)
            if joint_snapshot.shape != (2, 7):
                self._logger.warning(
                    "Skip takeover joint snapshot: expected shape (2, 7), got %s",
                    joint_snapshot.shape,
                )
                return False

            try:
                if self.config.debug_log:
                    self._logger.info(
                        "X2Robot takeover protocol send MSG_JOINT[%s]: t=%.6f left=%s right=%s",
                        phase,
                        time.time(),
                        np.array2string(joint_snapshot[0], precision=4),
                        np.array2string(joint_snapshot[1], precision=4),
                    )
                send_frame(
                    client_socket,
                    {
                        "header": self._next_header(MSG_JOINT),
                        "payload": {
                            "joint_left": joint_snapshot[0].tolist(),
                            "joint_right": joint_snapshot[1].tolist(),
                        },
                    },
                )
            except (OSError, ValueError) as exc:
                self._logger.warning("Failed to send takeover joint snapshot: %s", exc)
                self._drop_client_connection()
                return False
            return True

        if current_mode == self.config.takeover_mode_value and snapshot_dirty:
            # For the TCP master, MSG_MODE and MSG_JOINT are consumed by different
            # threads. Send the fresh joint snapshot before the mode transition so
            # the master cannot align to a stale cached snapshot. Send it once more
            # after mode for clients that clear joint caches on mode transitions.
            if not send_joint_snapshot("pre_mode"):
                return
            if mode_dirty:
                if not send_mode():
                    return
                with self._state_lock:
                    self._mode_dirty = False
                if not send_joint_snapshot("post_mode"):
                    return

            now = time.time()
            with self._state_lock:
                self._snapshot_dirty = False
                self._master_pose_left = None
                self._master_pose_right = None
                self._master_pose_timestamp_us = 0
                self._master_pose_seq = 0
                self._master_pose_recv_time = 0.0
                self._follow_start_time = now + self.config.takeover_delay_s
                self._min_pose_timestamp_us = int(self._follow_start_time * 1e6)
                if self.config.debug_log:
                    self._logger.info(
                        "X2Robot takeover protocol waiting for fresh master pose: now=%.6f follow_start=%.6f delay=%.3f min_pose_ts_us=%s max_pose_age_s=%.3f",
                        now,
                        self._follow_start_time,
                        self.config.takeover_delay_s,
                        self._min_pose_timestamp_us,
                        self.config.max_pose_age_s,
                    )
            return

        if mode_dirty:
            if not send_mode():
                return
            with self._state_lock:
                self._mode_dirty = False
            return


def make_ros_running_mode_getter(config: X2RobotTakeoverTCPConfig) -> Callable[[], int]:
    if config.running_mode_source != "ros_param":
        raise ValueError(
            f"Unsupported running_mode_source '{config.running_mode_source}'."
        )

    def _getter() -> int:
        try:
            import rospy
        except ImportError as exc:
            raise RuntimeError(
                "master_takeover.running_mode_source='ros_param' requires rospy "
                "to be importable in the RLinf env process."
            ) from exc

        value = rospy.get_param(config.running_mode_param, config.normal_mode_value)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"ROS param {config.running_mode_param!r} must be an integer "
                f"running mode, got {value!r}."
            ) from exc

    return _getter
