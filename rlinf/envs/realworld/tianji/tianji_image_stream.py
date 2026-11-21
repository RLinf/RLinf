#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Camera recorder of the Stage2 episode archive, run as its own process.

tianji_episode_archive starts it as ``python -I -B tianji_image_stream.py URL
PERIOD [MIN_PERIOD]`` (PERIOD: the collector's sync period; MIN_PERIOD: an
optional cap on the recorded rate). Standard library only: it shares no GPU,
interpreter lock or training state with the env worker, and the JPEG bytes
never pass through the training process. It reads each new three-camera
group from the collector with the collector page's own request
(``/api/packet?after=<id>``) and appends the original JPEG bytes to
``stream.bin`` and one JSON row per group to ``stream.jsonl`` in the
directory of the episode being recorded.

stdin: one JSON command per line: {"cmd": "start", "key", "dir", "after"},
{"cmd": "stop", "key", "after_capture_ns", "tail_s"} (stop once a group
captured at or after that time arrived, or after tail_s), {"cmd": "close"}.
stdout: one JSON line per stopped stream (counts, file hashes, errors).
EOF on stdin (the training process ended) closes the stream and exits.
"""

import hashlib
import json
import os
import select
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "loopinfra.sdk.v1"
MAX_HEADER = 256 * 1024
# Writing pauses below this much free disk and resumes above twice as much.
DISK_RESERVE_BYTES = 20 * 1024**3


def unpack(packet: bytes):
    """Metadata and {camera_id: jpeg} of an LIO1 packet, checked as the backend does."""
    if len(packet) < 8 or packet[:4] != b"LIO1":
        raise ValueError("invalid packet envelope")
    size = struct.unpack("!I", packet[4:8])[0]
    if not 0 < size <= min(MAX_HEADER, len(packet) - 8):
        raise ValueError("invalid packet header")
    metadata = json.loads(packet[8 : 8 + size])
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA:
        raise ValueError("packet schema mismatch")
    body = memoryview(packet)[8 + size :]
    images, offset = {}, 0
    for info in metadata.get("images") or []:
        length = info.get("length")
        if info.get("offset") != offset or type(length) is not int or not 0 < length <= len(body) - offset:
            raise ValueError("misaligned image payload")
        data = bytes(body[offset : offset + length])
        if hashlib.sha256(data).hexdigest() != info.get("sha256"):
            raise ValueError("image checksum mismatch")
        images[info.get("camera_id")] = (data, info)
        offset += length
    if offset != len(body) or not images:
        raise ValueError("incomplete packet")
    return metadata, images


def role_of(info: dict):
    calibration = info.get("calibration")
    return (
        info.get("physical_role")
        or info.get("role")
        or (calibration.get("physical_role") if isinstance(calibration, dict) else None)
        or info.get("camera_id")
    )


class Stream:
    """stream.bin + stream.jsonl of one episode, with running hashes."""

    def __init__(self, directory: str, key, after):
        self.directory, self.key, self.last_id = directory, key, after
        self.bin = open(os.path.join(directory, "stream.bin"), "xb")
        self.index = open(os.path.join(directory, "stream.jsonl"), "xb")
        self.bin_hash, self.index_hash = hashlib.sha256(), hashlib.sha256()
        self.bin_size = self.index_size = 0
        self.groups = self.requests = self.empty = self.skipped_for_disk = 0
        self.errors = {}
        self.last_capture = None
        self.paused = False
        self.disk_checked = 0.0

    def error(self, name: str):
        self.errors[name] = self.errors.get(name, 0) + 1

    def append(self, metadata: dict, images: dict):
        now = time.monotonic()
        if now - self.disk_checked > 10.0:
            self.disk_checked = now
            free = shutil.disk_usage(self.directory).free
            if free < DISK_RESERVE_BYTES:
                self.paused = True
            elif free > 2 * DISK_RESERVE_BYTES:
                self.paused = False
        capture = (metadata.get("sync") or {}).get("reference_monotonic_ns")
        if type(capture) is int:
            self.last_capture = capture
        self.last_id = str(metadata.get("observation_id"))
        if self.paused:
            self.skipped_for_disk += 1
            return
        entry = {
            "group": self.groups,
            "file": "stream.bin",
            "observation_id": self.last_id,
            "capture_ns": capture,
            "reference_time_ns": metadata.get("reference_time_ns"),
            "session_id": metadata.get("session_id"),
            "control_generation": metadata.get("control_generation"),
            "source": "stream",
            "images": {},
        }
        for data, info in images.values():
            entry["images"][role_of(info)] = {
                **{k: info[k] for k in ("camera_id", "width", "height", "sha256") if k in info},
                "offset": self.bin_size,
                "length": len(data),
            }
            self.bin.write(data)
            self.bin_hash.update(data)
            self.bin_size += len(data)
        line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        self.index.write(line)
        self.index_hash.update(line)
        self.index_size += len(line)
        self.groups += 1

    def close(self, reason: str) -> dict:
        for stream in (self.bin, self.index):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        return {
            "event": "stopped",
            "key": self.key,
            "stopped": reason,
            "groups": self.groups,
            "requests": self.requests,
            "no_new_group": self.empty,
            "skipped_for_disk": self.skipped_for_disk,
            "errors": self.errors,
            "last_capture_ns": self.last_capture,
            "files": {
                "images/stream.bin": {"sha256": self.bin_hash.hexdigest(), "bytes": self.bin_size},
                "images/stream.jsonl": {"sha256": self.index_hash.hexdigest(), "bytes": self.index_size},
            },
        }


def main(argv) -> int:
    url = argv[0].rstrip("/") + "/api/packet"
    floor = min(max(float(argv[2]), 0.0), 1.0) if len(argv) > 2 else 0.0
    period = max(min(max(float(argv[1]), 0.01), 0.1), floor)
    token = os.environ.get("TIANJI_IMAGE_TOKEN")
    headers = {"Authorization": "Bearer " + token} if token else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    fd = sys.stdin.fileno()
    pending, stream, tail, next_poll = b"", None, None, 0.0

    def report(message: dict):
        sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def finish(reason: str):
        nonlocal stream, tail
        if stream is not None:
            try:
                report(stream.close(reason))
            except Exception as exc:  # noqa: BLE001
                report({"event": "stopped", "key": stream.key, "stopped": reason,
                        "error": f"{type(exc).__name__}: {exc}"})
        stream, tail = None, None

    while True:
        timeout = None if stream is None else max(0.0, next_poll - time.monotonic())
        if select.select([fd], [], [], timeout)[0]:
            data = os.read(fd, 65536)
            if not data:
                finish("training_process_ended")
                return 0
            pending += data
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                command = json.loads(line)
                kind = command.get("cmd")
                if kind == "start":
                    finish("next_episode")
                    try:
                        stream = Stream(command["dir"], command.get("key"), command.get("after"))
                        next_poll = 0.0
                    except Exception as exc:  # noqa: BLE001
                        report({"event": "stopped", "key": command.get("key"), "stopped": "failed",
                                "error": f"{type(exc).__name__}: {exc}"})
                elif kind == "stop":
                    if stream is not None and stream.key == command.get("key") and tail is None:
                        tail = (command.get("after_capture_ns"),
                                time.monotonic() + float(command.get("tail_s", 0.3)))
                elif kind == "close":
                    finish("close")
                    return 0
            continue
        if stream is None:
            continue
        if tail is not None:
            target, deadline = tail
            if (target is None or (stream.last_capture or 0) >= target
                    or time.monotonic() > deadline):
                finish("episode_end")
                continue
        stream.requests += 1
        try:
            query = "?after=" + urllib.parse.quote(stream.last_id) if stream.last_id else ""
            with opener.open(urllib.request.Request(url + query, headers=headers), timeout=0.5) as response:
                body = response.read() if response.status == 200 else b""
            if not body:
                stream.empty += 1
                next_poll = time.monotonic() + 0.0025
                continue
            metadata, images = unpack(body)
        except urllib.error.HTTPError as exc:
            stream.error(f"http_{exc.code}")  # 409 while a Pico grip changes generation
            next_poll = time.monotonic() + 0.02
            continue
        except Exception as exc:  # noqa: BLE001
            stream.error(type(exc).__name__)
            next_poll = time.monotonic() + 0.05
            continue
        arrived = time.monotonic()
        if str(metadata.get("observation_id")) == stream.last_id:
            stream.empty += 1  # the display fallback repeats its last group
            next_poll = arrived + 0.0025
            continue
        try:
            stream.append(metadata, images)
        except Exception as exc:  # noqa: BLE001
            stream.error(type(exc).__name__)
            finish("write_failed")
            continue
        # A fixed period: the sync rate is configured, and the gaps between
        # the groups seen cannot reveal a faster one (missed groups look slow).
        next_poll = arrived + period - 0.0015


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
