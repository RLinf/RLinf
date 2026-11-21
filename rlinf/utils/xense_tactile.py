# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Xense Rectify wire v1 and replay tensor contract; no vendor SDK dependency."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping

import numpy as np

XENSE_SCHEMA = "loopinfra.xense_rectify.v1"
# LoopInfra's XTO1 snapshot always carries these four roles in this order.
WIRE_ROLES = ("left_gripper_1", "left_gripper_2", "right_gripper_1", "right_gripper_2")


def _model_roles() -> tuple[str, ...]:
    """The roles the actor/critic read, in this order (11-03: the two installed
    sensors). TIANJI_TACTILE_ROLES=role,role overrides it for every process."""
    import os

    names = tuple(r.strip() for r in os.environ.get(
        "TIANJI_TACTILE_ROLES", "left_gripper_1,right_gripper_2").split(",") if r.strip())
    if not names or len(set(names)) != len(names) or any(r not in WIRE_ROLES for r in names):
        raise ValueError(f"TIANJI_TACTILE_ROLES must name distinct roles of {WIRE_ROLES}: {names}")
    return names


TACTILE_ROLES = _model_roles()
TACTILE_HW = (64, 64)
TACTILE_KEYS = ("tactile_images", "tactile_valid", "tactile_age_ms")
TACTILE_IMAGE_SHAPE = (len(TACTILE_ROLES), 3, *TACTILE_HW)
MAX_TACTILE_PACKET = 80 * 1024
XENSE_RAW_SCHEMA = "loopinfra.xense_rectify_raw.v1"
MAX_TACTILE_RAW_PACKET = 4 * 4096 * 4096 * 3 + 64 * 1024


def empty_tactile() -> dict[str, np.ndarray]:
    """Explicit absence; zeros never stand for a valid no-contact measurement."""
    return {
        "tactile_images": np.zeros(TACTILE_IMAGE_SHAPE, dtype=np.uint8),
        "tactile_valid": np.zeros(len(TACTILE_ROLES), dtype=bool),
        "tactile_age_ms": np.zeros(len(TACTILE_ROLES), dtype=np.float32),
    }


def decode_tactile_raw_packet(packet: bytes, observation_id: str) -> dict:
    """XTR1 -> {"observation_id", "server_received_monotonic_ns", "sensors": [...]}.

    One sensor row per selected role; valid rows carry "image" (HxWx3 uint8,
    SDK channel order, full Rectify resolution) checked against its sha256.
    """
    if not isinstance(packet, bytes) or not 8 <= len(packet) <= MAX_TACTILE_RAW_PACKET:
        raise ValueError("invalid tactile raw packet size")
    if packet[:4] != b"XTR1":
        raise ValueError("invalid tactile raw packet magic")
    length = struct.unpack("!I", packet[4:8])[0]
    if not 0 < length <= 64 * 1024 or 8 + length > len(packet):
        raise ValueError("invalid tactile raw header size")
    meta = json.loads(packet[8:8 + length])
    if meta.get("schema") != XENSE_RAW_SCHEMA or meta.get("observation_id") != observation_id:
        raise ValueError("tactile raw snapshot mismatch")
    body = memoryview(packet)[8 + length:]
    if len(body) != meta.get("payload_bytes"):
        raise ValueError("invalid tactile raw payload length")
    sensors = []
    for row in meta.get("sensors") or []:
        if row.get("role") not in WIRE_ROLES:
            raise ValueError("invalid tactile raw role")
        row = dict(row)
        if row.get("valid"):
            shape = row.get("shape")
            if (row.get("dtype") != "uint8" or not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
                    or row["length"] != shape[0] * shape[1] * 3):
                raise ValueError("invalid tactile raw image layout")
            data = bytes(body[row["offset"]:row["offset"] + row["length"]])
            if hashlib.sha256(data).hexdigest() != row.get("sha256"):
                raise ValueError("tactile raw image checksum mismatch")
            row["image"] = np.frombuffer(data, dtype=np.uint8).reshape(shape)
        sensors.append(row)
    return {"observation_id": observation_id,
            "server_received_monotonic_ns": meta.get("server_received_monotonic_ns"), "sensors": sensors}


def decode_tactile_packet(packet: bytes, expected: Mapping) -> dict[str, np.ndarray]:
    """Verify the exact snapshot of the RGB ID, schema, role order and checksum."""
    if not isinstance(packet, bytes) or not 8 <= len(packet) <= MAX_TACTILE_PACKET:
        raise ValueError("invalid tactile packet size")
    if packet[:4] != b"XTO1":
        raise ValueError("invalid tactile packet magic")
    length = struct.unpack("!I", packet[4:8])[0]
    if not 0 < length <= 16 * 1024 or 8 + length > len(packet):
        raise ValueError("invalid tactile header size")
    def reject(value):
        raise ValueError(f"non-finite tactile metadata: {value}")
    meta = json.loads(packet[8:8 + length], parse_constant=reject)
    if not isinstance(meta, dict) or meta != dict(expected):
        raise ValueError("tactile snapshot descriptor mismatch")
    if (meta.get("schema") != XENSE_SCHEMA or meta.get("roles") != list(WIRE_ROLES)
            or meta.get("shape") != [4, *TACTILE_HW, 3] or meta.get("dtype") != "uint8"):
        raise ValueError("unsupported tactile schema or layout")
    body = packet[8 + length:]
    if len(body) != 4 * 64 * 64 * 3 or meta.get("payload_bytes") != len(body):
        raise ValueError("invalid tactile image length")
    if hashlib.sha256(body).hexdigest() != meta.get("sha256"):
        raise ValueError("tactile image checksum mismatch")
    sensors = meta.get("sensors")
    now = meta.get("server_received_monotonic_ns")
    if (type(now) is not int or now <= 0 or not isinstance(sensors, list)
            or len(sensors) != 4 or meta.get("max_age_ms") != 100.0
            or meta.get("alignment_method") != "server_receipt_approximation"
            or meta.get("physical_sync_verified") is not False):
        raise ValueError("invalid tactile timing contract")
    out = empty_tactile()
    images = np.frombuffer(body, dtype=np.uint8).reshape(4, 64, 64, 3).transpose(0, 3, 1, 2)
    for j, sensor in enumerate(sensors):
        if (not isinstance(sensor, Mapping) or sensor.get("role") != WIRE_ROLES[j]
                or type(sensor.get("valid")) is not bool):
            raise ValueError("invalid tactile role or validity")
        if sensor["role"] not in TACTILE_ROLES:
            continue                                   # not a model input
        i = TACTILE_ROLES.index(sensor["role"])
        if sensor["valid"]:
            age, stamp = sensor.get("age_ms"), sensor.get("monotonic_ns")
            if (not isinstance(age, (int, float)) or isinstance(age, bool)
                    or not math.isfinite(age) or not 0 <= age <= 100
                    or type(stamp) is not int or not 0 <= now - stamp <= 100_000_000
                    or abs(age - (now - stamp) / 1e6) > 1e-6):
                raise ValueError("invalid tactile sample age")
            out["tactile_images"][i] = images[j]
            out["tactile_valid"][i] = True
            out["tactile_age_ms"][i] = age
    return out
