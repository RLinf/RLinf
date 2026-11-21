# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Joint fork: a 16D joint Stage1 model driving LoopInfra joint targets.

``TIANJI_ACTION_SPACE=joint`` selects it. The model sees the image-aligned
measured joints LoopInfra's Stage1 adapter already publishes plus the two
commanded closures, and predicts absolute joint targets (the joints learn the
difference to the current measured joints, closures stay absolute).

The Stage1 loop is unchanged and keeps its 20D absolute-TCP rows for logs,
RTC bookkeeping and the panel: each predicted joint row is mapped by the
arm-local native FK with the learned TCP at flange +Z 160 mm, exactly how the
training labels and the server's Stage1 state are computed. The joint rows
stay in a small cache keyed by the rows the loop executes, so the backend
sends the predicted joints themselves and RTC gets the previous joint chunk.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
from pathlib import Path

import numpy as np

JOINT_TARGET_PROTOCOL = "loopinfra.policy_joint_target.v1"
JOINT_STATE_DIM = 16
JOINT_DIMS = [d for a0 in (0, 8) for d in range(a0, a0 + 7)]
JOINT_GRIPPER_DIMS = (7, 15)
TCP_FLANGE_Z_M = 0.160
# Encoded rows are within [-1, 1] (positions / 2 m); 1e-5 is 20 µm or 1e-5 of a
# rotation column, far below one control step of motion.
ROW_MATCH_ATOL = 1e-5


def joint_mode(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get("TIANJI_ACTION_SPACE", "eef")
    if value not in ("eef", "joint"):
        raise ValueError("TIANJI_ACTION_SPACE must be eef or joint")
    return value == "joint"


def joint_state(raw, eef_state) -> np.ndarray:
    """16D model state from the Stage1 observation: aligned joints + commanded closures."""
    payload = (raw or {}).get("tianji_stage1") or {}
    joints = np.asarray(payload.get("joints"), dtype=np.float64)
    if joints.shape != (14,) or not np.isfinite(joints).all():
        raise ValueError("LoopInfra Stage1 observation has no image-aligned joints")
    eef_state = np.asarray(eef_state, dtype=np.float32)
    state = np.concatenate([joints[:7], [eef_state[9]], joints[7:], [eef_state[19]]])
    return state.astype(np.float32)


from .tianji_contract import default_loopinfra_root


class NativeTcpFK:
    """Arm-local native FK of the official SDK, learned TCP at flange +Z 160 mm."""

    def __init__(self, loopinfra_root=None):
        root = default_loopinfra_root(loopinfra_root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from loopinfra.client.sdk_control_adapter import NativeKinematics

        self.kinematics = NativeKinematics(str(root / "third_party/TJ_FX_ROBOT_CONTRL_SDK"))

    def tcp_rows(self, joints) -> np.ndarray:
        """[H,16] joints + closures -> [H,20] absolute TCP (xyz, rot6 columns, closure)."""
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != JOINT_STATE_DIM or not np.isfinite(joints).all():
            raise ValueError("joint chunk must be finite [H,16]")
        rows = np.empty((len(joints), 20), dtype=np.float64)
        for arm, j0, t0 in (("A", 0, 0), ("B", 8, 10)):
            for i, q in enumerate(joints[:, j0 : j0 + 7]):
                m = np.asarray(self.kinematics.matrix([float(x) for x in q], arm), dtype=np.float64)
                r = m[:3, :3]
                rows[i, t0 : t0 + 3] = m[:3, 3] / 1000.0 + r @ [0.0, 0.0, TCP_FLANGE_Z_M]
                rows[i, t0 + 3 : t0 + 6] = r[:, 0]
                rows[i, t0 + 6 : t0 + 9] = r[:, 1]
            rows[:, t0 + 9] = joints[:, j0 + 7]
        return rows.astype(np.float32)


class JointTargetCache:
    """Recent predicted chunks: executed 20D rows -> the predicted joint rows."""

    def __init__(self, keep_chunks=4):
        self._lock = threading.Lock()
        self._chunks = collections.deque(maxlen=keep_chunks)

    def add(self, physical, encoded, joints):
        physical = np.asarray(physical, dtype=np.float32)
        encoded = np.asarray(encoded, dtype=np.float32)
        joints = np.asarray(joints, dtype=np.float64)
        if not (len(physical) == len(encoded) == len(joints)):
            raise ValueError("joint cache rows must align")
        with self._lock:
            self._chunks.append((physical.copy(), encoded.copy(), joints.copy()))

    def joints_for_encoded(self, row) -> np.ndarray:
        """Joint row predicted for this executed (encoded, validated) action row."""
        row = np.asarray(row, dtype=np.float32)
        with self._lock:
            for _, encoded, joints in reversed(self._chunks):
                hit = np.flatnonzero(np.all(np.abs(encoded - row) <= ROW_MATCH_ATOL, axis=1))
                if hit.size:
                    return joints[hit[0]].copy()
        raise KeyError("no predicted joint target for this action")

    def joints_for_physical_chunk(self, physical) -> np.ndarray:
        """The joint chunk whose TCP rows are ``physical`` (RTC's previous chunk)."""
        physical = np.asarray(physical, dtype=np.float32)
        with self._lock:
            for rows, _, joints in reversed(self._chunks):
                if rows.shape == physical.shape and np.allclose(rows, physical, atol=ROW_MATCH_ATOL * 2):
                    return joints.copy()
        raise KeyError("previous chunk is not a cached joint prediction")

    def clear(self):
        with self._lock:
            self._chunks.clear()


# Predictor and LoopInfra backend run in the same Stage1 execute process.
CACHE = JointTargetCache()
