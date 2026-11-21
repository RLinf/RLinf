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

"""Small append-only transition recorder for HIL smoke and playback tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .tianji_kinematics import JOINT_ACTION_SPACES, JOINT_ROW_SEMANTICS

JOINT_STAGE2_FIELDS = (
    "state_joint",
    "next_state_joint",
    "action_tcp",
    "exec_action_joint",
    "exec_action_joint_source",
    "intervene_action_tcp",
    "intervene_action_joint_source",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


class TianjiTransitionRecorder:
    """Write transition rows without silently dropping HIL metadata."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        source_jpeg=False,
        image_roles=("head", "wrist_A", "wrist_B"),
        write_rows=True,
    ):
        """``write_rows=False`` (11-06, with the episode archive and hil_raw): rows are
        still built and returned for the archive's steps.jsonl, but transitions.jsonl
        and frames/ are not written (both duplicate the archive). Events still are."""
        self.source_jpeg = bool(source_jpeg)
        self.image_roles = tuple(image_roles)
        self.write_rows = bool(write_rows)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames_directory = self.directory / "frames"
        self.path = self.directory / "transitions.jsonl"
        self._stream = None
        if self.write_rows:
            self.frames_directory.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")
        self.count = 0

    def append(
        self,
        *,
        observation,
        action,
        next_observation,
        reward,
        terminated,
        truncated,
        info,
    ):
        frame_started = time.monotonic()
        frame_path = (
            self._write_frames(self.count, observation, next_observation) if self.write_rows else None
        )
        frames_write_seconds = time.monotonic() - frame_started
        row = {
            # rtj joint Stage2 rows mix spaces; "action_spaces" names each field's.
            "semantics": info.get("row_semantics", "tianji.absolute_tcp_rot6.v2"),
            "index": self.count,
            "step_before_record_seconds": info.get("step_before_record_seconds"),
            "step_started_monotonic": info.get("step_started_monotonic"),
            "frames_write_seconds": frames_write_seconds,
            "observation_id": observation.observation_id,
            "next_observation_id": next_observation.observation_id,
            "state": observation.state,
            "next_state": next_observation.state,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "intervene_flag": bool(info.get("intervene_flag", False)),
            "intervene_action": info.get("intervene_action"),
            "iv_kind": info.get("iv_kind", 0),
            "exec_action": info.get("exec_action", action),
            "intervene_action_available": bool(
                info.get("intervene_action_available", False)
            ),
            "intervene_state": info.get("intervene_state"),
            "exec_action_available": bool(info.get("exec_action_available", False)),
            "receipt": info.get("receipt", {}),
            "hil_outcome": info.get("hil_outcome"),
            "step_failure": bool(info.get("step_failure", False)),
            "control_mode": info.get("control_mode"),
            "observed_action": info.get("observed_action"),
            "transition_padding": bool(info.get("transition_padding", False)),
            # Stage1 sparse execution rows intentionally carry no RGB.  Keep
            # the absence explicit so readers never infer that an older frame
            # belongs to this state/time.
            "observation_frame_roles": sorted(
                (getattr(observation, "frames", {}) or {}).keys()
            ),
            "next_observation_frame_roles": sorted(
                (getattr(next_observation, "frames", {}) or {}).keys()
            ),
        }
        row["missing_frame_roles"] = {
            "observation": sorted(
                set(self.image_roles) - set(row["observation_frame_roles"])
            ),
            "next_observation": sorted(
                set(self.image_roles) - set(row["next_observation_frame_roles"])
            ),
        }
        for key in JOINT_STAGE2_FIELDS:
            # rtj joint Stage2: every state and action also in the other space.
            if key in info:
                row[key] = info[key]
        for name, obs in (("tactile_observation", observation), ("tactile_next_observation", next_observation)):
            descriptor = (getattr(obs, "raw", None) or {}).get("xense_tactile")
            if descriptor is not None:
                row[name] = descriptor
        if row["semantics"] == JOINT_ROW_SEMANTICS:
            row["action_spaces"] = JOINT_ACTION_SPACES
        if "stage" in info:
            # Subtask number, its prompt and completed-subtask rewards.
            row.update(
                subtask=info["stage"],
                subtask_prompt=info.get("task_description"),
                milestone=info.get("milestone", 0),
            )
        if frame_path is not None:
            row["frames_path"] = frame_path
        line = json.dumps(_json_value(row), ensure_ascii=False, allow_nan=False)
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()
        self.count += 1
        # The episode archive reuses the encoded row (tianji_episode_archive).
        return line

    def _write_frames(self, index, observation, next_observation):
        """Persist current and next RGB frames without putting pixels in JSON."""
        current = getattr(observation, "frames", {}) or {}
        following = getattr(next_observation, "frames", {}) or {}
        roles = sorted(set(current) | set(following))
        current_touch = getattr(observation, "tactile", None)
        next_touch = getattr(next_observation, "tactile", None)
        if not roles and current_touch is None and next_touch is None:
            return None
        arrays = {}
        for prefix, tactile in (("observation", current_touch), ("next", next_touch)):
            if tactile is not None:
                for key, value in tactile.items():
                    arrays[f"{prefix}_{key}"] = np.asarray(value)
        current_jpeg = getattr(observation, "images_jpeg", {}) or {}
        following_jpeg = getattr(next_observation, "images_jpeg", {}) or {}
        use_jpeg = self.source_jpeg and all(
            (role not in current or role in current_jpeg)
            and (role not in following or role in following_jpeg)
            for role in roles
        )
        for role in roles:
            if role in current:
                arrays[f"observation_{role}"] = (
                    np.frombuffer(current_jpeg[role], dtype=np.uint8)
                    if use_jpeg
                    else np.asarray(current[role], dtype=np.uint8)
                )
            if role in following:
                arrays[f"next_{role}"] = (
                    np.frombuffer(following_jpeg[role], dtype=np.uint8)
                    if use_jpeg
                    else np.asarray(following[role], dtype=np.uint8)
                )
        if self.source_jpeg:
            # Keep the exact camera JPEG bytes already received for inference.
            # No re-encoding, duplicate lossy compression, or queued records.
            arrays["frame_encoding"] = np.asarray("jpeg" if use_jpeg else "rgb")
        filename = self.frames_directory / f"{index:08d}.npz"
        writer = np.savez if self.source_jpeg else np.savez_compressed
        writer(filename, **arrays)
        return str(filename.relative_to(self.directory))

    def close(self):
        if self._stream is not None and not self._stream.closed:
            self._stream.close()
        manifest = {
            "format": "rlinf.tianji.transitions.v2",
            "rows": self.count,
            # 11-06: False = rows only in episodes/*/steps.jsonl (TIANJI_HIL_RAW_RECORDING=1).
            "rows_written": self.write_rows,
            "path": str(self.path),
            "frames_directory": str(self.frames_directory),
            "frames_format": (
                "npz; frame_encoding=jpeg: original JPEG uint8 bytes; rgb: RGB arrays; keys observation_<role>/next_<role>"
                if self.source_jpeg
                else "npz; keys observation_<role>/next_<role>"
            ),
        }
        temporary = self.directory / ".manifest.partial"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.directory / "manifest.json")

    def record_event(self, event):
        """Record reset outcomes separately so they never become training actions."""
        with (self.directory / "episode_events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    _json_value({"timestamp_ns": time.time_ns(), **event}),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def load_transition_frames(path):
    """Decode both legacy RGB archives and Stage1 source-JPEG archives."""
    from .tianji_backend import _decode_image

    with np.load(path, allow_pickle=False) as archive:
        encoding = (
            str(archive["frame_encoding"]) if "frame_encoding" in archive else "rgb"
        )
        if encoding not in {"rgb", "jpeg"}:
            raise ValueError(f"Unknown transition frame encoding: {encoding}")
        return {
            key: _decode_image(archive[key].tobytes())
            if encoding == "jpeg"
            else archive[key].copy()
            for key in archive.files
            if key != "frame_encoding"
        }
