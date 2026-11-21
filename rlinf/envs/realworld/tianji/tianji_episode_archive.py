# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode archive of Stage2 HIL training for offline RL and LeRobot.

Recording only: nothing here feeds training or control, no entry point
raises, and no disk I/O runs on the control, inference or env-worker threads
(they hand copies to one writer thread). The cameras are recorded by a
separate process (tianji_image_stream.py), so their bytes and their polling
never touch the training process. A failed write is listed in the episode
manifest's ``issues`` and the episode keeps everything else.

``runs/<run>/hil/episodes/episode_NNNNNN/`` (NNNNNN: the panel's episode) is
published in the background within a fraction of a second of the episode's
end, normally during the return motion; nothing waits for it, and a clean
shutdown finishes it. ``.episode_NNNNNN.partial`` is an episode still
recording or interrupted; a discarded one is deleted.

- steps.jsonl: one row per executed step: the step recorder's row
  (tianji_recorder) plus ``episode``, ``chunk`` (rollout serial),
  ``chunk_step``, the SDK capture times of the step's observation and of the
  next one (the clock of the image indexes), and takeover start/end markers.
- chunks.jsonl: one row per action chunk: its steps, the policy input
  observation (``policy_input``) and how the inference was used.
- policy/<chunk>.npz: the chunk's inference as the rollout returned it
  (``inference.*``: ref_chunk = VLA reference, action/model_action = actor,
  actions_by_stage, z_rl, proprio, ...) and as the env worker executed it
  (``routed.*``: after stage-head selection and stage relabelling).
- labels/<chunk>.npz: the env's per-step training labels of the chunk after
  its own completion rules (filled executed actions, masks, rewards, iv_kind).
- images/stream.bin + stream.jsonl: every three-camera group the collector
  published during the episode (original JPEG bytes, at its sync rate);
  images/inputs.bin + inputs.jsonl: the groups read as policy inputs. Index
  rows give each group's SDK capture time and its bytes' offsets.
- manifest.json: outcome, counts, action coding, image coverage, issues,
  file hashes, the run's training configuration.
"""

from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import logging
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .tianji_recorder import _json_value

LOG = logging.getLogger(__name__)
FORMAT = "rlinf.tianji.hil_episode.v1"
# An ended episode waits this long for its last chunk's inference record.
INFERENCE_WAIT_S = 10.0
# The camera process ends an episode's stream once a group captured after
# its last step arrived, or after this (longer than the collector's rare
# publishing gaps); publication waits for its report at most IMAGE_REPORT_WAIT_S.
IMAGE_TAIL_S = 0.5
IMAGE_REPORT_WAIT_S = 3.0
FLUSH_INTERVAL_S = 1.0
# A step without an image group this close (SDK capture time) is reported.
IMAGE_MATCH_NS = 50_000_000


HIL_RAW_ENV = "TIANJI_HIL_RAW_RECORDING"
HIL_RAW_START_TIMEOUT_S = 150.0   # the collector first waits for the previous episode to seal
HIL_RAW_STOP_TIMEOUT_S = 20.0


class _HilRawRecording:
    """rtj fork (11-06): one LoopInfra raw recording per HIL episode, at the source rates.

    The collector client records its three cameras (H.264 at 30 fps, capture clock), the 300 Hz
    arm feedback and every sent joint reference / gripper command; the collector server records
    Xense at 30 Hz and the Pico streams (LoopInfra_rtj loopinfra/server/hil_recording.py). This
    worker thread starts one at ``begin_episode`` and stops it at the episode's end; the env
    control thread only enqueues. Directories: runs/<run>/hil_raw/<timestamp>/ (LoopInfra raw
    episode layout, manifest.json there); the episode archive's manifest.json ``hil_raw``
    names it. All times are this host's monotonic clock, as in steps.jsonl.
    """

    def __init__(self, archive, url: str, token: str | None, root: Path):
        self.archive, self.url, self.token, self.root = archive, url.rstrip("/"), token, Path(root)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="tianji-hil-raw", daemon=True)
        self._thread.start()

    def start(self, key, number, info: dict):
        self._queue.put(("start", key, {"number": number, **info,
                                        "begin_monotonic_ns": time.monotonic_ns()}))

    def stop(self, key, outcome, reason):
        end = time.monotonic_ns()
        # The episode end goes to the manifest now: without the camera process the episode
        # can publish before the collector's stop reply (stop_state, best effort) arrives.
        self.archive._queue.put(("hil_raw", key, {"end_monotonic_ns": end}))
        self._queue.put(("stop", key, {"outcome": outcome, "reason": reason, "end_monotonic_ns": end}))

    def stage(self, key, payload: dict):
        """11-06: the episode's stage at its first step and at every change (hil_raw events)."""
        self._queue.put(("stage", key, payload))

    def close(self, timeout=30.0):
        self._queue.put(None)
        self._thread.join(timeout)

    def _post(self, body: dict, timeout: float) -> dict:
        import urllib.error
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.url + "/api/hil-recording",
                                         data=json.dumps(body).encode("utf-8"), headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"http_{exc.code}: {exc.read()[:300].decode(errors='replace')}") from None

    def _report(self, key, report: dict):
        print("TIANJI_HIL_RAW " + json.dumps({"key": key, **report}, ensure_ascii=False, default=str), flush=True)
        self.archive._queue.put(("hil_raw", key, report))

    def _run(self):
        running = {}  # archive episode key -> LoopInfra raw episode_id
        while True:
            item = self._queue.get()
            if item is None:
                for key, episode_id in list(running.items()):
                    try:
                        self._post({"operation": "stop", "episode_id": episode_id,
                                    "reason": "archive_closed", "normal": False}, HIL_RAW_STOP_TIMEOUT_S)
                    except Exception:  # noqa: BLE001
                        pass
                return
            operation, key, payload = item
            if operation == "stage":
                episode_id = running.get(key)
                if episode_id is not None:
                    try:
                        self._post({"operation": "stage", "episode_id": episode_id, "stage": payload}, 5.0)
                    except Exception as exc:  # noqa: BLE001 -- only printed; the archive is unchanged
                        print(f"TIANJI_HIL_RAW_STAGE_FAILED key={key} {type(exc).__name__}: {exc}", flush=True)
                continue
            try:
                if operation == "start":
                    asked = time.monotonic_ns()
                    result = self._post({"operation": "start", "root": str(self.root),
                                         "label": payload["number"], "info": payload}, HIL_RAW_START_TIMEOUT_S)
                    if result.get("episode_id"):
                        running[key] = result["episode_id"]
                    started = result.get("started_monotonic_ns")
                    self._report(key, {
                        "directory": result.get("directory"), "episode_id": result.get("episode_id"),
                        "state": result.get("state"), "error": result.get("error"),
                        "begin_monotonic_ns": payload["begin_monotonic_ns"],
                        "started_monotonic_ns": started,
                        "start_gap_s": (started - payload["begin_monotonic_ns"]) / 1e9 if started else None,
                        "request_s": (time.monotonic_ns() - asked) / 1e9})
                else:
                    episode_id = running.pop(key, None)
                    if episode_id is None:
                        self._report(key, {"stop": "not started"})
                        continue
                    result = self._post({"operation": "stop", "episode_id": episode_id,
                                         "reason": str(payload["outcome"]), "normal": True}, HIL_RAW_STOP_TIMEOUT_S)
                    self._report(key, {"end_monotonic_ns": payload["end_monotonic_ns"],
                                       "stop_state": result.get("state"), "stop_error": result.get("error")})
            except Exception as exc:  # noqa: BLE001 -- recording never stops training
                self._report(key, {f"{operation}_error": f"{type(exc).__name__}: {exc}"})


def _capture_ns(observation) -> int | None:
    sync = ((getattr(observation, "raw", None) or {}).get("sync")) or {}
    stamp = sync.get("reference_monotonic_ns")
    return stamp if type(stamp) is int and stamp > 0 else None


def _image_group(metadata: dict, images: dict, source: str) -> dict:
    """One camera group keyed by role; ``images`` is keyed by camera id or role."""
    infos, roles = {}, {}
    for info in metadata.get("images") or []:
        calibration = info.get("calibration")
        role = (
            info.get("physical_role")
            or info.get("role")
            or (calibration.get("physical_role") if isinstance(calibration, dict) else None)
            or info.get("camera_id")
        )
        roles[info.get("camera_id")] = role
        infos[role] = info
    by_role = {}
    for key, data in images.items():
        role = roles.get(key, key)
        info = infos.get(role) or {}
        by_role[role] = (
            bytes(data),
            {k: info[k] for k in ("camera_id", "width", "height", "sha256") if k in info},
        )
    return {
        "observation_id": str(metadata.get("observation_id")),
        "capture_ns": (metadata.get("sync") or {}).get("reference_monotonic_ns"),
        "reference_time_ns": metadata.get("reference_time_ns"),
        "session_id": metadata.get("session_id"),
        "control_generation": metadata.get("control_generation"),
        "source": source,
        "images": by_role,
    }


def _arrays(value, prefix, out):
    """Copy tensors/arrays (also inside dicts) into ``out`` as numpy arrays."""
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _arrays(item, f"{prefix}.{key}", out)
        return
    if hasattr(value, "detach"):
        value = value.detach()
        if value.dtype.is_floating_point and value.dtype.itemsize < 4:
            value = value.float()  # bf16 has no numpy dtype
        out[prefix] = value.cpu().numpy().copy()
    elif isinstance(value, np.ndarray):
        out[prefix] = value.copy()


def snapshot_inference(policy_output) -> dict:
    """Copy a rollout output before the env worker routes it in place."""
    out = {}
    _arrays(getattr(policy_output, "forward_inputs", None), "inference", out)
    for name in ("actions", "versions", "bootstrap_values", "prev_logprobs", "prev_values",
                 "intervene_flags"):
        _arrays(getattr(policy_output, name, None), f"inference.output_{name}", out)
    return out


def _npz(arrays: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _sync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _File:
    """An append-only file of one episode with its running SHA256."""

    def __init__(self, path: Path):
        self.stream = path.open("xb")
        self.hash = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        offset = self.size
        self.stream.write(data)
        self.hash.update(data)
        self.size += len(data)
        return offset

    def close(self, sync=True):
        if self.stream.closed:
            return
        self.stream.flush()
        if sync:
            os.fsync(self.stream.fileno())
        self.stream.close()


# LoopInfra raw_recording.py's Xense encoder: libx264 crf 15, 30 fps, yuv420p,
# rgb24 input (SDK channel order kept), fragmented MP4. It runs in the
# collector's Python (PyAV), fed raw HxWx3 bytes on stdin.
XENSE_VIDEO_CRF = os.environ.get("LOOPINFRA_XENSE_VIDEO_CRF", "15")
_TACTILE_ENCODER = r"""
import json, os, struct, sys
frames, error = 0, None
try:
    os.nice(5)
    import av
    import numpy as np
    path, height, width, crf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    source = sys.stdin.buffer
    container = av.open(path, 'w', options={'movflags': 'frag_keyframe+empty_moov+default_base_moof'})
    try:
        stream = container.add_stream('libx264', rate=30)
        stream.height, stream.width, stream.pix_fmt = height, width, 'yuv420p'
        stream.options = {'crf': crf, 'preset': 'veryfast', 'g': '30', 'threads': '2'}
        while True:
            head = source.read(8)
            size = struct.unpack('<Q', head)[0] if len(head) == 8 else 0
            if not size:
                break
            data = source.read(size)
            if len(data) != size:
                break
            image = np.frombuffer(data, np.uint8).reshape(height, width, 3)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                container.mux(packet)
            frames += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
except BaseException as exc:
    error = repr(exc)
sys.stdout.write(json.dumps({'encoded': frames, 'error': error}))
"""


def _encoder_python() -> str:
    """PyAV lives in the collector's environment, not in the training one."""
    explicit = os.environ.get("TIANJI_XENSE_ENCODER_PYTHON")
    if explicit:
        return explicit
    marker = Path(os.environ.get("LOOPINFRA_ROOT", str(Path.home() / "LoopInfra_rtj"))) / ".loopinfra-python"
    try:
        return marker.read_text().strip() or "python3"
    except OSError:
        return str(Path.home() / "miniconda3/envs/loopinfra/bin/python")


class TactileVideo:
    """One encoder subprocess per role; frames keep submission order (= frame_index)."""

    def __init__(self, path: Path, shape):
        import subprocess

        self.path, self.shape = Path(path), tuple(shape)
        self.frames = 0
        self.error = None
        self.process = subprocess.Popen(
            [_encoder_python(), "-c", _TACTILE_ENCODER, str(self.path), str(self.shape[0]),
             str(self.shape[1]), XENSE_VIDEO_CRF],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def submit(self, image: np.ndarray):
        if self.error:
            return None
        try:
            data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
            self.process.stdin.write(struct.pack("<Q", len(data)))
            self.process.stdin.write(data)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        index, self.frames = self.frames, self.frames + 1
        return index

    def close(self, timeout=60, kill=False) -> dict:
        encoded = None
        try:
            if kill:
                self.process.kill()
            else:
                self.process.stdin.write(struct.pack("<Q", 0))
                self.process.stdin.close()
                result = json.loads(self.process.stdout.read() or b"{}")
                self.process.wait(timeout=timeout)
                encoded = result.get("encoded")
                self.error = self.error or result.get("error") or (
                    None if self.process.returncode == 0 else f"xense_encoder_exit_{self.process.returncode}")
        except Exception as exc:  # noqa: BLE001
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            if self.process.poll() is None:
                self.process.kill()
        return {"video": self.path.name, "frames": self.frames, "encoded": encoded,
                "dropped": 0, "error": self.error}


class _Episode:
    """Writer-thread state of one episode directory."""

    NAMES = ("steps.jsonl", "chunks.jsonl", "images/inputs.jsonl", "images/inputs.bin")

    def __init__(self, root: Path, key: int, info: dict):
        self.key = key
        self.number = int(info["number"])
        name = f"episode_{self.number:06d}"
        for n in itertools.count(2):
            if not (root / name).exists() and not (root / f".{name}.partial").exists():
                break
            name = f"episode_{self.number:06d}_{n}"
        self.final = root / name
        self.partial = root / f".{name}.partial"
        for sub in ("policy", "labels", "images"):
            (self.partial / sub).mkdir(parents=True)
        self.files = {name: _File(self.partial / name) for name in self.NAMES}
        self.blobs = {}  # npz path -> (sha256, bytes)
        self.info = dict(info)
        self.started_ns = time.time_ns()
        self.steps = 0
        self.step_capture = []
        self.chunks = {}  # chunk serial -> chunks.jsonl row
        self.order = []
        self.inference = set()
        self.labelled = set()
        self.input_groups = []  # SDK capture_ns of the policy-input groups
        self.input_ids = set()
        self.input_bytes = 0
        self.images_started = False
        self.stream = None  # the camera process's report
        self.outcome = self.reason = None
        self.steps_ended = self.done = self.flushed = False
        self.ended_at = self.done_at = self.done_enqueued = None
        self.issues = []
        self.result = None
        self.tactile_videos = {}   # role -> TactileVideo
        self.tactile_report = {}
        self.tactile_ids = set()
        self.tactile_errors = set()

    def write_blob(self, folder: str, serial: int, arrays: dict) -> str:
        name = f"{folder}/{serial:06d}.npz"
        data = _npz(arrays)
        with (self.partial / name).open("xb") as stream:
            stream.write(data)
        self.blobs[name] = (hashlib.sha256(data).hexdigest(), len(data))
        return name

    def images_reported(self, now: float) -> bool:
        return (
            not self.images_started
            or self.stream is not None
            or now - (self.ended_at or now) > IMAGE_REPORT_WAIT_S
        )


class TianjiEpisodeArchive:
    """Record every Stage2 training episode without touching training.

    The env worker calls ``next_chunk`` and ``record_inference``; the env's
    control thread ``begin_episode``, ``begin_chunk``, ``add_step``,
    ``end_episode`` and ``end_chunk``; inputs read for inference arrive through
    ``policy_input``. One writer thread does all file I/O; the cameras are
    read and written by the camera process while an episode runs.
    """

    def __init__(
        self,
        recorder_dir,
        *,
        server_url: str | None,
        token: str | None = None,
        contract=None,
        control_hz: float | None = None,
        image_roles=("head", "wrist_A", "wrist_B"),
        prompts=(),
        image_hz: float | None = None,
        action_contract=None,
    ):
        """``image_hz``: cap on the recorded camera rate; None keeps every group.
        ``action_contract``: the actions' contract when it is not ``contract``
        (rtj joint Stage2: 16D joints; ``contract`` then frames the TCP labels)."""
        self.root = Path(recorder_dir) / "episodes"
        self.root.mkdir(parents=True, exist_ok=True)
        self._static = {
            "format": FORMAT,
            "action_semantics": "tianji.absolute_tcp_rot6.v2",
            "action_coordinates": "rl_bounds_normalized",
            "control_hz": control_hz,
            "image_roles": list(image_roles),
            "image_rate_cap_hz": image_hz or None,
            "subtask_prompts": list(prompts),
        }
        if contract is not None:
            self._static.update(
                action_scale=np.asarray(contract.action_scale).tolist(),
                action_bias=np.asarray(contract.action_bias).tolist(),
                action_chunk_length=getattr(contract, "action_chunk_length", None),
            )
        if action_contract is not None and contract is not None and action_contract is not contract:
            from .tianji_kinematics import JOINT_ACTION_SPACES

            # rtj joint Stage2: actions are joints; the critic's labels are TCP.
            self._static.update(
                action_semantics=getattr(action_contract, "semantics", None),
                action_scale=np.asarray(action_contract.action_scale).tolist(),
                action_bias=np.asarray(action_contract.action_bias).tolist(),
                tcp_action_semantics="tianji.absolute_tcp_rot6.v2",
                tcp_action_scale=np.asarray(contract.action_scale).tolist(),
                tcp_action_bias=np.asarray(contract.action_bias).tolist(),
                action_spaces=JOINT_ACTION_SPACES,
            )
        self._run_info = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._keys = itertools.count(1)
        self._numbers = itertools.count(1)
        self._own_serials = itertools.count(-1, -1)
        self._next_serial = 0
        self._pending_serial = None
        self._episode = None  # (key, number) while an episode records
        self._chunk = None
        self._chunk_step = 0
        self._takeover = False
        self._last_capture = None
        self._last_input = None
        self._logged = set()
        self._closed = False
        self._images = None
        if server_url and os.environ.get(HIL_RAW_ENV, "0") == "1":
            # 11-06: the cameras (and Xense) are in hil_raw at 30 Hz; no ~11.5 Hz copy here.
            self._static["camera_stream"] = "hil_raw (30 Hz LoopInfra raw episode, see hil_raw)"
            self._static["tactile_full_resolution"] = "hil_raw/xense (30 Hz); xense/ here stays empty"
        elif server_url:
            try:
                # The collector syncs its cameras at the control rate (TIANJI_CONTROL_HZ).
                self._images = _ImageProcess(
                    self, server_url, token, 1.0 / control_hz if control_hz else 1 / 30,
                    1.0 / image_hz if image_hz else 0.0,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("episode archive: no camera process (%s); steps still recorded", exc)
                self._static["camera_process_error"] = str(exc)
        # rtj fork (11-06): every episode also recorded at the source at 30 Hz (LoopInfra).
        self._hil_raw = None
        if server_url and os.environ.get(HIL_RAW_ENV, "0") == "1":
            raw_root = Path(recorder_dir).resolve().parent / "hil_raw"
            self._hil_raw = _HilRawRecording(self, server_url, token, raw_root)
            self._static["hil_raw_root"] = str(raw_root)
        self._writer = threading.Thread(
            target=self._write_loop, name="tianji-episode-archive", daemon=True
        )
        self._writer.start()

    def _note(self, where: str, exc: BaseException, key=None):
        message = f"{where}: {type(exc).__name__}: {exc}"
        if where not in self._logged:
            self._logged.add(where)
            LOG.warning("episode archive %s (training continues)", message)
        self._queue.put(("issue", key, message))

    # -- env worker -------------------------------------------------------
    def set_run_info(self, info: dict):
        try:
            self._run_info = _json_value(dict(info))
        except Exception as exc:  # noqa: BLE001
            self._note("set_run_info", exc)

    def next_chunk(self) -> int | None:
        """Serial of the chunk the env worker is about to execute."""
        with self._lock:
            serial = self._next_serial
            self._next_serial += 1
            self._pending_serial = serial
        return serial

    def record_inference(self, serial, received: dict | None, executed: dict, notes: dict):
        """``received``: snapshot_inference() of the rollout output; ``executed``:
        the routed tensors sent to the env (copied here, before later edits)."""
        if serial is None:
            return
        try:
            arrays = dict(received or {})
            for name, value in executed.items():
                _arrays(value, f"routed.{name}", arrays)
            self._queue.put(("inference", None, {"serial": serial, "arrays": arrays,
                                                 "notes": dict(notes)}))
        except Exception as exc:  # noqa: BLE001
            self._note("record_inference", exc)

    # -- env control thread -----------------------------------------------
    def begin_episode(self, number, observation, *, subtask=None):
        try:
            if self._episode is not None:
                self._end_steps("interrupted", "a new episode started first")
            key = next(self._keys)
            number = int(number) if number else next(self._numbers)
            self._episode = (key, number)
            self._chunk, self._chunk_step, self._takeover = None, 0, False
            self._last_capture = _capture_ns(observation)
            self._hil_stage, self._hil_step = None, 0
            if self._hil_raw is not None:
                self._hil_raw.start(key, number, {
                    "subtask": subtask, "run": self.root.parent.parent.name,
                    "first_observation_id": getattr(observation, "observation_id", None),
                    "first_capture_ns": self._last_capture})
            self._queue.put(("begin", key, {
                "number": number,
                "subtask": subtask,
                "first_observation_id": getattr(observation, "observation_id", None),
                "first_capture_ns": self._last_capture,
            }))
            self.policy_input(observation, "reset")
        except Exception as exc:  # noqa: BLE001
            self._note("begin_episode", exc)

    def begin_chunk(self, size=None):
        try:
            with self._lock:
                serial, self._pending_serial = self._pending_serial, None
            if serial is None:  # no env-worker hook (tests, other runners)
                serial = next(self._own_serials)
            self._chunk, self._chunk_step = serial, 0
            if self._episode is not None:
                self._queue.put(("chunk", self._episode[0], {
                    "serial": serial, "size": size, "policy_input": self._last_input,
                }))
        except Exception as exc:  # noqa: BLE001
            self._note("begin_chunk", exc)

    def add_step(self, line: str | None, previous, observation, info: dict):
        """``line``: the step recorder's JSON row of this step."""
        try:
            if self._episode is None or not line:
                return
            takeover = bool(info.get("human_latched")) or info.get("iv_kind") in (2, 3)
            head = json.dumps({
                "episode": self._episode[1],
                "chunk": self._chunk,
                "chunk_step": self._chunk_step,
                "capture_ns": _capture_ns(previous),
                "next_capture_ns": _capture_ns(observation),
                "takeover_start": takeover and not self._takeover,
                "takeover_end": self._takeover and not takeover,
            })
            self._takeover = takeover
            if self._hil_raw is not None:
                # 11-06: the stage timeline into hil_raw (the first step and every change).
                stage = info.get("stage")
                if stage is not None and stage != getattr(self, "_hil_stage", None):
                    prompts = self._static.get("subtask_prompts") or []
                    self._hil_raw.stage(self._episode[0], {
                        "stage": int(stage), "previous_stage": getattr(self, "_hil_stage", None),
                        "prompt": prompts[int(stage) - 1] if 1 <= int(stage) <= len(prompts) else None,
                        "rlinf_episode": self._episode[1], "rlinf_step": getattr(self, "_hil_step", 0),
                        "step_capture_ns": _capture_ns(previous), "monotonic_ns": time.monotonic_ns()})
                    self._hil_stage = stage
                self._hil_step = getattr(self, "_hil_step", 0) + 1
            self._chunk_step += 1
            self._last_capture = _capture_ns(observation) or self._last_capture
            row = (head[:-1] + "," + line[1:] + "\n").encode("utf-8")
            self._queue.put(("step", self._episode[0], (row, _capture_ns(previous))))
        except Exception as exc:  # noqa: BLE001
            self._note("add_step", exc)

    def end_episode(self, outcome, reason=None):
        """At the terminal or fault step, before the return motion."""
        try:
            self._end_steps(outcome, reason)
        except Exception as exc:  # noqa: BLE001
            self._note("end_episode", exc)

    def _end_steps(self, outcome, reason):
        if self._episode is None:
            return
        key = self._episode[0]
        self._queue.put(("end_steps", key, {"outcome": outcome, "reason": reason}))
        if self._images is not None:
            self._images.stop(key, self._last_capture)
        if self._hil_raw is not None:
            self._hil_raw.stop(key, outcome, reason)

    @staticmethod
    def _label_arrays(labels: dict) -> dict:
        arrays = {}
        for name, value in labels.items():
            if isinstance(value, (list, tuple)):
                if value:
                    steps = [np.asarray(v.detach().cpu() if hasattr(v, "detach") else v)
                             for v in value]
                    arrays[name] = np.stack(steps, axis=1)[0]  # [C, ...] of env 0
            elif value is not None:
                _arrays(value, name, arrays)
                if name in arrays and arrays[name].ndim and arrays[name].shape[0] == 1:
                    arrays[name] = arrays[name][0]
        return arrays

    def end_chunk(self, labels: dict, done: bool):
        """After the env's own chunk labels, before an auto-reset."""
        try:
            if self._episode is None:
                return
            key = self._episode[0]
            try:
                arrays = self._label_arrays(labels)
            except Exception as exc:  # noqa: BLE001 -- the episode still ends here
                self._note("end_chunk labels", exc, key)
                arrays = {}
            if done:
                self._episode = None
                if self._images is not None:  # also when a wrapper ended it
                    self._images.stop(key, self._last_capture)
            # Never waits: the writer publishes in the background, normally
            # during the return motion that precedes this call.
            self._queue.put(("labels", key, {"serial": self._chunk, "arrays": arrays,
                                             "done": bool(done), "at": time.monotonic()}))
        except Exception as exc:  # noqa: BLE001
            self._note("end_chunk", exc)

    def wait_published(self, timeout=10.0) -> bool:
        """Block until every ended episode is published (tests, shutdown)."""
        done = threading.Event()
        self._queue.put(("barrier", None, done))
        return done.wait(timeout)

    def policy_input(self, observation, source: str):
        """An observation read for inference (reset, prefetch or refresh)."""
        try:
            episode = self._episode
            if episode is None or observation is None:
                return
            jpeg = getattr(observation, "images_jpeg", None) or {}
            self._last_input = {
                "source": source,
                "observation_id": getattr(observation, "observation_id", None),
                "capture_ns": _capture_ns(observation),
                "images": bool(jpeg),
            }
            descriptor = (getattr(observation, "raw", None) or {}).get("xense_tactile")
            if descriptor is not None:
                # The inference tensor blob keeps uint8 pixels; this keeps
                # their serials, sequence and server-clock provenance.
                self._last_input["xense_tactile"] = copy.deepcopy(descriptor)
            if jpeg:
                self._queue.put(("input", episode[0],
                                 _image_group(observation.raw or {}, jpeg, source)))
            raw = getattr(observation, "tactile_raw", None)
            if raw is not None:
                self._queue.put(("tactile", episode[0], raw))
        except Exception as exc:  # noqa: BLE001
            self._note("policy_input", exc)

    def close(self, timeout=10.0):
        """Publish ended episodes; one still running stays .partial."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._hil_raw is not None:
                self._hil_raw.close()  # stops a running recording; its reports are queued first
            if self._images is not None:
                self._images.close()  # its last reports are queued before "close"
            self._queue.put(("close", None, None))
            self._writer.join(timeout)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("episode archive close: %s", exc)

    def _images_done(self, key, report):
        self._queue.put(("images_done", key, report))

    # -- writer thread ----------------------------------------------------
    def _write_loop(self):
        episodes: dict[int, _Episode] = {}
        early = {}  # inference that arrived before its chunk began
        owner = {}  # chunk serial -> episode key
        barriers = []  # wait_published() callers
        last_flush = time.monotonic()
        while True:
            try:
                kind, key, payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                kind = key = payload = None
            now = time.monotonic()
            ep = episodes.get(key)
            try:
                if kind == "close":
                    for ep in list(episodes.values()):
                        self._finish(ep, episodes)
                    return
                if kind == "begin":
                    episodes[key] = _Episode(self.root, key, payload)
                elif kind == "barrier":
                    barriers.append(payload)
                elif kind == "issue":
                    target = ep or next(reversed(episodes.values()), None)
                    if target is not None:
                        target.issues.append(payload)
                elif kind == "inference":
                    serial = payload["serial"]
                    target = episodes.get(owner.get(serial))
                    if target is not None:
                        self._write_inference(target, payload)
                    elif serial not in owner:
                        early[serial] = payload  # its chunk has not begun yet
                elif ep is None:
                    pass  # a failed or already published episode
                elif kind == "chunk":
                    if self._images is not None and not ep.order:
                        # From the first chunk: the reset observation is already an
                        # input group, and the first episode waits for the learner.
                        ep.images_started = self._images.start(
                            key, ep.info["first_observation_id"], ep.partial / "images"
                        )
                    serial = payload["serial"]
                    owner[serial] = key
                    ep.order.append(serial)
                    ep.chunks[serial] = {
                        "chunk": serial, "index": len(ep.order) - 1, "first_step": ep.steps,
                        "size": payload["size"], "policy_input": payload["policy_input"],
                        "inference": None, "labels": None,
                    }
                    if serial in early:
                        self._write_inference(ep, early.pop(serial))
                elif kind == "step":
                    row, capture = payload
                    ep.files["steps.jsonl"].write(row)
                    ep.steps += 1
                    if capture is not None:
                        ep.step_capture.append(capture)
                elif kind == "input":
                    self._write_input(ep, payload)
                elif kind == "tactile":
                    self._write_tactile(ep, payload)
                elif kind == "hil_raw":
                    ep.__dict__.setdefault("hil_raw", {}).update(payload)
                elif kind == "images_done":
                    ep.stream = payload
                    if payload.get("error"):
                        ep.issues.append(f"camera stream: {payload['error']}")
                elif kind == "end_steps":
                    if not ep.steps_ended:
                        ep.steps_ended, ep.ended_at = True, now
                        ep.outcome, ep.reason = payload["outcome"], payload["reason"]
                elif kind == "labels":
                    self._write_labels(ep, payload)
                    if payload["done"]:
                        ep.done, ep.done_at, ep.done_enqueued = True, now, payload["at"]
                        if not ep.steps_ended:  # ended by a wrapper, not by the env
                            ep.steps_ended, ep.ended_at, ep.outcome = True, now, "ended"
            except Exception as exc:  # noqa: BLE001
                LOG.warning("episode archive write failed: %s", exc)
                if ep is not None:
                    ep.issues.append(f"{kind}: {type(exc).__name__}: {exc}")
            for ep in list(episodes.values()):
                try:
                    self._advance(ep, episodes, now)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("episode archive publish failed: %s", exc)
                    ep.issues.append(f"publish: {type(exc).__name__}: {exc}")
                    self._release(ep, episodes, {"saved": False, "error": str(exc)})
            if barriers and not any(ep.steps_ended for ep in episodes.values()):
                for event in barriers:
                    event.set()
                barriers.clear()
            if now - last_flush > FLUSH_INTERVAL_S:
                last_flush = now
                for ep in episodes.values():
                    for f in ep.files.values():
                        try:
                            if not f.stream.closed:
                                f.stream.flush()
                        except Exception:  # noqa: BLE001
                            pass

    def _write_inference(self, ep: _Episode, payload):
        serial = payload["serial"]
        name = ep.write_blob("policy", serial, payload["arrays"])
        ep.inference.add(serial)
        row = ep.chunks.get(serial)
        if row is not None:
            row["inference"] = name
            row.update(payload["notes"])

    def _write_labels(self, ep: _Episode, payload):
        serial = payload["serial"]
        row = ep.chunks.get(serial)
        if row is None:
            ep.issues.append(f"labels for chunk {serial}, which never began")
            return
        row["steps"] = ep.steps - row["first_step"]
        if payload["arrays"]:
            row["labels"] = ep.write_blob("labels", serial, payload["arrays"])
            ep.labelled.add(serial)

    def _write_input(self, ep: _Episode, group: dict):
        if group["observation_id"] in ep.input_ids:
            return
        data_file, index = ep.files["images/inputs.bin"], ep.files["images/inputs.jsonl"]
        entry = {k: v for k, v in group.items() if k != "images"}
        entry.update(group=len(ep.input_ids), file="inputs.bin", images={})
        for role, (data, info) in group["images"].items():
            entry["images"][role] = dict(info, offset=data_file.write(data), length=len(data))
            ep.input_bytes += len(data)
        index.write((json.dumps(_json_value(entry), ensure_ascii=False) + "\n").encode("utf-8"))
        ep.input_ids.add(group["observation_id"])
        if type(group["capture_ns"]) is int:
            ep.input_groups.append(group["capture_ns"])

    def _write_tactile(self, ep: _Episode, raw: dict):
        """One inference-time snapshot into xense/<role>.mp4 + .jsonl, rows as the collector writes them."""
        if raw.get("error"):
            if raw["error"] not in ep.tactile_errors:
                ep.tactile_errors.add(raw["error"])
                ep.issues.append(f"tactile raw: {raw['error']}")
            return
        if raw["observation_id"] in ep.tactile_ids:
            return
        ep.tactile_ids.add(raw["observation_id"])
        for row in raw["sensors"]:
            image = row.get("image")
            if image is None:
                continue                                  # the collector writes no row without a sample
            role = row["role"]
            video = ep.tactile_videos.get(role)
            if video is None:
                (ep.partial / "xense").mkdir(exist_ok=True)
                video = ep.tactile_videos[role] = TactileVideo(ep.partial / "xense" / f"{role}.mp4", image.shape)
                ep.files[f"xense/{role}.jsonl"] = _File(ep.partial / "xense" / f"{role}.jsonl")
            if tuple(image.shape) != video.shape:
                ep.issues.append(f"tactile raw {role}: shape {list(image.shape)} != {list(video.shape)}")
                continue
            sample = {"sequence": row.get("sequence"), "monotonic_ns": row.get("monotonic_ns"),
                      "sdk_timestamp": row.get("sdk_timestamp"),
                      "payload": {"encoding": "h264_mp4", "video": f"{role}.mp4",
                                  "frame_index": video.submit(image), "dtype": "uint8",
                                  "shape": list(image.shape)}}
            line = {"receive_monotonic_ns": row.get("monotonic_ns"), "sample": sample}
            ep.files[f"xense/{role}.jsonl"].write(
                (json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))

    def _close_tactile(self, ep: _Episode, *, discard: bool = False):
        for role, video in ep.tactile_videos.items():
            result = video.close(kill=discard)
            ep.tactile_report[role] = result
            if result.get("error") and not discard:
                ep.issues.append(f"tactile video {role}: {result['error']}")
        ep.tactile_videos.clear()

    def _advance(self, ep: _Episode, episodes: dict, now: float):
        """Flush once the steps and the stream ended; publish once every chunk is complete."""
        if not ep.steps_ended or ep.result is not None or not ep.images_reported(now):
            return
        if ep.outcome == "discard":
            if ep.done:
                # 11-05: TIANJI_KEEP_DISCARDED=1 keeps it under <recorder>/discarded (never trained on).
                keep = os.environ.get("TIANJI_KEEP_DISCARDED", "0") == "1"
                self._close_tactile(ep, discard=not keep)
                for f in ep.files.values():
                    f.close(sync=False)
                kept = None
                if keep:
                    kept = self.root.parent / "discarded" / ep.final.name
                    try:
                        kept.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(ep.partial, kept)
                    except OSError as exc:
                        print(f"EPISODE_ARCHIVE_DISCARD_KEEP_FAILED episode={ep.number}: {exc}", flush=True)
                        shutil.rmtree(ep.partial, ignore_errors=True)
                        kept = None
                else:
                    shutil.rmtree(ep.partial, ignore_errors=True)
                waited = round((time.monotonic() - ep.done_enqueued) * 1000)
                print(f"EPISODE_ARCHIVE_DISCARDED episode={ep.number} deleted_after_ms={waited}"
                      + (f" kept={kept}" if kept is not None else ""), flush=True)
                self._release(ep, episodes, {"saved": False, "outcome": "discard"})
            return
        if not ep.flushed:  # normally during the return motion
            for f in ep.files.values():
                f.stream.flush()
                os.fsync(f.stream.fileno())
            ep.flushed = True
        if not ep.done:
            return
        if any(s not in ep.inference for s in ep.order) and now - ep.done_at < INFERENCE_WAIT_S:
            return
        self._publish(ep, episodes)

    def _write_chunk_rows(self, ep: _Episode) -> list:
        missing = [s for s in ep.order if s not in ep.inference]
        unlabelled = [s for s in ep.order if s not in ep.labelled]
        if missing:
            ep.issues.append(f"inference missing for chunks {missing}")
        if unlabelled:
            ep.issues.append(f"labels missing for chunks {unlabelled}")
        for serial in ep.order:
            ep.files["chunks.jsonl"].write(
                (json.dumps(_json_value(ep.chunks[serial]), ensure_ascii=False) + "\n").encode("utf-8")
            )
        return missing + unlabelled

    def _publish(self, ep: _Episode, episodes: dict):
        incomplete = self._write_chunk_rows(ep)
        self._close_tactile(ep)
        if ep.images_started and ep.stream is None:
            ep.issues.append("camera stream did not report its end")
        for f in ep.files.values():
            f.close()
        manifest = self._manifest(ep, complete=not incomplete and ep.steps > 0)
        with (ep.partial / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for sub in ("policy", "labels", "images") + (("xense",) if (ep.partial / "xense").exists() else ()):
            _sync_directory(ep.partial / sub)
        _sync_directory(ep.partial)
        os.replace(ep.partial, ep.final)
        _sync_directory(self.root)
        summary = {
            "path": str(ep.final), "outcome": ep.outcome, "complete": manifest["complete"],
            "steps": ep.steps, "chunks": len(ep.order),
            "image_groups": manifest["images"]["stream_groups"], "issues": len(ep.issues),
            "published_after_ms": round((time.monotonic() - ep.done_enqueued) * 1000)
            if ep.done_enqueued else None,
        }
        print("EPISODE_ARCHIVE_SAVED " + json.dumps(summary, ensure_ascii=False), flush=True)
        self._release(ep, episodes, dict(summary, saved=True))

    def _finish(self, ep: _Episode, episodes: dict):
        """Shutdown: an ended episode is published, a running one stays .partial."""
        if ep.result is not None:
            return
        if ep.outcome == "discard":
            self._close_tactile(ep, discard=True)
            for f in ep.files.values():
                f.close(sync=False)
            shutil.rmtree(ep.partial, ignore_errors=True)
        elif ep.done:
            self._publish(ep, episodes)
            return
        else:
            self._write_chunk_rows(ep)
            self._close_tactile(ep)
            for f in ep.files.values():
                f.close()
            ep.issues.append("interrupted before the episode ended")
            (ep.partial / "manifest.json").write_text(
                json.dumps(self._manifest(ep, complete=False), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._release(ep, episodes, {"saved": False})

    def _release(self, ep: _Episode, episodes: dict, result: dict):
        ep.result = result
        episodes.pop(ep.key, None)

    def _manifest(self, ep: _Episode, complete: bool) -> dict:
        stream = ep.stream or {}
        captures = list(ep.input_groups)
        index = ep.partial / "images" / "stream.jsonl"
        if index.exists():
            for line in index.read_text(encoding="utf-8").splitlines():
                capture = json.loads(line).get("capture_ns")
                if type(capture) is int:
                    captures.append(capture)
        groups = np.unique(np.asarray(captures, dtype=np.int64))
        gaps = np.diff(groups)
        unmatched = len(ep.step_capture)
        if len(groups):
            capture = np.asarray(ep.step_capture, dtype=np.int64)
            right = np.clip(np.searchsorted(groups, capture), 0, len(groups) - 1)
            left = np.clip(right - 1, 0, len(groups) - 1)
            near = np.minimum(np.abs(groups[right] - capture), np.abs(groups[left] - capture))
            unmatched = int((near > IMAGE_MATCH_NS).sum())
        files = {name: {"sha256": f.hash.hexdigest(), "bytes": f.size} for name, f in ep.files.items()}
        files.update({name: {"sha256": h, "bytes": n} for name, (h, n) in ep.blobs.items()})
        files.update(stream.get("files") or {})
        for role, report in ep.tactile_report.items():
            video = ep.partial / "xense" / report["video"]
            if video.exists():
                files[f"xense/{report['video']}"] = {"bytes": video.stat().st_size}
        return _json_value({
            **self._static,
            "complete": bool(complete),
            "episode": ep.number,
            "outcome": ep.outcome,
            "end_reason": ep.reason,
            "steps": ep.steps,
            "chunks": len(ep.order),
            "chunks_with_inference": len(ep.inference),
            "chunks_with_labels": len(ep.labelled),
            "first_subtask": ep.info.get("subtask"),
            "first_observation_id": ep.info.get("first_observation_id"),
            "first_capture_ns": ep.info.get("first_capture_ns"),
            "started_unix_ns": ep.started_ns,
            "images": {
                "stream_groups": stream.get("groups", 0),
                "input_groups": len(ep.input_ids),
                "median_interval_ms": round(float(np.median(gaps)) / 1e6, 2) if len(gaps) else None,
                "max_interval_ms": round(float(gaps.max()) / 1e6, 2) if len(gaps) else None,
                "steps_without_group_within_50ms": unmatched,
                "stream": {k: v for k, v in stream.items() if k not in ("files", "event", "key")},
            },
            "xense": {
                "layout": "xense/<role>.mp4 + xense/<role>.jsonl as LoopInfra raw_recording writes them "
                          "(h264 crf 15, 30 fps container, frame_index = row order); one frame per "
                          "inference read (reset / prefetch / refresh), not a 30 Hz stream",
                "snapshots": len(ep.tactile_ids),
                "roles": ep.tactile_report,
            },
            # rtj fork (11-06): the 30 Hz source recording of this episode (LoopInfra raw
            # layout; cut it to [begin_monotonic_ns, end_monotonic_ns) for this episode).
            "hil_raw": getattr(ep, "hil_raw", None),
            "issues": ep.issues,
            "files": files,
            "run": self._run_info,
            "notes": {
                "steps": "rows of the step recorder; capture_ns is the SDK clock of the image indexes",
                "actions": "physical = normalized * action_scale + action_bias; grippers are commanded closures"
                + ("; TCP-space fields (action_spaces) use tcp_action_scale / tcp_action_bias"
                   if "tcp_action_scale" in self._static else ""),
                "policy_input": "a chunk's inference read chunks.jsonl policy_input (the prefetch of the previous chunk, or the reset observation)",
            },
        })


class _ImageProcess:
    """The archive's camera process (tianji_image_stream.py) and its reports."""

    def __init__(self, archive: TianjiEpisodeArchive, url: str, token: str | None, period: float,
                 min_period: float = 0.0):
        self._archive = archive
        # Held while a command is decided and sent, so a stop never overtakes its start.
        self._lock = threading.RLock()
        self._active = None
        self._last_started = 0
        self._stops = {}  # stops that arrived before their stream started
        env = dict(os.environ, TIANJI_IMAGE_TOKEN=token or "")
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).with_name("tianji_image_stream.py")),
             str(url), repr(float(period)), repr(float(min_period))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )
        self._reader = threading.Thread(target=self._read, name="tianji-archive-images", daemon=True)
        self._reader.start()

    def _send(self, **command) -> bool:
        try:
            with self._lock:
                self._process.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
                self._process.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def start(self, key, after_id, directory) -> bool:
        with self._lock:
            self._active, self._last_started = key, max(self._last_started, key)
            started = self._send(cmd="start", key=key, dir=str(directory), after=after_id)
            stop = self._stops.pop(key, None)
            if started and stop is not None:
                self._send(cmd="stop", key=key, after_capture_ns=stop[0], tail_s=IMAGE_TAIL_S)
            if not started:
                self._active = None
            return started

    def stop(self, key, last_capture_ns):
        with self._lock:
            if self._active != key:
                if key > self._last_started:
                    self._stops[key] = (last_capture_ns,)  # its start is still queued
                return
            self._send(cmd="stop", key=key, after_capture_ns=last_capture_ns, tail_s=IMAGE_TAIL_S)

    def close(self):
        self._send(cmd="close")
        try:
            self._process.stdin.close()
            self._process.wait(5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self._process.kill()
        self._reader.join(5)

    def _read(self):
        try:
            for line in self._process.stdout:
                report = json.loads(line)
                with self._lock:
                    if self._active == report.get("key"):
                        self._active = None
                self._archive._images_done(report.get("key"), report)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("episode archive camera process: %s", exc)
        with self._lock:
            active, self._active = self._active, None
        if active is not None:
            self._archive._images_done(active, {"error": "camera process exited"})


def load_episode(path) -> dict:
    """Read one archived episode: manifest, steps, chunks and image index rows."""
    path = Path(path)

    def rows(name):
        file = path / name
        return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] if file.exists() else []

    images = rows("images/inputs.jsonl") + rows("images/stream.jsonl")
    images.sort(key=lambda g: (g.get("capture_ns") is None, g.get("capture_ns") or 0))
    return {
        "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        "steps": rows("steps.jsonl"),
        "chunks": rows("chunks.jsonl"),
        "images": images,
    }


def read_jpeg(path, entry: dict, role: str) -> bytes:
    """Original JPEG bytes of one camera of an image index row."""
    image = entry["images"][role]
    with (Path(path) / "images" / entry["file"]).open("rb") as stream:
        stream.seek(image["offset"])
        return stream.read(image["length"])
