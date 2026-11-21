"""Best-effort observation transport and the accepted HIL event journal."""
import json
import logging
import os
import queue
import socket
import threading
import time
from pathlib import Path

from .review import append_review, episode_file, human_row

SIZE = 104
logger = logging.getLogger(__name__)
ROOT = os.getenv("TIANJI_SHADOW_RUNTIME")
_gate = Path(ROOT, "enabled") if ROOT else None
_output = os.getenv("TIANJI_RLT_OUTPUT")
REVIEW_ROOT = Path(_output) if ROOT and _output and os.getenv("TIANJI_SHADOW_EPISODE_REVIEW") == "1" else None
_outbox = queue.Queue(maxsize=8)
_thread = None
_dropped = 0


def selected():
    try:
        return _gate is not None and _gate.is_file()
    except OSError:
        return False


def selection(value):
    if _gate is not None:
        if value:
            _gate.touch()
        else:
            _gate.unlink(missing_ok=True)


def status():
    if ROOT is None:
        return None
    try:
        model = json.loads(Path(ROOT, "status.json").read_text())
    except (OSError, ValueError):
        model = {"state": "loading"}
    return dict(model, selected=selected(), version=os.environ.get("TIANJI_SHADOW_VERSION", ""))


def _send(kind, header, payload=b""):
    if ROOT is None:
        return
    prefix = json.dumps(dict(header, kind=kind), allow_nan=False).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
        client.setblocking(False)
        client.sendto(prefix + b"\0" + payload, str(Path(ROOT, "feed")))


def event(name, hil, *, journal=True, **extra):
    if ROOT is None:
        return
    try:
        event = dict(kind="event", name=name, ns=time.monotonic_ns(), episode=hil._episode,
                     stage=hil._subtask, revision=hil._subtask_revision,
                     collection=hil._collection_revision, phase=hil._phase,
                     human=hil._human_latched, **extra)
        if REVIEW_ROOT is not None and hil.mode == "training":
            row = human_row(event)
            if row is not None:
                if name == "start":
                    hil._review_start_ns = event["ns"]
                    row["monitor_enabled"] = journal
                start_ns = hil._review_start_ns
                if start_ns is not None:
                    event["review_start_ns"] = start_ns
                    try:
                        path = episode_file(REVIEW_ROOT, event["episode"], start_ns)
                        append_review(path, row)
                    except OSError:
                        logger.exception("Could not write human review event")
                    if row["kind"] == "episode_end":
                        hil._review_start_ns = None
        if not journal:
            return
        # The low-rate accepted event journal is separate from the feature
        # datagrams, so a full prediction queue cannot drop a human label.
        with Path(ROOT, "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, allow_nan=False)+"\n")
    except (OSError, ValueError):
        # Dropped accepted events are detectable by the per-session sequence.
        pass


def metadata(observation, panel, read_ns):
    import numpy as np
    value = np.zeros(SIZE, dtype=np.int64)
    if not panel or panel["phase"] != "running" or panel["human_latched"]:
        return value
    if not selected() or not observation.frames:
        return value
    sync = observation.raw.get("sync") or {}
    capture = sync.get("reference_monotonic_ns")
    if type(capture) is not int or sync.get("sync_valid") is False:
        capture = 0
    oid = str(observation.observation_id).encode()[:SIZE - 7]
    value[:7] = (panel["episode"], panel["subtask"], panel["subtask_revision"],
                 panel["collection_revision"], capture, read_ns, len(oid))
    value[7:7+len(oid)] = list(oid)
    return value


def _drain():
    global _dropped
    while True:
        header, z, q = _outbox.get()
        try:
            _send("feature", dict(header, dropped=_dropped), z.float().cpu().numpy().tobytes() +
                  q.float().cpu().numpy().tobytes())
        except Exception:
            _dropped += 1


def tap(obs, rlt):
    global _thread, _dropped
    if ROOT is None or "shadow_meta" not in obs:
        return
    try:
        if not selected():
            return
        metas = obs["shadow_meta"].tolist()
        for i, m in enumerate(metas):
            if m[0] <= 0:
                continue
            episode, stage, revision, collection, capture, read, length = m[:7]
            if not 1 <= stage <= 4:
                continue
            z, q = rlt["z_rl"][i], rlt["proprio"][i]
            if z.numel() != 2048 or q.numel() != 16:
                _send("invalid", dict(episode=episode, reason="feature_shape"))
                continue
            header = dict(episode=episode, stage=stage, revision=revision,
                          collection=collection, capture=capture, read=read,
                          ns=time.monotonic_ns(), oid=bytes(m[7:7+length]).decode(),
                          prompt=obs["task_descriptions"][i], dropped=_dropped)
            if _thread is None:
                _thread = threading.Thread(target=_drain, daemon=True, name="shadow-feature")
                _thread.start()
            _outbox.put_nowait((header, z.detach(), q.detach()))
    except queue.Full:
        _dropped += 1
    except Exception:
        _dropped += 1
