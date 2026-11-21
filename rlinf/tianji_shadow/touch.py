"""Read-only Xense sampler. Stage2 remains the only control client."""
import hashlib
import json
import struct
import threading
import time
import urllib.parse
import urllib.request
from collections import deque

import cv2
import numpy as np

ROLES = ("left_gripper_1", "right_gripper_2")
MAX_AGE_NS = 46_509_044

def decode_raw(packet, observation_id):
    if len(packet) < 8 or packet[:4] != b"XTR1" or len(packet) > 12_000_000:
        raise ValueError("invalid XTR1 packet")
    n = struct.unpack("!I", packet[4:8])[0]
    if not 0 < n <= min(65536, len(packet)-8):
        raise ValueError("invalid XTR1 header")
    meta = json.loads(packet[8:8+n]); body = packet[8+n:]
    if meta.get("schema") != "loopinfra.xense_rectify_raw.v1" or meta.get("observation_id") != observation_id:
        raise ValueError("raw observation mismatch")
    if meta.get("payload_bytes") != len(body):
        raise ValueError("raw length mismatch")
    seen = set(); end = 0
    for sensor in meta["sensors"]:
        if sensor["role"] in seen:
            raise ValueError("duplicate tactile role")
        seen.add(sensor["role"])
        if sensor["valid"]:
            offset, length = sensor["offset"], sensor["length"]
            if sensor.get("shape") != [700,400,3] or sensor.get("dtype") != "uint8":
                raise ValueError("training requires Rectify uint8 H700 W400 C3")
            if offset != end or length != 700*400*3 or offset+length > len(body):
                raise ValueError("invalid tactile extent")
            data = body[offset:offset+length]; end += length
            if hashlib.sha256(data).hexdigest() != sensor["sha256"]:
                raise ValueError("tactile checksum mismatch")
            sensor["image"] = np.frombuffer(data,np.uint8).reshape(700,400,3)
    if end != len(body):
        raise ValueError("unreferenced tactile payload")
    return meta


class TouchFeed:
    def __init__(self, collector, selected):
        self.collector = collector.rstrip("/")
        self.selected = selected
        self.buffers = [deque(maxlen=48) for _ in ROLES]
        self.lock = threading.Lock()
        self.refs = [None, None]
        self.invalidated = [False, False]
        self.start_ns = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="shadow-xense")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.thread.start()

    def get(self, path):
        with self.opener.open(self.collector+path, timeout=.25) as response:
            return response.read(12_000_001)

    def run(self):
        while not self.stop.wait(.001 if self.selected() else .08):
            if not self.selected():
                with self.lock:
                    for frames in self.buffers:
                        frames.clear()
                continue
            tick = time.monotonic()
            try:
                oid = json.loads(self.get("/api/observation"))["observation_id"]
                payload = self.get("/api/tianji/tactile_raw?observation_id="+urllib.parse.quote(oid, safe=""))
                self.ingest(decode_raw(payload, oid))
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            self.stop.wait(max(.001, 1/30 - (time.monotonic()-tick)))

    def ingest(self, packet):
        receipt = packet["server_received_monotonic_ns"]
        if not 0 <= time.monotonic_ns()-receipt <= 1_000_000_000:
            raise ValueError("collector monotonic clock mismatch")
        roles = {r["role"]: r for r in packet["sensors"]}
        with self.lock:
            self.error = None
            for i, role in enumerate(ROLES):
                row = roles.get(role, {})
                if not row.get("valid"):
                    continue
                ns, seq, serial = [row.get(k) for k in ("monotonic_ns", "sequence", "serial")]
                if type(ns) is not int or type(seq) is not int or not serial or not 0 <= receipt-ns <= 100_000_000:
                    continue
                frames = self.buffers[i]
                if frames:
                    prev = frames[-1]
                    if prev["serial"] != serial or ns < prev["ns"] or seq < prev["seq"]:
                        frames.clear()
                        self.invalidated[i] = True
                    elif prev["ns"] == ns or prev["seq"] == seq:
                        continue
                rgb = cv2.resize(row["image"], (80, 140), interpolation=cv2.INTER_AREA).transpose(2,0,1)
                frame = dict(ns=ns, seq=seq, serial=serial, rgb=rgb)
                frames.append(frame)
                if self.start_ns is not None and self.refs[i] is None and self.start_ns <= ns <= self.start_ns + 350_000_000:
                    self.refs[i] = frame

    def arm(self, ns):
        with self.lock:
            self.start_ns = ns
            self.invalidated = [False, False]
            self.refs = [next((f for f in reversed(frames) if 0 <= ns-f["ns"] <= 150_000_000), None)
                         for frames in self.buffers]

    def align(self, capture):
        rgb = np.zeros((2, 3, 140, 80), np.uint8)
        refs = rgb.copy()
        meta = np.zeros((2, 3), np.float32)
        mapping = []
        with self.lock:
            for i, role in enumerate(ROLES):
                frame = next((f for f in reversed(self.buffers[i]) if f["ns"] <= capture), None)
                ref = self.refs[i]
                age = capture-frame["ns"] if frame else None
                valid = (frame is not None and ref is not None and ref["ns"] <= capture and
                         age <= MAX_AGE_NS and frame["serial"] == ref["serial"] and not self.invalidated[i])
                if valid:
                    rgb[i], refs[i] = frame["rgb"], ref["rgb"]
                    meta[i] = [1, 1, np.log1p(np.float32(age/1e6))]
                reason = ("valid" if valid else "sensor_changed" if self.invalidated[i] else
                          "missing_past" if frame is None else "missing_reference" if not ref else
                          "future_reference" if ref["ns"] > capture else "too_old")
                mapping.append(dict(role=role, valid=bool(valid), reason=reason, age_ns=age,
                    sequence=frame["seq"] if frame else None, ns=frame["ns"] if frame else None,
                    serial=frame["serial"] if frame else None, ref_ns=ref["ns"] if ref else None,
                    collector_error=self.error))
        return rgb, refs, meta, mapping

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
