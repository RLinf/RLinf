"""Optional one-way inference observation feed for the car monitor."""

import io
import os
import queue
import socket
import threading

import numpy as np


class MonitorTap:
    """Best-effort AF_UNIX datagrams. A slow subscriber never blocks rollout."""

    def __init__(self, endpoint, max_stage: int | None = None):
        if max_stage is None:
            raw_max = os.environ.get("TIANJI_CAR_NUM_SUBTASKS", "15").strip()
            max_stage = int(raw_max) if raw_max else 15
        self._max_stage = int(max_stage) if max_stage and int(max_stage) > 0 else None
        self._pending = queue.Queue(maxsize=4) if endpoint else None
        if self._pending is not None:
            threading.Thread(target=self._send, args=(endpoint,), daemon=True).start()

    def publish(self, metadata, rlt, joint):
        if self._pending is None:
            return
        try:
            self._pending.put_nowait((metadata.detach(), rlt.detach(), joint.detach()))
        except queue.Full:
            return

    def _send(self, endpoint):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.setblocking(False)
            while True:
                metadata, rlt, joint = self._pending.get()
                try:
                    metadata = metadata.cpu().numpy()
                    rlt, joint = rlt.float().cpu().numpy(), joint.float().cpu().numpy()
                    if rlt.shape != (len(metadata), 2048) or joint.shape != (len(metadata), 16):
                        continue
                    for meta, z, state in zip(metadata, rlt, joint, strict=True):
                        if meta[0] <= 0 or not meta[5] or meta[2] < 1:
                            continue
                        if self._max_stage is not None and meta[2] > self._max_stage:
                            continue
                        if not np.isfinite(z).all() or not np.isfinite(state).all():
                            continue
                        packet = io.BytesIO()
                        np.savez(packet, capture_ns=meta[0], episode=meta[1],
                                 stage=meta[2], stage_revision=meta[3],
                                 control_revision=meta[4], rlt=z, joint=state)
                        client.sendto(packet.getvalue(), endpoint)
                except (OSError, RuntimeError, ValueError):
                    # Optional consumer may stop, be absent or run behind the policy.
                    continue
