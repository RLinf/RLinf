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

"""Process and wire contracts for the upstream HumanoidArena runtimes."""

from __future__ import annotations

import atexit
import base64
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def request_json(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    """Exchange JSON with a local upstream service without using proxy settings."""
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    ) as response:
        return json.load(response)


def decode_observation(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decode the original front RGB image and native 64-dimensional state."""
    observation = payload["observation"]
    state = np.asarray(observation["state"], dtype=np.float32)
    item = observation["images"]["front"]
    shape = tuple(item["shape"])
    if len(shape) != 3 or shape[-1] != 3 or min(shape) < 1 or item["dtype"] != "uint8":
        raise ValueError(f"Expected uint8 HWC RGB image, got {shape}, {item['dtype']}")
    raw = base64.b64decode(item.get("data_b64", item.get("data")), validate=True)
    if len(raw) != int(np.prod(shape)):
        raise ValueError("RGB byte count does not match image shape")
    if state.shape != (64,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite state[64], got {state.shape}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy(), state


def validate_actions(actions: Any, chunk_size: int) -> np.ndarray:
    """Preserve native semantic40 values, including continuous hand targets."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[1:] != (chunk_size, 40):
        raise ValueError(f"Expected actions[B, {chunk_size}, 40], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("HumanoidArena actions must be finite")
    return actions


class RuntimeProcess:
    """Own a subprocess tree whose lifetime is bounded by its parent's pipe."""

    def __init__(
        self, command: list[str], log_path: Path, *, cwd: Path, env: dict[str, str]
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        with log_path.open("w") as log:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *command],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
        atexit.register(self.close)

    def check(self) -> None:
        """Fail when the supervised runtime exited before completing a request."""
        status = self.process.poll()
        if status is not None:
            raise RuntimeError(
                f"Runtime exited with status {status}; see {self.log_path}"
            )

    def close(self) -> None:
        """Release the lease and wait for the runtime tree to terminate."""
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        atexit.unregister(self.close)


def _supervise(command: list[str]) -> int:
    child = subprocess.Popen(
        [sys.executable, __file__, "--child", str(os.getpid()), *command],
        start_new_session=True,
    )

    def watch_lease() -> None:
        sys.stdin.buffer.read()
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    threading.Thread(target=watch_lease, daemon=True).start()
    return child.wait()


def _exec_child(parent_pid: int, command: list[str]) -> None:
    # Ray can kill the supervisor directly; bind the runtime to it in the kernel.
    if sys.platform != "linux":
        raise RuntimeError("The upstream Isaac runtime requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set runtime parent-death signal")
    if os.getppid() != parent_pid:
        raise SystemExit(1)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        _exec_child(int(sys.argv[2]), sys.argv[3:])
    raise SystemExit(_supervise(sys.argv[1:]))
