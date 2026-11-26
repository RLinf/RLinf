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

"""Private socket protocol and Kit process entry point. Keep startup stdlib-only."""

from __future__ import annotations

import os
import pickle
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, suppress
from importlib.machinery import PathFinder
from importlib.util import find_spec
from pathlib import Path

_LENGTH = struct.Struct("!Q")


def _set_timeout(sock: socket.socket, deadline: float | None) -> None:
    if deadline is None:
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Isaac Lab-Arena IPC deadline expired")
    sock.settimeout(remaining)


def send_message(
    sock: socket.socket, value: object, deadline: float | None = None
) -> None:
    """Send a complete header and payload within one request deadline."""
    # NumPy 2's protocol-5 array reconstructor is absent in OpenPI's NumPy 1.26.
    payload = pickle.dumps(value, protocol=4)
    for block in (_LENGTH.pack(len(payload)), payload):
        _set_timeout(sock, deadline)
        sock.sendall(block)


def _read(sock: socket.socket, size: int, deadline: float | None) -> bytearray:
    data = bytearray(size)
    view = memoryview(data)
    while view:
        _set_timeout(sock, deadline)
        received = sock.recv_into(view)
        if not received:
            raise EOFError("Arena socket closed before the complete message arrived")
        view = view[received:]
    return data


def receive_message(sock: socket.socket, deadline: float | None = None):
    """Receive a full frame, including when only part of its payload is available."""
    (size,) = _LENGTH.unpack(_read(sock, _LENGTH.size, deadline))
    return pickle.loads(_read(sock, size, deadline))


def start_watchdog(leader: int) -> subprocess.Popen:
    """Give the controller a waitable child independent of Kit's native shutdown."""
    return subprocess.Popen(
        [sys.executable, "-I", __file__, "--watch", str(os.getpid()), str(leader)],
        start_new_session=True,
    )


def _watch_process_group(owner: int, leader: int) -> None:
    status = Path(f"/proc/{leader}/stat")
    while os.getppid() == owner:
        try:
            state, parent, *_ = status.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            break
        if state == "Z" or int(parent) != owner:
            break
        time.sleep(0.5)
    with suppress(ProcessLookupError):
        os.killpg(leader, signal.SIGKILL)


def _prepare_kit_environment() -> None:
    # Different Kit installations cannot mix carb plugins or USD schemas.
    # The pinned wheel fills unset paths itself, also for split Kit packages.
    spec = find_spec("isaacsim")
    if spec and spec.origin:
        root = Path(spec.origin).resolve().parent
        kit = root / "kit"
        if not (kit / "extscore").is_dir():
            # Resolve a split omniverse-kit wheel without importing its bootstrap.
            omni = find_spec("omni")
            if omni and omni.submodule_search_locations:
                split = PathFinder.find_spec(
                    "omni.kit_app", omni.submodule_search_locations
                )
                if split and split.origin:
                    kit = Path(split.origin).resolve().parent
        for key, target in (
            ("ISAAC_PATH", root),
            ("EXP_PATH", root / "apps"),
            ("CARB_APP_PATH", kit),
        ):
            value = os.environ.get(key)
            if value and (
                not target.is_dir() or Path(value).resolve() != target.resolve()
            ):
                os.environ.pop(key)
    for key in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES"):
        value = os.environ.get(key)
        if value:
            # Vulkan accepts both manifest files and manifest directories.
            paths = [p for p in value.split(os.pathsep) if p and Path(p).exists()]
            if paths:
                os.environ[key] = os.pathsep.join(paths)
            else:
                os.environ.pop(key)


def _serve(sock: socket.socket, stack: ExitStack) -> None:
    # The controller arms the watchdog before sending any launch settings.
    settings = receive_message(sock)
    _prepare_kit_environment()
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        device=settings["device"], enable_cameras=True, headless=True, multi_gpu=False
    ).app
    stack.callback(app.close)
    # -I excludes the script directory. Load only our sibling after Kit starts.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simulation import ArenaRollout, _make_env

    auto_reset = settings.pop("auto_reset")
    all_frames = settings.pop("all_frames")
    env = _make_env(**settings)
    stack.callback(env.close)
    rollout = ArenaRollout(env, auto_reset, all_frames)
    send_message(sock, (True, env.unwrapped.metadata["render_fps"]))
    while True:
        try:
            command, payload = receive_message(sock)
        except EOFError:
            return
        if command == "reset":
            result = rollout.reset(*payload)
        elif command == "chunk_step":
            result = rollout.chunk_step(payload)
        else:
            raise ValueError(f"Unknown Arena request {command!r}")
        send_message(sock, (True, result))


if __name__ == "__main__":
    if sys.argv[1] == "--watch":
        _watch_process_group(int(sys.argv[2]), int(sys.argv[3]))
    else:
        with socket.socket(fileno=int(sys.argv.pop())) as sock, ExitStack() as stack:
            try:
                _serve(sock, stack)
            except Exception:
                send_message(sock, (False, traceback.format_exc()))
