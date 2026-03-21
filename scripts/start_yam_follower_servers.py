#!/usr/bin/env python3
# Copyright 2026 Ying-Chun Lee
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

"""Launch YAM follower arm servers for RLinf inference."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_RIGHT_CAN = "can_right"
DEFAULT_LEFT_CAN = "can_left"
DEFAULT_RIGHT_PORT = 1234
DEFAULT_LEFT_PORT = 1235
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_GRIPPER = "linear_4310"
DEFAULT_MODE = "follower"
PORT_WAIT_TIMEOUT_S = 20.0
PORT_POLL_INTERVAL_S = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch YAM follower arm servers for RLinf inference."
    )
    parser.add_argument(
        "--right-can",
        type=str,
        default=DEFAULT_RIGHT_CAN,
        help=f"Right arm CAN channel (default: {DEFAULT_RIGHT_CAN}).",
    )
    parser.add_argument(
        "--left-can",
        type=str,
        default=DEFAULT_LEFT_CAN,
        help=f"Left arm CAN channel (default: {DEFAULT_LEFT_CAN}).",
    )
    parser.add_argument(
        "--right-port",
        type=int,
        default=DEFAULT_RIGHT_PORT,
        help=f"Right follower server port (default: {DEFAULT_RIGHT_PORT}).",
    )
    parser.add_argument(
        "--left-port",
        type=int,
        default=DEFAULT_LEFT_PORT,
        help=f"Left follower server port (default: {DEFAULT_LEFT_PORT}).",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=None,
        help="Optional gripper open limit passed through to minimum_gello.py when supported.",
    )
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=None,
        help="Optional gripper close limit passed through to minimum_gello.py when supported.",
    )
    parser.add_argument(
        "--minimum-gello-path",
        type=str,
        default=None,
        help="Explicit path to minimum_gello.py. Overrides auto-discovery.",
    )
    return parser.parse_args()


def _candidate_follower_server_paths(explicit_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("YAM_FOLLOWER_SERVER_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    env_path = os.environ.get("I2RT_MINIMUM_GELLO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "scripts" / "yam_follower_server.py")

    try:
        import i2rt  # type: ignore
    except ImportError:
        pass
    else:
        i2rt_root = Path(i2rt.__file__).resolve().parent.parent
        candidates.append(i2rt_root / "scripts" / "minimum_gello.py")

    # Match the user's known-good external follower checkout.
    candidates.append(
        Path("/home/prior/thomas/lerobot_pi05/i2rt/scripts/minimum_gello.py")
    )
    return candidates


def find_follower_server_script(explicit_path: str | None = None) -> Path:
    for candidate in _candidate_follower_server_paths(explicit_path):
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(
        str(path) for path in _candidate_follower_server_paths(explicit_path)
    )
    raise RuntimeError(
        "Could not find a follower server script. Set --minimum-gello-path, "
        "YAM_FOLLOWER_SERVER_PATH, or I2RT_MINIMUM_GELLO_PATH to a valid script.\n"
        f"Searched:\n{searched}"
    )


def script_supports_gripper_limits(script_path: Path) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    help_text = f"{result.stdout}\n{result.stderr}"
    return "--gripper_open_limit" in help_text and "--gripper_close_limit" in help_text


def port_is_open(port: int, host: str = DEFAULT_SERVER_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_port(
    port: int, host: str = DEFAULT_SERVER_HOST, timeout_s: float = PORT_WAIT_TIMEOUT_S
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if port_is_open(port, host):
            return
        time.sleep(PORT_POLL_INTERVAL_S)
    raise TimeoutError(f"Timed out waiting for follower server on {host}:{port}.")


def ensure_port_is_free(port: int, host: str = DEFAULT_SERVER_HOST) -> None:
    if port_is_open(port, host):
        raise RuntimeError(
            f"Port {host}:{port} is already in use. Stop the stale follower server first."
        )


def build_server_command(
    script_path: Path,
    can_channel: str,
    server_port: int,
    gripper_open: float | None,
    gripper_close: float | None,
    supports_gripper_limits: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--can_channel",
        can_channel,
        "--gripper",
        DEFAULT_GRIPPER,
        "--mode",
        DEFAULT_MODE,
        "--server_port",
        str(server_port),
    ]

    if gripper_open is not None or gripper_close is not None:
        if supports_gripper_limits:
            if gripper_open is not None:
                cmd.extend(["--gripper_open_limit", str(gripper_open)])
            if gripper_close is not None:
                cmd.extend(["--gripper_close_limit", str(gripper_close)])
    return cmd


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> None:
    args = parse_args()
    script_path = find_follower_server_script(args.minimum_gello_path)
    supports_gripper_limits = script_supports_gripper_limits(script_path)

    if (
        args.gripper_open is not None or args.gripper_close is not None
    ) and not supports_gripper_limits:
        print(
            "[FollowerServers] minimum_gello.py does not expose "
            "--gripper_open_limit/--gripper_close_limit; ignoring those overrides.",
            flush=True,
        )

    ensure_port_is_free(args.right_port)
    ensure_port_is_free(args.left_port)

    processes: list[subprocess.Popen[bytes]] = []

    def _shutdown(_signum: int | None = None, _frame: object | None = None) -> None:
        print("\n[FollowerServers] Stopping follower servers...", flush=True)
        terminate_processes(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[FollowerServers] Launching follower arm servers...", flush=True)
    print(f"[FollowerServers] follower server script: {script_path}", flush=True)

    server_specs = [
        ("right", args.right_can, args.right_port),
        ("left", args.left_can, args.left_port),
    ]

    for side, can_channel, port in server_specs:
        cmd = build_server_command(
            script_path=script_path,
            can_channel=can_channel,
            server_port=port,
            gripper_open=args.gripper_open,
            gripper_close=args.gripper_close,
            supports_gripper_limits=supports_gripper_limits,
        )
        print(
            f"[FollowerServers] Starting {side} follower: {' '.join(cmd)}", flush=True
        )
        process = subprocess.Popen(cmd)
        processes.append(process)

    try:
        wait_for_port(args.right_port)
        wait_for_port(args.left_port)
        print(
            f"[FollowerServers] Ready. Right: {DEFAULT_SERVER_HOST}:{args.right_port}, "
            f"Left: {DEFAULT_SERVER_HOST}:{args.left_port}",
            flush=True,
        )

        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Follower server process {process.pid} exited with code {process.returncode}."
                    )
            time.sleep(1.0)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
