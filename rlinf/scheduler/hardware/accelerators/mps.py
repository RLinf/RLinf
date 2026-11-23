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

"""Run-local NVIDIA MPS lifecycle and pre-CUDA worker selection."""

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rlinf.utils.logging import get_logger

SESSION_ENV = "RLINF_MPS_SESSION"


def _control(environment: dict[str, str], command: str) -> str:
    """Serialize daemon commands, with a timeout for each control subprocess."""
    pipes = Path(environment["CUDA_MPS_PIPE_DIRECTORY"])
    with (pipes / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            text=True,
            capture_output=True,
            env=environment,
            check=True,
            timeout=60 if command == "quit" else 30,
        ).stdout.strip()


def _pids(output: str) -> set[int]:
    """Parse the PID list returned by an MPS control command."""
    return {int(line) for line in output.splitlines() if line.strip().isdigit()}


def wait_for_clients(
    environment: dict[str, str], expected: set[int], timeout_s: float = 30
) -> None:
    """Wait for existing CUDA contexts to attach until all expected clients appear."""
    deadline = time.monotonic() + timeout_s
    missing = expected.copy()
    while missing:
        servers = _pids(_control(environment, "get_server_list"))
        for server in servers:
            missing -= _pids(_control(environment, f"get_client_list {server}"))
        if not missing:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"CUDA clients {sorted(missing)} did not attach to MPS")
        time.sleep(min(0.1, remaining))


def _inventory() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    devices = {}
    for line in result.stdout.splitlines():
        index, uuid, capability = (field.strip() for field in line.split(","))
        if float(capability) < 7:
            raise ValueError("Managed MPS requires Volta or newer NVIDIA GPUs")
        devices[index] = uuid
    if not devices:
        raise RuntimeError("No NVIDIA GPUs available for MPS")
    # RLinf places workers by physical index independently of the shell's mask.
    if "MIG-" in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise ValueError("Managed MPS does not support MIG devices")
    return devices


@contextmanager
def mps_daemon(log_dir: Path) -> Iterator[tuple[dict[str, str], dict[str, str]]]:
    """Own a private daemon without changing os.environ.

    Args:
        log_dir: Directory for daemon logs.

    Yields:
        Client environment and the physical-index to GPU UUID mapping.
    """
    devices = _inventory()
    directory = Path(log_dir).resolve() / "mps"
    directory.mkdir(parents=True, exist_ok=True)
    pipe_root = Path(tempfile.mkdtemp(prefix="rlinf-mps-"))
    pipes = pipe_root / "pipes"
    pipes.mkdir()
    environment = disabled_mps_environment(pipe_root)
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(devices.values()),
        CUDA_MPS_PIPE_DIRECTORY=str(pipes),
        CUDA_MPS_LOG_DIRECTORY=str(directory),
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100",
    )
    started = False
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        started = True
        _control(environment, f"start_server -uid {os.getuid()}")
        yield environment, devices
    finally:
        if started:
            try:
                _control(environment, "quit")
            except subprocess.TimeoutExpired:
                get_logger().warning(
                    "MPS daemon shutdown timed out; it will exit when remaining "
                    "clients disconnect"
                )
        shutil.rmtree(pipe_root, ignore_errors=True)


@contextmanager
def mps_session(
    cfg, log_dir: Path, *, caps: dict[str, int] | None = None
) -> Iterator[dict[str, str]]:
    """Yield a child environment with worker quotas; absent caps disable MPS.

    The driver and unselected groups inherit a nonexistent pipe directory.
    Selected workers connect to the owned daemon before CUDA initializes.
    """
    if not caps:
        yield disabled_mps_environment(log_dir)
        return
    with mps_daemon(log_dir) as (environment, devices):
        session = {
            "groups": {cfg[side]["group_name"]: cap for side, cap in caps.items()},
            "devices": devices,
            "pipe_directory": environment["CUDA_MPS_PIPE_DIRECTORY"],
            "log_directory": environment["CUDA_MPS_LOG_DIRECTORY"],
        }
        child_environment = disabled_mps_environment(log_dir)
        child_environment[SESSION_ENV] = json.dumps(session)
        yield child_environment


def disabled_mps_environment(log_dir: Path) -> dict[str, str]:
    """Isolate native trials and fallback training from inherited MPS daemons."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CUDA_MPS_") and key != SESSION_ENV
    }
    environment.update(
        CUDA_MPS_PIPE_DIRECTORY=str(log_dir.resolve() / "mps-disabled"),
        RAY_ADDRESS="local",
    )
    return environment


def worker_environment(environment: dict[str, str]) -> dict[str, str]:
    """Return overrides applied by Cluster.allocate before the worker imports CUDA."""
    session = json.loads(environment[SESSION_ENV])
    group = environment["GROUP_NAME"]
    if group not in session["groups"]:
        return {}
    visible = environment["CUDA_VISIBLE_DEVICES"].split(",")
    return {
        "CUDA_MPS_PIPE_DIRECTORY": session["pipe_directory"],
        "CUDA_VISIBLE_DEVICES": ",".join(
            session["devices"][index.strip()] for index in visible
        ),
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(session["groups"][group]),
        "CUDA_MPS_LOG_DIRECTORY": session["log_directory"],
    }


def visible_device_indices(visible: str) -> list[int]:
    """Resolve worker visibility using the managed session's physical GPU IDs.

    Args:
        visible: Comma-separated GPU UUIDs or physical indices, in CUDA order.

    Returns:
        Physical indices in the same order, without querying the driver.
    """
    devices = json.loads(os.environ[SESSION_ENV])["devices"]
    indices = {uuid: index for index, uuid in devices.items()}
    try:
        return [
            int(indices.get(item.strip(), item.strip())) for item in visible.split(",")
        ]
    except ValueError as error:
        raise ValueError(
            f"Invalid MPS visible devices: {visible}; expected physical indices "
            "or GPU UUIDs from the managed session"
        ) from error
