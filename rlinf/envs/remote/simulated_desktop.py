# Copyright 2026 Shirui Chen
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

"""Helpers for simulating the robot desktop input path via a local RobotServer."""

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

_LOCAL_REMOTE_HOSTS = {"127.0.0.1", "::1", "[::1]", "localhost"}


@dataclass(frozen=True)
class SimulatedDesktopServerSpec:
    """Configuration for a locally spawned dummy RobotServer."""

    host: str
    port: int
    env_config_path: str
    dummy: bool
    startup_timeout: float


def parse_host_port(server_url: str) -> tuple[str, int]:
    """Parse a gRPC host:port string used by RemoteEnv."""
    parts = str(server_url).rsplit(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Expected remote_server_url in '<host>:<port>' form, got: {server_url!r}"
        )
    return parts[0], int(parts[1])


def _iter_active_remote_env_cfgs(cfg: DictConfig) -> list[tuple[str, DictConfig]]:
    """Return active env configs that use RemoteEnv for this run."""
    remote_env_cfgs: list[tuple[str, DictConfig]] = []

    if not bool(getattr(cfg.runner, "only_eval", False)) and (
        str(cfg.env.train.get("env_type", "")) == "remote"
    ):
        remote_env_cfgs.append(("train", cfg.env.train))

    enable_eval = bool(getattr(cfg.runner, "only_eval", False)) or (
        int(getattr(cfg.runner, "val_check_interval", -1)) > 0
    )
    if enable_eval and str(cfg.env.eval.get("env_type", "")) == "remote":
        remote_env_cfgs.append(("eval", cfg.env.eval))

    return remote_env_cfgs


def build_simulated_desktop_server_spec(
    cfg: DictConfig,
    default_env_config_path: str | os.PathLike | None = None,
) -> SimulatedDesktopServerSpec | None:
    """Resolve the local dummy RobotServer config from Hydra settings."""
    sim_cfg = cfg.env.get("remote_desktop_simulation", None)
    if sim_cfg is None or not bool(sim_cfg.get("enabled", False)):
        return None

    active_remote_env_cfgs = _iter_active_remote_env_cfgs(cfg)
    if not active_remote_env_cfgs:
        raise RuntimeError(
            "env.remote_desktop_simulation is enabled, but no active env uses "
            "env_type: remote."
        )

    remote_urls = {
        str(env_cfg.get("remote_server_url", "localhost:50051"))
        for _, env_cfg in active_remote_env_cfgs
    }
    if len(remote_urls) != 1:
        raise RuntimeError(
            "env.remote_desktop_simulation requires all active RemoteEnv configs "
            "to share the same remote_server_url."
        )

    server_url = next(iter(remote_urls))
    host, port = parse_host_port(server_url)
    if host not in _LOCAL_REMOTE_HOSTS:
        raise RuntimeError(
            "env.remote_desktop_simulation only supports local RemoteEnv URLs. "
            f"Expected localhost/127.0.0.1/::1, got: {server_url}"
        )

    env_config_path = sim_cfg.get("env_config_path", None)
    if env_config_path is None:
        env_config_path = default_env_config_path or (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "embodiment"
            / "config"
            / "env"
            / "yam.yaml"
        )
    env_config_path = os.path.abspath(os.path.expanduser(str(env_config_path)))

    return SimulatedDesktopServerSpec(
        host=host,
        port=port,
        env_config_path=env_config_path,
        dummy=bool(sim_cfg.get("dummy", True)),
        startup_timeout=float(sim_cfg.get("startup_timeout", 30.0)),
    )


def can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True when a TCP listener is accepting connections."""
    normalized_host = "::1" if host in {"::1", "[::1]"} else host
    family = socket.AF_INET6 if ":" in normalized_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((normalized_host, port)) == 0


def stop_process(proc: subprocess.Popen | None) -> None:
    """Terminate a subprocess if it is still alive."""
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def launch_simulated_desktop_server(
    cfg: DictConfig,
    default_env_config_path: str | os.PathLike | None = None,
) -> subprocess.Popen | None:
    """Launch a local dummy RobotServer when configured."""
    spec = build_simulated_desktop_server_spec(cfg, default_env_config_path)
    if spec is None:
        return None

    if not os.path.exists(spec.env_config_path):
        raise FileNotFoundError(
            "env.remote_desktop_simulation.env_config_path does not exist: "
            f"{spec.env_config_path}"
        )

    if can_connect(spec.host, spec.port):
        raise RuntimeError(
            "env.remote_desktop_simulation cannot start because "
            f"{spec.host}:{spec.port} is already in use."
        )

    cmd = [
        sys.executable,
        "-m",
        "rlinf.envs.remote.robot_server",
        "--config-path",
        spec.env_config_path,
        "--port",
        str(spec.port),
    ]
    if spec.dummy:
        cmd.append("--dummy")

    print(
        "[train_embodied_agent_staged] Starting simulated remote desktop "
        f"RobotServer at {spec.host}:{spec.port} "
        f"(dummy={spec.dummy}, config={spec.env_config_path})"
    )
    proc = subprocess.Popen(cmd)

    deadline = time.monotonic() + spec.startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "Simulated RobotServer exited before becoming ready. "
                f"Exit code: {proc.returncode}"
            )
        if can_connect(spec.host, spec.port):
            return proc
        time.sleep(0.2)

    stop_process(proc)
    raise TimeoutError(
        "Timed out waiting for the simulated RobotServer to accept "
        f"connections on {spec.host}:{spec.port}."
    )
