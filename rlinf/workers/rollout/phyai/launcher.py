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

"""Launch a local PhyAI model server and Gateway under RLinf placement."""

from __future__ import annotations

import copy
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Cluster, PlacementStrategy, Worker
from rlinf.utils.logging import get_logger

# Child bootstrap that watches its parent PID in a daemon thread and kills its
# own process group if the spawning Ray worker exits abnormally.
_WATCHDOG_BOOTSTRAP = (
    "import os, runpy, signal, sys, threading, time\n"
    "ppid, mod = int(sys.argv[1]), sys.argv[2]\n"
    "def _kill():\n"
    "    try:\n"
    "        pgrp = os.getpgrp()\n"
    "        os.killpg(pgrp, signal.SIGKILL) if pgrp == os.getpid() "
    "else os.kill(os.getpid(), signal.SIGKILL)\n"
    "    except OSError:\n"
    "        os.kill(os.getpid(), signal.SIGKILL)\n"
    "def _watch():\n"
    "    while True:\n"
    "        if os.getppid() != ppid:\n"
    "            _kill()\n"
    "        time.sleep(1.0)\n"
    "threading.Thread(target=_watch, daemon=True).start()\n"
    "sys.argv = [mod, *sys.argv[3:]]\n"
    "runpy.run_module(mod, run_name='__main__', alter_sys=True)\n"
)


def _build_watched_cmd(
    module_name: str, args: list[str], parent_pid: Optional[int] = None
) -> list[str]:
    """Build a Python command guarded by a parent-death watchdog."""
    pid = os.getpid() if parent_pid is None else parent_pid
    return [sys.executable, "-c", _WATCHDOG_BOOTSTRAP, str(pid), module_name, *args]


class PhyAIServiceWorker(Worker):
    """Own one PhyAI model server and its Gateway sidecar."""

    def __init__(self, phyai_cfg: DictConfig):
        Worker.__init__(self)
        self._cfg = _to_dict(phyai_cfg)
        self._server_proc: Optional[subprocess.Popen] = None
        self._gateway_proc: Optional[subprocess.Popen] = None
        self._server_config: Optional[Path] = None
        self._server_log = None
        self._gateway_log = None
        self._server_log_path: Optional[Path] = None
        self._gateway_log_path: Optional[Path] = None
        self._server_port: Optional[int] = None
        self._gateway_http_port: Optional[int] = None
        self._gateway_grpc_port: Optional[int] = None
        self._gateway_url: Optional[str] = None
        self._child_env = os.environ.copy()
        # RLinf injects distributed-runtime variables (WORLD_SIZE, RANK, ...)
        # into every Worker process. They must not leak into the PhyAI model
        # server or Gateway subprocess: phyai reads WORLD_SIZE to infer the
        # replica world size, so a multi-GPU placement would make the
        # single-card plugin misreport itself as N-way parallel and
        # refuse to start.
        for _var in (
            "WORLD_SIZE",
            "RANK",
            "NODE_RANK",
            "CLUSTER_NODE_RANK",
            "LOCAL_ACCELERATOR_RANK",
            "NODE_LOCAL_RANK",
            "NODE_LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_NAME",
            "WORKER_NAME",
            "CLUSTER_NAMESPACE",
            "VISIBLE_DEVICES",
            "ACCELERATOR_TYPE",
            "ACCELERATOR_MODEL",
            "ISOLATE_ACCELERATOR",
            "LOCAL_HARDWARE_RANKS",
            "NODE_GROUP_LABEL",
            "RAY_ACTOR",
            "CATCH_SYSTEM_FAILURE",
        ):
            self._child_env.pop(_var, None)

    def start(self) -> None:
        """Start both subprocesses and wait until the Gateway backend is healthy."""
        if self._server_proc is not None or self._gateway_proc is not None:
            raise RuntimeError("PhyAI service is already running.")

        model_name = str(self._cfg.get("model_name", "pi05"))
        startup_timeout = float(self._cfg.get("startup_timeout_s", 300))
        if startup_timeout <= 0:
            raise ValueError("rollout.phyai.startup_timeout_s must be positive.")

        deadline = time.monotonic() + startup_timeout
        try:
            self._server_port = self.acquire_free_port()
            self._gateway_grpc_port = self.acquire_free_port()
            self._gateway_http_port = self.acquire_free_port()
            self._server_config = self._prepare_server_config(self._server_port)
            self._open_logs()

            server_endpoint = f"127.0.0.1:{self._server_port}"
            server_args = ["server", str(self._server_config)]
            self.log_info(
                f"Launching PhyAI model server: {sys.executable} -m phyai "
                + " ".join(server_args)
            )
            self._server_proc = subprocess.Popen(
                _build_watched_cmd("phyai", server_args),
                start_new_session=True,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=self._child_env,
            )

            _wait_for_tcp(
                self._server_proc,
                "127.0.0.1",
                self._server_port,
                deadline,
                "PhyAI model server",
                log_path=self._server_log_path,
            )

            gateway_args = [
                "--gateway",
                f"127.0.0.1:{self._gateway_grpc_port}",
                "--http-host",
                str(self._cfg.get("http_host", "0.0.0.0")),
                "--http-port",
                str(self._gateway_http_port),
                "--backend",
                model_name,
                server_endpoint,
            ]
            self.log_info(
                f"Launching PhyAI Gateway: {sys.executable} -m phyai_gateway.server "
                + " ".join(gateway_args)
            )
            self._gateway_proc = subprocess.Popen(
                _build_watched_cmd("phyai_gateway.server", gateway_args),
                start_new_session=True,
                stdout=self._gateway_log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=self._child_env,
            )

            advertise_host = self._cfg.get("advertise_host")
            if not isinstance(advertise_host, str) or not advertise_host.strip():
                import ray.util

                advertise_host = ray.util.get_node_ip_address()
            self._gateway_url = (
                f"http://{advertise_host.strip()}:{self._gateway_http_port}"
            )
            _wait_for_gateway(
                self._gateway_proc,
                self._server_proc,
                self._gateway_url,
                model_name,
                server_endpoint,
                deadline,
                gateway_log_path=self._gateway_log_path,
                server_log_path=self._server_log_path,
            )
            self.log_info(f"PhyAI Gateway is ready at {self._gateway_url}")
        except BaseException:
            self.stop()
            raise

    def get_gateway_url(self) -> str:
        """Return the reachable Gateway URL after :meth:`start`."""
        if self._gateway_url is None:
            raise RuntimeError("PhyAI service has not been started.")
        return self._gateway_url

    def is_healthy(self) -> bool:
        """Return whether the model server, Gateway, and backend are healthy."""
        if self._server_proc is None or self._server_proc.poll() is not None:
            return False
        if self._gateway_proc is None or self._gateway_proc.poll() is not None:
            return False
        if self._gateway_url is None or self._server_port is None:
            return False
        try:
            response = requests.get(
                f"{self._gateway_url}/v1/backends",
                timeout=2,
                proxies={"http": None, "https": None},
            )
            if response.status_code != 200:
                return False
            endpoint = f"127.0.0.1:{self._server_port}"
            return _backend_is_healthy(
                response.json().get("backends", []),
                self._cfg.get("model_name", "pi05"),
                endpoint,
            )
        except (requests.RequestException, ValueError):
            return False

    def stop(self) -> None:
        """Stop the Gateway and model-server process groups."""
        _terminate_process(self._gateway_proc, "PhyAI Gateway", self.log_info)
        _terminate_process(self._server_proc, "PhyAI model server", self.log_info)
        self._gateway_proc = None
        self._server_proc = None
        self._gateway_url = None
        for stream_name in ("_gateway_log", "_server_log"):
            stream = getattr(self, stream_name)
            if stream is not None:
                stream.close()
                setattr(self, stream_name, None)
        self._server_log_path = None
        self._gateway_log_path = None
        if self._server_config is not None:
            self._server_config.unlink(missing_ok=True)
            self._server_config = None

    def _prepare_server_config(self, server_port: int) -> Path:
        raw_server = self._cfg.get("server")
        if not isinstance(raw_server, dict) or not raw_server:
            raise ValueError(
                "rollout.phyai.server must be a nonempty mapping for automatic "
                "Gateway launch."
            )
        checkpoint = _resolve_directory(
            self._cfg.get("checkpoint_dir"), "checkpoint_dir"
        )
        tokenizer = _resolve_directory(self._cfg.get("tokenizer_dir"), "tokenizer_dir")
        config = copy.deepcopy(raw_server)
        server = config.setdefault("server", {})
        if not isinstance(server, dict):
            raise ValueError("PhyAI server config field 'server' must be a mapping.")
        server["host"] = "127.0.0.1"
        server["port"] = server_port
        server["model_name"] = str(self._cfg.get("model_name", "pi05"))
        plugin_args = config.setdefault("plugin_args", {})
        adapter_args = config.setdefault("adapter", {}).setdefault("args", {})
        if not isinstance(plugin_args, dict) or not isinstance(adapter_args, dict):
            raise ValueError("PhyAI server config has invalid plugin or adapter args.")
        plugin_args["checkpoint_dir"] = str(checkpoint)
        if self._cfg.get("max_batch_size") is not None:
            plugin_args["max_batch_size"] = int(self._cfg["max_batch_size"])
        adapter_args["tokenizer_dir"] = str(tokenizer)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="rlinf-phyai-", delete=False
        ) as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
            return Path(stream.name)

    def _open_logs(self) -> None:
        log_dir = self._cfg.get("log_dir")
        if not isinstance(log_dir, str) or not log_dir.strip():
            return
        path = Path(log_dir).expanduser()
        if not path.is_absolute():
            base = os.environ.get("REPO_PATH")
            path = (Path(base) if base else _repo_root()) / path
        path.mkdir(parents=True, exist_ok=True)
        rank = int(self._rank)
        self._server_log_path = path / f"phyai-server-{rank}.log"
        self._gateway_log_path = path / f"phyai-gateway-{rank}.log"
        self._server_log = open(self._server_log_path, "w", buffering=1)
        self._gateway_log = open(self._gateway_log_path, "w", buffering=1)


def launch_phyai_gateway_and_server(
    config: DictConfig,
    cluster: Cluster,
    placement_strategy: PlacementStrategy,
    phyai_cfg: Optional[DictConfig] = None,
):
    """Launch PhyAI service workers with the supplied RLinf placement."""
    if placement_strategy is None:
        raise ValueError("PhyAI Gateway mode requires a phyai_server placement.")
    if phyai_cfg is None:
        phyai_cfg = config.rollout.phyai
    group_name = str(phyai_cfg.get("service_group_name", "PhyAIServiceGroup"))
    service_group = PhyAIServiceWorker.create_group(phyai_cfg=phyai_cfg).launch(
        cluster,
        name=group_name,
        placement_strategy=placement_strategy,
    )
    try:
        service_group.start().wait()
        gateway_urls = list(service_group.get_gateway_url().wait())
        if not gateway_urls:
            raise RuntimeError("PhyAI service workers returned no Gateway URL.")
        return gateway_urls, service_group
    except BaseException:
        try:
            service_group.stop().wait()
        except Exception as error:
            get_logger().warning(
                f"Failed to stop PhyAI service group after launch error: {error}"
            )
        raise


def _to_dict(config: DictConfig) -> dict:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise ValueError("rollout.phyai must be a mapping.")
    return value


def _repo_root() -> Path:
    # rlinf/workers/rollout/phyai/launcher.py -> repo root is 4 levels up.
    return Path(__file__).resolve().parents[4]


def _resolve_directory(value, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"rollout.phyai.{field} is required for automatic Gateway launch."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = os.environ.get("REPO_PATH")
        path = (Path(base) if base else _repo_root()) / path
    if not path.is_dir():
        raise NotADirectoryError(
            f"PhyAI {field} does not exist or is not a directory: {path}"
        )
    return path.resolve()


def _format_log_hint(log_path: Optional[Path]) -> str:
    return f" See {log_path} for details." if log_path is not None else ""


def _wait_for_tcp(
    proc: subprocess.Popen,
    host: str,
    port: int,
    deadline: float,
    service_name: str,
    log_path: Optional[Path] = None,
) -> None:
    while time.monotonic() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{service_name} exited with code {return_code} before becoming ready."
                f"{_format_log_hint(log_path)}"
            )
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(
        f"Timed out waiting for {service_name} on {host}:{port}."
        f"{_format_log_hint(log_path)}"
    )


def _backend_is_healthy(backends: list, model_name: str, endpoint: str) -> bool:
    return any(
        backend.get("model_name") == model_name
        and backend.get("endpoint") == endpoint
        and backend.get("healthy") is True
        for backend in backends
    )


def _wait_for_gateway(
    gateway_proc: subprocess.Popen,
    server_proc: subprocess.Popen,
    gateway_url: str,
    model_name: str,
    endpoint: str,
    deadline: float,
    gateway_log_path: Optional[Path] = None,
    server_log_path: Optional[Path] = None,
) -> None:
    last_error = None
    while time.monotonic() < deadline:
        gateway_code = gateway_proc.poll()
        if gateway_code is not None:
            raise RuntimeError(
                f"PhyAI Gateway exited with code {gateway_code} before becoming ready."
                f"{_format_log_hint(gateway_log_path)}"
            )
        server_code = server_proc.poll()
        if server_code is not None:
            raise RuntimeError(
                "PhyAI model server exited with code "
                f"{server_code} while waiting for Gateway."
                f"{_format_log_hint(server_log_path)}"
            )
        try:
            response = requests.get(
                f"{gateway_url}/v1/backends",
                timeout=2,
                proxies={"http": None, "https": None},
            )
            if response.status_code == 200:
                if _backend_is_healthy(
                    response.json().get("backends", []), model_name, endpoint
                ):
                    return
        except (requests.RequestException, ValueError) as error:
            last_error = error
        time.sleep(1)
    raise RuntimeError(
        f"Timed out waiting for healthy PhyAI Gateway backend at {gateway_url}; "
        f"last error: {last_error!r}.{_format_log_hint(gateway_log_path)}"
    )


def _terminate_process(proc, service_name: str, log_info) -> None:
    if proc is None:
        return
    log_info(f"Stopping {service_name} pid={proc.pid}.")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
        proc.wait(timeout=5)
