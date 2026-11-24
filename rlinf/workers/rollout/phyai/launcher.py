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

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import ray.util
import requests
import yaml
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Cluster, PlacementStrategy, Worker


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
        cudart_lib_name = self._child_env.get("PHYAI_CUDART_LIB_NAME")
        if cudart_lib_name:
            self._child_env["CUDNN_FRONTEND_CUDART_LIB_NAME"] = cudart_lib_name

    def start(self) -> None:
        """Start both subprocesses and wait until the Gateway backend is healthy."""
        if self._server_proc is not None or self._gateway_proc is not None:
            raise RuntimeError("PhyAI service is already running.")

        model_name = str(self._cfg.get("model_name", "pi05"))
        startup_timeout = float(self._cfg.get("startup_timeout_s", 900))
        if startup_timeout <= 0:
            raise ValueError("rollout.phyai.startup_timeout_s must be positive.")

        self._server_port = self.acquire_free_port()
        self._gateway_grpc_port = self.acquire_free_port()
        self._gateway_http_port = self.acquire_free_port()
        self._server_config = self._prepare_server_config(self._server_port)
        self._open_logs()

        server_endpoint = f"127.0.0.1:{self._server_port}"
        server_cmd = [
            sys.executable,
            "-m",
            "phyai",
            "server",
            str(self._server_config),
        ]
        self.log_info(
            "Launching PhyAI model server: " + " ".join(server_cmd)
        )
        self._server_proc = subprocess.Popen(
            server_cmd,
            start_new_session=True,
            stdout=self._server_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self._child_env,
        )

        try:
            _wait_for_tcp(
                self._server_proc,
                "127.0.0.1",
                self._server_port,
                startup_timeout,
                "PhyAI model server",
            )

            gateway_cmd = [
                sys.executable,
                "-m",
                "phyai_gateway.server",
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
                "Launching PhyAI Gateway: " + " ".join(gateway_cmd)
            )
            self._gateway_proc = subprocess.Popen(
                gateway_cmd,
                start_new_session=True,
                stdout=self._gateway_log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=self._child_env,
            )

            advertise_host = self._cfg.get("advertise_host")
            if not isinstance(advertise_host, str) or not advertise_host.strip():
                advertise_host = ray.util.get_node_ip_address()
            self._gateway_url = (
                f"http://{advertise_host.strip()}:{self._gateway_http_port}"
            )
            _wait_for_gateway(
                self._gateway_proc,
                self._gateway_url,
                model_name,
                server_endpoint,
                startup_timeout,
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
        """Return whether the Gateway process and backend are still healthy."""
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
            return any(
                backend.get("model_name") == self._cfg.get("model_name", "pi05")
                and backend.get("endpoint") == endpoint
                and backend.get("healthy") is True
                for backend in response.json().get("backends", [])
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
        if self._server_config is not None:
            self._server_config.unlink(missing_ok=True)
            self._server_config = None

    def _prepare_server_config(self, server_port: int) -> Path:
        source = _resolve_path(self._cfg.get("server_config"), "server_config")
        checkpoint = _resolve_directory(
            self._cfg.get("checkpoint_dir"), "checkpoint_dir"
        )
        tokenizer = _resolve_directory(
            self._cfg.get("tokenizer_dir"), "tokenizer_dir"
        )
        config = yaml.safe_load(source.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"PhyAI server config must contain a mapping: {source}")
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
            path = Path(os.environ.get("REPO_PATH", os.getcwd())) / path
        path.mkdir(parents=True, exist_ok=True)
        rank = int(self._rank)
        self._server_log = open(path / f"phyai-server-{rank}.log", "w", buffering=1)
        self._gateway_log = open(path / f"phyai-gateway-{rank}.log", "w", buffering=1)


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
        service_group.stop().wait()
        service_group._close()
        raise


def _to_dict(config: DictConfig) -> dict:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise ValueError("rollout.phyai must be a mapping.")
    return value


def _resolve_path(value, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rollout.phyai.{field} is required for automatic Gateway launch.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        repo_path = Path(os.environ.get("REPO_PATH", os.getcwd()))
        path = repo_path / path
        if not path.exists():
            source_root = Path(__file__).resolve().parents[4]
            path = source_root / value
    if not path.is_file():
        raise FileNotFoundError(f"PhyAI {field} does not exist: {path}")
    return path.resolve()


def _resolve_directory(value, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rollout.phyai.{field} is required for automatic Gateway launch.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        repo_path = Path(os.environ.get("REPO_PATH", os.getcwd()))
        path = repo_path / path
    if not path.is_dir():
        raise NotADirectoryError(f"PhyAI {field} does not exist or is not a directory: {path}")
    return path.resolve()


def _wait_for_tcp(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout: float,
    service_name: str,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{service_name} exited with code {return_code} before becoming ready."
            )
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {service_name} on {host}:{port}.")


def _wait_for_gateway(
    proc: subprocess.Popen,
    gateway_url: str,
    model_name: str,
    endpoint: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            raise RuntimeError(
                f"PhyAI Gateway exited with code {return_code} before becoming ready."
            )
        try:
            response = requests.get(
                f"{gateway_url}/v1/backends",
                timeout=2,
                proxies={"http": None, "https": None},
            )
            if response.status_code == 200:
                backends = response.json().get("backends", [])
                if any(
                    backend.get("model_name") == model_name
                    and backend.get("endpoint") == endpoint
                    and backend.get("healthy") is True
                    for backend in backends
                ):
                    return
        except (requests.RequestException, ValueError) as error:
            last_error = error
        time.sleep(1)
    raise RuntimeError(
        f"Timed out waiting for healthy PhyAI Gateway backend at {gateway_url}; "
        f"last error: {last_error!r}."
    )


def _terminate_process(proc, service_name: str, log_info) -> None:
    if proc is None:
        return
    log_info(f"Stopping {service_name} pid={proc.pid}.")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)
