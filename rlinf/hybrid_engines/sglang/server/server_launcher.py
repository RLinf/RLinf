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


from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from dataclasses import asdict
from typing import Optional

import ray.util
import requests
from omegaconf import DictConfig, OmegaConf
from sglang.srt.server_args import ServerArgs

from rlinf.scheduler import Worker


def _run_sglang_server(server_args_dict: dict, ready_pipe) -> None:
    """Child-process entrypoint: launches a single sglang HTTP server.

    Runs in a *spawned* subprocess so the parent's Ray actor isn't blocked
    by sglang's uvicorn loop. ``ready_pipe`` is a one-shot
    ``multiprocessing.Pipe`` end the child writes to once initialization
    either completes or throws (mirrors sglang's ``pipe_finish_writer``
    contract).
    """
    # Put this process in its own group so SIGTERM to the parent can
    # forward via os.killpg without killing the Ray actor itself.
    try:
        os.setpgrp()
    except OSError:
        pass

    from sglang.srt.entrypoints.http_server import launch_server

    server_args = ServerArgs(**server_args_dict)
    try:
        launch_server(server_args, pipe_finish_writer=ready_pipe)
    except Exception as e:  # pragma: no cover — surface failures to parent
        try:
            ready_pipe.send(repr(e))
        except Exception:
            pass
        raise


def _wait_for_http_health(host: str, port: int, timeout: float = 300.0) -> None:
    """Block until ``GET http://host:port/health`` returns 200, or raise."""
    deadline = time.perf_counter() + timeout
    url = f"http://{host}:{port}/health"
    last_err: Optional[BaseException] = None
    while time.perf_counter() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(
        f"sglang server at {url} did not become healthy within {timeout:.0f}s "
        f"(last error: {last_err!r})."
    )


class SGLangServerWorker(Worker):
    """Worker that owns one sglang HTTP server process.

    Args:
        config: Full RLinf ``DictConfig``. Kept for parity with other
            workers / future use; the sglang server itself is configured
            entirely from ``sglang_cfg``.
        sglang_cfg: The sub-config block whose keys are forwarded verbatim
            as ``ServerArgs(**)`` kwargs — except ``host`` / ``port`` /
            ``dist_init_addr``, which are filled in at runtime here.
            Typically ``config.rollout.server`` when used inside a rollout,
            but any compatible block works (the server isn't tied to the
            rollout pipeline).
        bind_host: Optional explicit bind host. If ``None``, we bind to
            ``0.0.0.0`` so the router worker on another node can reach us.
        advertise_host: Optional explicit advertise host (the URL we
            hand to the router). If ``None``, we fall back to the Ray
            actor's node IP via ``ray.util.get_node_ip_address()``.
    """

    def __init__(
        self,
        config: DictConfig,
        sglang_cfg: DictConfig,
        bind_host: str = "0.0.0.0",
        advertise_host: Optional[str] = None,
    ):
        Worker.__init__(self)
        self._cfg = config
        self._sglang_cfg = sglang_cfg
        self._bind_host = bind_host
        self._advertise_host = advertise_host

        self._server_proc: Optional[mp.Process] = None
        self._server_port: Optional[int] = None
        self._ready_pipe = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init_server(self) -> str:
        """Spawn the sglang HTTP server subprocess and wait for /health.

        Returns:
            The URL the router should register (``http://host:port``).
        """
        assert self._server_proc is None, "sglang server already initialized."

        # Acquire two distinct free ports: one for HTTP, one for the
        # internal torch.distributed bootstrap. ``acquire_free_port``
        # uses the worker's PortLock so neither port collides with any
        # other worker on this node.
        http_port = self.acquire_free_port()
        dist_port = self.acquire_free_port()

        sglang_kwargs = OmegaConf.to_container(self._sglang_cfg, resolve=True)
        sglang_kwargs["host"] = self._bind_host
        sglang_kwargs["port"] = http_port
        sglang_kwargs["dist_init_addr"] = f"127.0.0.1:{dist_port}"
        server_args = ServerArgs(**sglang_kwargs)

        self.log_info(
            f"Launching sglang server: tp_size={server_args.tp_size}, "
            f"http=:{http_port}, dist_init={server_args.dist_init_addr}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )

        ctx = mp.get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe(duplex=False)
        self._ready_pipe = parent_pipe
        proc = ctx.Process(
            target=_run_sglang_server,
            args=(asdict(server_args), child_pipe),
            daemon=False,
        )
        proc.start()
        # We hand the write end to the child; close ours so the read end
        # signals EOF if the child dies before sending anything.
        child_pipe.close()

        self._server_proc = proc
        self._server_port = http_port
        # Resolve the host we want to advertise to the router *before*
        # we block on /health so a slow server doesn't gate URL lookup.
        if self._advertise_host is None:
            self._advertise_host = ray.util.get_node_ip_address()

        _wait_for_http_health(self._advertise_host, http_port)
        self.log_info(f"sglang server ready at {self.get_server_url()}")
        return self.get_server_url()

    def get_server_url(self) -> str:
        """Return the advertised ``http://host:port`` URL for this server."""
        assert self._server_port is not None, "init_server() has not been called."
        host = self._advertise_host or "0.0.0.0"
        return f"http://{host}:{self._server_port}"

    def is_healthy(self) -> bool:
        if self._server_proc is None or not self._server_proc.is_alive():
            return False
        try:
            url = f"http://{self._advertise_host}:{self._server_port}/health"
            return requests.get(url, timeout=2).status_code == 200
        except requests.exceptions.RequestException:
            return False

    def shutdown(self) -> None:
        """Terminate the sglang server subprocess (and its process group)."""
        proc = self._server_proc
        if proc is None:
            return
        self.log_info(f"Shutting down sglang server pid={proc.pid}.")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():  # pragma: no cover — best-effort kill
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.join(timeout=5)
        self._server_proc = None
        self._server_port = None
