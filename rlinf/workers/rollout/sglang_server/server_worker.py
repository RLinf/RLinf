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
from typing import Callable, Optional

import ray.util
import requests
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Worker
from rlinf.utils.http_client import no_proxy_env


def _ensure_no_proxy_for_localhost() -> None:
    """Make sure sglang's intra-node IPC never tunnels through a proxy."""
    local = "127.0.0.1,localhost,::1"
    current = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    if not any(h in current for h in ("127.0.0.1", "localhost")):
        os.environ["NO_PROXY"] = (
            f"{current},{local}".strip(",") if current else local
        )


def _run_sglang_server(
    server_type: str,
    server_args_kwargs: dict,
    dist_port: int,
    launch_router: bool,
) -> None:
    """Child-process entrypoint: launch one sglang HTTP server.

    Args:
        server_type: ``"srt"`` (language model) or ``"embodied"`` (VLA /
            diffusion). Selects which sglang dispatch entrypoint to call:
            ``"srt"``      -> :func:`sglang.launch_server.run_server`
            (what ``sglang serve`` calls for an LLM);
            ``"embodied"`` -> :func:`sglang.multimodal_gen.runtime.launch_server.dispatch_launch`
            (what ``sglang serve`` calls for a diffusion model). Any other
            value raises (no auto-detection — the caller must declare the
            family explicitly).
        server_args_kwargs: The server config block, with ``host``/``port``
            already filled in by the parent. The type-specific distributed
            bootstrap port (``dist_init_addr`` for SRT, ``master_port`` for
            embodied) is injected here, since the two ServerArgs don't share
            that field.
        dist_port: Free port for the internal torch.distributed bootstrap.
        launch_router: Whether the caller is also bringing up a sglang router
            (read from ``rollout.sglang.launch_router``). The router can't
            forward the multimodal action endpoint, so ``embodied`` + router
            is rejected here.
    """
    # Own process group so SIGTERM via os.killpg(pid) reaches the sglang serve
    # tree, not the Ray actor.
    try:
        os.setpgrp()
    except OSError:
        pass

    _ensure_no_proxy_for_localhost()
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    if server_type not in ("srt", "embodied"):
        raise ValueError(
            f"Unsupported server_type {server_type!r}; "
            "expected 'srt' (language model) or 'embodied' (VLA/diffusion)."
        )

    if server_type == "embodied" and launch_router:
        raise RuntimeError(
            "launch_router is not supported for the multimodal (diffusion/VLA) "
            "sglang server: the sglang router only forwards fixed endpoints "
            "(/generate, /v1/chat/completions, ...), not the dreamzero action "
            "endpoint /v1/actions/generations. Set rollout.sglang.launch_router: "
            "false (rollout workers hit their rank-assigned server URL directly)."
        )

    with no_proxy_env():
        if server_type == "embodied":
            from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
            from sglang.multimodal_gen.runtime.server_args import (
                ServerArgs,
                set_global_server_args,
            )

            server_args_kwargs["master_port"] = dist_port
            server_args = ServerArgs.from_kwargs(**server_args_kwargs)
            set_global_server_args(server_args)
            dispatch_launch(server_args)
        else:  # "srt"
            from sglang.launch_server import run_server
            from sglang.srt.server_args import ServerArgs

            server_args_kwargs["dist_init_addr"] = f"127.0.0.1:{dist_port}"
            server_args = ServerArgs(**server_args_kwargs)
            run_server(server_args)


def _wait_for_http_health(
    host: str,
    port: int,
    timeout: float = 300.0,
    is_alive: Optional[Callable[[], bool]] = None,
) -> None:
    """Block until ``GET http://host:port/health`` returns 200, or raise."""
    deadline = time.perf_counter() + timeout
    url = f"http://{host}:{port}/health"
    last_err: Optional[BaseException] = None
    while time.perf_counter() < deadline:
        if is_alive is not None and not is_alive():
            raise RuntimeError(
                f"sglang server subprocess exited before /health went 200 "
                f"({url}); see the worker log for the child's error."
            )
        try:
            resp = requests.get(url, timeout=5, proxies={"http": None, "https": None})
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
        config: Full RLinf ``DictConfig``. The sglang server itself is
            configured entirely from ``sglang_cfg``; ``config`` is read for
            the optional ``rollout.sglang.spawn_timeout`` /
            ``launch_router`` knobs.
        sglang_cfg: The sub-config block whose keys are forwarded verbatim
            as ``ServerArgs`` kwargs — except ``host`` / ``port`` (filled in
            at runtime here) and the distributed-bootstrap port
            (``dist_init_addr`` for SRT, ``master_port`` for embodied, filled
            in by the child since the two ServerArgs don't share that field).
            Typically ``config.rollout.server`` when used inside a rollout,
            but any compatible block works (the server isn't tied to the
            rollout pipeline).
        bind_host: Optional explicit bind host. If ``None``, we bind to
            ``0.0.0.0`` so the router worker on another node can reach us.
        advertise_host: Optional explicit advertise host (the URL we
            hand to the router). If ``None``, we fall back to the Ray
            actor's node IP via ``ray.util.get_node_ip_address()``.
        server_type: ``"srt"`` (language model) or ``"embodied"``
            (VLA/diffusion). Required — any other value raises. Defaults to
            ``"srt"`` so a plain language-model rollout need not set it.
    """

    def __init__(
        self,
        config: DictConfig,
        sglang_cfg: DictConfig,
        bind_host: str = "0.0.0.0",
        advertise_host: Optional[str] = None,
        server_type: str = "srt",
    ):
        Worker.__init__(self)
        if server_type not in ("srt", "embodied"):
            raise ValueError(
                f"Unsupported server_type {server_type!r}; "
                "expected 'srt' (language model) or 'embodied' (VLA/diffusion)."
            )
        self._cfg = config
        self._sglang_cfg = sglang_cfg
        self._bind_host = bind_host
        self._advertise_host = advertise_host
        self._server_type = server_type

        self._server_proc: Optional[mp.Process] = None
        self._server_port: Optional[int] = None

    def init_server(self) -> None:
        """Spawn the sglang HTTP server subprocess and wait for /health.

        On failure the subprocess is torn down via ``shutdown`` before
        ``RuntimeError`` is re-raised, so the caller can retry or fail fast
        without leaking a zombie sglang process.

        Raises:
            RuntimeError: if the server fails to become healthy.
        """
        assert self._server_proc is None, "sglang server already initialized."

        # Acquire two distinct free ports: one for HTTP, one for the
        # internal torch.distributed bootstrap. ``acquire_free_port``
        # uses the worker's PortLock so neither port collides with any
        # other worker on this node.
        http_port = self.acquire_free_port()
        dist_port = self.acquire_free_port()

        server_kwargs = OmegaConf.to_container(self._sglang_cfg, resolve=True) or {}
        server_kwargs["host"] = self._bind_host
        server_kwargs["port"] = http_port

        launch_router = bool(self._cfg.rollout.sglang.get("launch_router", False))

        tp_size = server_kwargs.get("tp_size") or server_kwargs.get("tp-size")
        self.log_info(
            f"Launching sglang server (server_type={self._server_type}): "
            f"tp_size={tp_size}, http=:{http_port}, dist_port={dist_port}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )

        # Pin the spawned server to torch's bundled CUDA runtime. With
        # `enable_memory_saver`, sglang LD_PRELOADs a torch_memory_saver hook
        # that has no RUNPATH, so it would otherwise resolve libcudart via
        # ld.so.cache to the (older) system CUDA and crash torch with
        # "undefined symbol: cudaGetDriverEntryPointByVersion". LD_LIBRARY_PATH
        # is searched before ld.so.cache; the spawned child inherits it.
        try:
            import nvidia.cuda_runtime

            _cudart_lib = os.path.join(
                os.path.dirname(nvidia.cuda_runtime.__file__), "lib"
            )
            existing_paths = [
                path
                for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                if path and path != _cudart_lib
            ]
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                [_cudart_lib, *existing_paths]
            )
        except ImportError:
            # torch built against system/conda CUDA (no bundled runtime).
            pass

        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_run_sglang_server,
            args=(self._server_type, server_kwargs, dist_port, launch_router),
            daemon=False,
        )
        proc.start()

        self._server_proc = proc
        self._server_port = http_port
        # Resolve the host we want to advertise to the router *before*
        # we block on /health so a slow server doesn't gate URL lookup.
        if self._advertise_host is None:
            self._advertise_host = ray.util.get_node_ip_address()

        # multimodal_gen (VLA) warmup is heavier than an LLM, so it gets a
        # longer default. Overridable via rollout.sglang.spawn_timeout.
        default_spawn_timeout = 1800.0 if self._server_type == "embodied" else 300.0
        spawn_timeout = float(
            self._cfg.rollout.sglang.get("spawn_timeout", default_spawn_timeout)
        )
        try:
            _wait_for_http_health(
                self._advertise_host,
                http_port,
                spawn_timeout,
                is_alive=lambda: self._server_proc.is_alive(),
            )
        except RuntimeError as e:
            self.log_error(f"sglang server failed to become healthy: {e!r}")
            self.shutdown()
            raise
        self.log_info(f"sglang server ready at {self.get_server_url()}")

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
            return (
                requests.get(
                    url, timeout=2, proxies={"http": None, "https": None}
                ).status_code
                == 200
            )
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
