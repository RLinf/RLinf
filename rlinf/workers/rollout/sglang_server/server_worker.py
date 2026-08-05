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

import inspect
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
        os.environ["NO_PROXY"] = f"{current},{local}".strip(",") if current else local


def _run_sglang_server(
    server_type: str,
    server_args_kwargs: dict,
    dist_port: int,
    ready_pipe,
) -> None:
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

    _ensure_no_proxy_for_localhost()
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    # Strip proxy env vars so sglang's internal HTTP calls (e.g. the
    # tokenizer-manager / scheduler IPC that /get_server_info touches)
    # don't tunnel through a user-configured proxy — otherwise the router's
    # discover_metadata step hangs and worker registration fails.
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
        else:
            from sglang.srt.entrypoints.http_server import launch_server
            from sglang.srt.server_args import ServerArgs

            server_args_kwargs["dist_init_addr"] = f"127.0.0.1:{dist_port}"
            server_args = ServerArgs(**server_args_kwargs)
            # sglang dropped pipe_finish_writer after 0.5.4; readiness is established by
            # polling /health either way, so the pipe is only used to surface exceptions.
            launch_kwargs = {}
            if "pipe_finish_writer" in inspect.signature(launch_server).parameters:
                launch_kwargs["pipe_finish_writer"] = ready_pipe
            try:
                launch_server(server_args, **launch_kwargs)
            except Exception as e:  # pragma: no cover — surface to parent
                try:
                    ready_pipe.send(repr(e))
                except Exception:
                    pass
                raise


def _wait_for_http_health(
    host: str,
    port: int,
    timeout: float = 900.0,
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


# sglang derives its gRPC port as ``port + SGLANG_GRPC_PORT_OFFSET`` and rejects
# the result above 65535, so the HTTP port has to leave room for it.
SGLANG_GRPC_PORT_OFFSET = 10000
MAX_SGLANG_HTTP_PORT = 65535 - SGLANG_GRPC_PORT_OFFSET


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
        self._ready_pipe = None

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
        http_port = self.acquire_free_port(max_port_num=MAX_SGLANG_HTTP_PORT)
        dist_port = self.acquire_free_port()

        server_kwargs = OmegaConf.to_container(self._sglang_cfg, resolve=True) or {}
        server_kwargs["host"] = self._bind_host
        server_kwargs["port"] = http_port

        self.log_info(
            f"Launching sglang server (server_type={self._server_type}): "
            f"http=:{http_port}, dist_port={dist_port}, "
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
        parent_pipe, child_pipe = ctx.Pipe(duplex=False)
        self._ready_pipe = parent_pipe
        proc = ctx.Process(
            target=_run_sglang_server,
            args=(self._server_type, server_kwargs, dist_port, child_pipe),
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

        try:
            _wait_for_http_health(
                host=self._advertise_host,
                port=http_port,
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
