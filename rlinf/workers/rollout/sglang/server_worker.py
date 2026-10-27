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

from typing import Optional

from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Worker

from .process import SGLangServerProcess


class SGLangServerWorker(Worker):
    """Worker that owns one sglang HTTP server process.

    Thin delegation to :class:`SGLangServerProcess`: this Worker exists so
    the server fits the scheduler's worker-group lifecycle (Ray actor,
    placement, channels); the spawn/monitor/terminate orchestration lives
    in the process class, shared with the server-mode rollout backend.

    Args:
        config: Full RLinf ``DictConfig``. Kept for parity with other
            workers / future use; the sglang server itself is configured
            entirely from ``sglang_cfg``.
        sglang_cfg: The sub-config block whose keys are forwarded verbatim
            as ``ServerArgs(**)`` kwargs — except ``host`` / ``port``, which
            are filled in at runtime here. Typically ``config.rollout.server``
            when used inside a rollout, but any compatible block works
            (the server isn't tied to the rollout pipeline).
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
        self._cfg = config
        self._sglang_cfg = sglang_cfg
        self._bind_host = bind_host
        self._advertise_host = advertise_host
        self._server_type = server_type

        self._process: Optional[SGLangServerProcess] = None

    def init_server(self) -> None:
        """Spawn the sglang HTTP server subprocess and wait for /health.

        A failed spawn is cleaned up by ``SGLangServerProcess.start``
        itself, so the caller never sees a leaked zombie sglang process.

        Raises:
            RuntimeError: if the server fails to become healthy.
        """
        assert self._process is None, "sglang server already initialized."

        server_kwargs = OmegaConf.to_container(self._sglang_cfg, resolve=True) or {}
        self._process = SGLangServerProcess(
            server_args_kwargs=server_kwargs,
            acquire_free_port=self.acquire_free_port,
            server_type=self._server_type,
            bind_host=self._bind_host,
            advertise_host=self._advertise_host,
            log_info=self.log_info,
            log_error=self.log_error,
        )
        self._process.start()

    def get_server_url(self) -> str:
        """Return the advertised ``http://host:port`` URL for this server."""
        assert self._process is not None, "init_server() has not been called."
        return self._process.get_server_url()

    def is_healthy(self) -> bool:
        return self._process is not None and self._process.is_healthy()

    def stop(self) -> None:
        """Terminate the sglang server subprocess (and its process group)."""
        if self._process is None:
            return
        self._process.terminate()
        self._process = None
