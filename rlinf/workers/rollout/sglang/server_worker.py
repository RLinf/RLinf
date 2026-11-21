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
        apply_rlinf_patch: bool = False,
    ):
        Worker.__init__(self)
        self._cfg = config
        self._sglang_cfg = sglang_cfg
        self._bind_host = bind_host
        self._advertise_host = advertise_host
        self._server_type = server_type
        self._apply_rlinf_patch = bool(apply_rlinf_patch)

        self._process: Optional[SGLangServerProcess] = None

    async def init_server(self) -> None:
        """Spawn the sglang HTTP server subprocess and wait for /health.

        If the server fails to become healthy, ``SGLangServerProcess.start``
        tears the subprocess down and re-raises, so the caller never sees a
        leaked sglang process.

        An engine that fits on one node is a single process. An engine that
        does not negotiates one ``dist_init_addr`` on its entry node and
        starts one process per node, the same rendezvous the reasoning
        SGLang worker uses.
        """
        assert self._process is None, "sglang server already initialized."

        from rlinf.workers.rollout.sglang.rendezvous import (
            negotiate_model_instance_dist_addr,
        )

        nnodes = int(self._cfg.rollout.sglang.get("nnodes_per_model_instance", 1))
        self._node_rank = 0
        dist_init_addr = None
        if nnodes > 1:
            placement = self._ipc_placement()
            self._node_rank = placement.rollout_node_rank_in_model_instance(self._rank)
            host, port = await negotiate_model_instance_dist_addr(self, placement)
            dist_init_addr = f"{host}:{port}"

        server_kwargs = OmegaConf.to_container(self._sglang_cfg, resolve=True) or {}
        self._process = SGLangServerProcess(
            server_args_kwargs=server_kwargs,
            acquire_free_port=self.acquire_free_port,
            server_type=self._server_type,
            nnodes=nnodes,
            node_rank=self._node_rank,
            dist_init_addr=dist_init_addr,
            bind_host=self._bind_host,
            advertise_host=self._advertise_host,
            apply_rlinf_patch=self._apply_rlinf_patch,
            log_info=self.log_info,
            log_error=self.log_error,
        )
        self._process.start()

    async def init_rlinf_weight_sync(self) -> None:
        """Register scheduler children so the FSDP actor can CUDA-IPC weights.

        The patched server exposes ``/rlinf/run_task_method``. One call on the
        entry process fans ``init_rlinf_worker`` out to every TP rank.
        """
        if not self._apply_rlinf_patch:
            raise RuntimeError(
                "init_rlinf_weight_sync requires rollout.sglang.apply_rlinf_patch"
            )
        placement = self._ipc_placement()
        if placement.rollout_node_rank_in_model_instance(self._rank) != 0:
            # The entry process fans init out to every TP rank, including
            # ranks whose scheduler runs on another node of this engine.
            return
        from rlinf.hybrid_engines.sglang.common.io_struct import TaskMethodInput
        from rlinf.utils.http_client import InferenceHTTPClient

        client = InferenceHTTPClient(self.get_server_url())
        await client.async_run_task_method(
            TaskMethodInput(
                method_name="init_rlinf_worker",
                args=[
                    self.worker_address,
                    "sync",
                    placement,
                    self._cfg,
                    placement.rollout_model_instance_id(self._rank),
                ],
            )
        )

    def _ipc_placement(self):
        from rlinf.workers.rollout.sglang.ipc_placement import CollocatedIpcPlacement

        sglang_cfg = self._cfg.rollout.sglang
        if "actor_world_size" not in sglang_cfg:
            raise RuntimeError(
                "rollout.sglang.actor_world_size is unset. The training "
                "driver fills it from the actor placement before launch."
            )
        return CollocatedIpcPlacement(
            int(sglang_cfg.actor_world_size),
            int(sglang_cfg.tensor_parallel_size),
            int(sglang_cfg.get("nnodes_per_model_instance", 1)),
        )

    def get_server_url(self) -> str | None:
        """Return the advertised ``http://host:port`` URL for this server.

        Non-entry processes of a cross-node engine do not serve requests.
        """
        assert self._process is not None, "init_server() has not been called."
        if getattr(self, "_node_rank", 0) != 0:
            return None
        return self._process.get_server_url()

    def is_healthy(self) -> bool:
        return self._process is not None and self._process.is_healthy()

    def stop(self) -> None:
        """Terminate the sglang server subprocess (and its process group)."""
        if self._process is None:
            return
        self._process.terminate()
        self._process = None
