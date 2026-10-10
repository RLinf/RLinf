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

"""sglang implementations of :class:`RolloutBackend`.

`SGLangEngineBackend` wraps the in-process sglang :class:`Engine`. Every
method body is a verbatim move of the corresponding `self._engine.*` call
from `SGLangWorker`; that equivalence is a hard constraint (plan decision
2.3) so Stage 0 Engine/Server comparison stays attributable.

`SGLangServerBackend` runs the same rollout contract against a spawned
sglang HTTP server subprocess (plan: server mode).
"""

import asyncio
from typing import Callable, Optional

from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.workers.rollout.backend import RlinfContext, RolloutBackend
from rlinf.workers.rollout.sglang_server.process import SGLangServerProcess
from rlinf.workers.rollout.utils import RolloutEngineStats

from . import Engine, io_struct


class SGLangEngineBackend(RolloutBackend):
    """In-process sglang Engine as a rollout backend."""

    def __init__(self, server_args: dict, rlinf_ctx: RlinfContext):
        super().__init__(server_args, rlinf_ctx)
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """The wrapped sglang Engine (set by `initialize`)."""
        if self._engine is None:
            raise RuntimeError(
                "SGLangEngineBackend.engine is not available before initialize()"
            )
        return self._engine

    @property
    def tokenizer_manager(self):
        """The sglang TokenizerManager of the wrapped Engine.

        sglang-specific accessor (plan section 9.3): consumers that need the
        real sglang object (e.g. `SGLangAgentWorkerWithHTTPServer`) go through
        here instead of reaching into backend internals. Currently reached
        indirectly via `SGLangWorker._engine`; this accessor is reserved for
        the later decoupling.
        """
        return self.engine.tokenizer_manager

    async def initialize(self) -> None:
        self._engine = Engine(**self._server_args)
        await self.engine.tokenizer_manager.run_task_method(
            io_struct.TaskMethodInput(
                method_name="init_rlinf_worker",
                args=(
                    self._rlinf_ctx.parent_address,
                    self._rlinf_ctx.weight_reload,
                    self._rlinf_ctx.placement,
                    self._rlinf_ctx.cfg,
                ),
            )
        )

    async def async_generate(
        self,
        *,
        prompt: Optional[list[str] | str] = None,
        input_ids: Optional[list[list[int]] | list[int]] = None,
        sampling_params: list[dict] | dict,
        image_data: Optional[list],
        return_logprob: list[bool] | bool,
    ) -> dict:
        return await self.engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            image_data=image_data,
            return_logprob=return_logprob,
        )

    async def sync_weights(self) -> None:
        await self.engine.tokenizer_manager.sync_hf_weight(
            obj=io_struct.SyncHFWeightInput()
        )

    async def offload(self, tags: Optional[list[str]] = None) -> None:
        await self.engine.tokenizer_manager.release_memory_occupation(
            obj=ReleaseMemoryOccupationReqInput(tags=tags)
        )

    async def onload(self, tags: Optional[list[str]] = None) -> None:
        await self.engine.tokenizer_manager.resume_memory_occupation(
            obj=ResumeMemoryOccupationReqInput(tags=tags)
        )

    async def abort_generation(self) -> None:
        await self.engine.tokenizer_manager.abort_generation(
            obj=io_struct.AbortGenerationInput()
        )

    async def get_running_state(self) -> RolloutEngineStats:
        state = await self.engine.tokenizer_manager.run_task_method(
            io_struct.TaskMethodInput(method_name="get_scheduler_running_state")
        )
        return RolloutEngineStats(**state)

    def shutdown(self) -> None:
        self.engine.shutdown()


class SGLangServerBackend(RolloutBackend):
    """A spawned sglang HTTP server subprocess as a rollout backend.

    All rollout control flows over loopback HTTP: generation via the native
    `/generate`, offload/onload via the native memory-occupation routes,
    and `run_task_method` / `sync_hf_weight` / `abort_generation` via the
    custom `/rlinf/*` routes mounted by
    `rlinf.hybrid_engines.sglang.common.http_routes`. Weight sync itself
    still goes over NCCL inside the scheduler subprocess (started with the
    RLinf patch; see `SGLangServerProcess`).

    The server binds to 127.0.0.1: the only client is this worker's own
    process, and the /rlinf/* routes unpickle their request body, so the
    server must not be reachable from other machines.
    """

    def __init__(
        self,
        server_args: dict,
        rlinf_ctx: RlinfContext,
        *,
        acquire_free_port: Callable[[Optional[int]], int],
        log_info: Callable[[str], None],
        log_error: Callable[[str], None],
    ):
        super().__init__(server_args, rlinf_ctx)
        self._acquire_free_port = acquire_free_port
        self._log_info = log_info or (lambda msg: None)
        self._log_error = log_error or (lambda msg: None)
        self._process: Optional[SGLangServerProcess] = None
        self._client: Optional[InferenceHTTPClient] = None

    @property
    def http_client(self) -> InferenceHTTPClient:
        """The client for the server subprocess (set by `initialize`)."""
        if self._client is None:
            raise RuntimeError(
                "SGLangServerBackend.http_client is not available before initialize()"
            )
        return self._client

    @property
    def server_url(self) -> str:
        """The loopback base URL of the server subprocess."""
        if self._process is None:
            raise RuntimeError(
                "SGLangServerBackend.server_url is not available before initialize()"
            )
        return self._process.get_server_url()

    async def initialize(self) -> None:
        self._process = SGLangServerProcess(
            server_args_kwargs=dict(self._server_args),
            acquire_free_port=self._acquire_free_port,
            server_type="srt",
            bind_host="127.0.0.1",
            advertise_host="127.0.0.1",
            apply_rlinf_patch=True,
            log_info=self._log_info,
            log_error=self._log_error,
        )
        self._process.start()
        self._client = InferenceHTTPClient(self._process.get_server_url())
        await self._client.async_run_task_method(
            io_struct.TaskMethodInput(
                method_name="init_rlinf_worker",
                args=(
                    self._rlinf_ctx.parent_address,
                    self._rlinf_ctx.weight_reload,
                    self._rlinf_ctx.placement,
                    self._rlinf_ctx.cfg,
                ),
            )
        )

    async def async_generate(
        self,
        *,
        prompt: Optional[list[str] | str] = None,
        input_ids: Optional[list[list[int]] | list[int]] = None,
        sampling_params: list[dict] | dict,
        image_data: Optional[list],
        return_logprob: list[bool] | bool,
    ) -> dict:
        return await self.http_client.async_generate(
            prompt=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
        )

    async def sync_weights(self) -> None:
        await self.http_client.async_sync_hf_weight(obj=io_struct.SyncHFWeightInput())

    async def offload(self, tags: Optional[list[str]] = None) -> None:
        await self.http_client.async_release_memory_occupation(tags=tags)

    async def onload(self, tags: Optional[list[str]] = None) -> None:
        await self.http_client.async_resume_memory_occupation(tags=tags)

    async def abort_generation(self) -> None:
        await self.http_client.async_abort_generation(
            obj=io_struct.AbortGenerationInput()
        )

    async def get_running_state(self) -> RolloutEngineStats:
        state = await self.http_client.async_run_task_method(
            io_struct.TaskMethodInput(method_name="get_scheduler_running_state")
        )
        return RolloutEngineStats(**state)

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._client.aclose())
            except RuntimeError:
                asyncio.run(self._client.aclose())
            self._client = None
        if self._process is not None:
            self._process.terminate()
            self._process = None


def make_backend(
    backend_type: str,
    server_args: dict,
    rlinf_ctx: RlinfContext,
    *,
    acquire_free_port: Optional[Callable[[Optional[int]], int]] = None,
    log_info: Optional[Callable[[str], None]] = None,
    log_error: Optional[Callable[[str], None]] = None,
) -> RolloutBackend:
    """Select a rollout backend implementation by config value.

    Args:
        backend_type: Value of `rollout.sglang.backend_type` (`engine` or
            `server`).
        server_args: Shared kwargs dict produced by
            `SGLangWorker._build_server_args`.
        rlinf_ctx: RLinf runtime context consumed during `initialize`.
        acquire_free_port: Worker-bound port allocator (uses the worker's
            port lock); required by the server backend.
        log_info: Worker-bound info logger (server backend only).
        log_error: Worker-bound error logger (server backend only).

    Returns:
        A backend instance (not yet initialized).
    """
    if backend_type == "engine":
        return SGLangEngineBackend(server_args, rlinf_ctx)
    if backend_type == "server":
        assert acquire_free_port is not None, (
            "rollout.sglang.backend_type=server requires the worker's "
            "acquire_free_port (node-local port lock)."
        )
        return SGLangServerBackend(
            server_args,
            rlinf_ctx,
            acquire_free_port=acquire_free_port,
            log_info=log_info,
            log_error=log_error,
        )
    raise ValueError(f"Unsupported rollout.sglang.backend_type: {backend_type!r}")
