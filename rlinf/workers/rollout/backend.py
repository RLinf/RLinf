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

"""Backend-neutral contract between a rollout Worker and its inference engine.

One backend instance represents one model instance's handle on this process.
Orchestration logic (e.g. `SGLangWorker.rollout` / `_async_generate_group`)
depends only on this interface, so the same orchestration serves in-process
engines and out-of-process HTTP servers alike.

This module must stay backend-neutral: only RLinf types and plain Python
types appear in the signatures. Backend-specific imports belong in the
backend implementations.

Temporary state: `SGLangWorker` and `VLLMWorker` still duplicate their
orchestration. This round deliberately does not merge them — that is a
settled architecture decision (2026-09-10: keep two subclasses; the ABC is
a cross-worker contract, not shared implementation). Trigger for revisiting:
an explicit new proposal to unify the two worker classes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from omegaconf import DictConfig

from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.utils import RolloutEngineStats


@dataclass
class RlinfContext:
    """Runtime context a backend needs to join the RLinf communication mesh.

    Fields mirror the real signature of the sglang-side entry point
    (`init_rlinf_worker(parent_address, weight_reload, placement, config,
    model_instance_id)`) but are all RLinf-neutral types, so future backends
    (e.g. vLLM) can consume the same data.
    """

    parent_address: WorkerAddress
    weight_reload: Literal["sync", "cpu", None]
    placement: ModelParallelComponentPlacement
    cfg: DictConfig
    model_instance_id: int


class RolloutBackend(ABC):
    """Handle to one model instance on this process.

    Args:
        server_args: Backend-specific kwargs produced by the worker
            (e.g. `dataclasses.asdict(ServerArgs)`), shared verbatim by all
            backend implementations of the same framework.
        rlinf_ctx: RLinf runtime context consumed during `initialize`.
    """

    def __init__(self, server_args: dict, rlinf_ctx: RlinfContext):
        # Store only: no process spawning, no connections, no RPCs here.
        self._server_args = server_args
        self._rlinf_ctx = rlinf_ctx

    @abstractmethod
    async def initialize(self) -> None:
        """Bring the backend up and join the RLinf runtime.

        Includes passing the RLinf context to the backend (each backend does
        this at its own stage of initialization). Once this returns, the
        backend is ready to serve requests.
        """

    @abstractmethod
    async def async_generate(
        self,
        *,
        prompt: Optional[list[str] | str] = None,
        input_ids: Optional[list[list[int]] | list[int]] = None,
        sampling_params: list[dict] | dict,
        image_data: Optional[list],
        return_logprob: list[bool] | bool,
    ) -> dict:
        """Generate for a batch of prompts, given as text or as token ids."""

    @abstractmethod
    async def sync_weights(self) -> None:
        """Pull the latest weights from the actor side."""

    @abstractmethod
    async def offload(self, tags: Optional[list[str]] = None) -> None:
        """Release backend-held GPU memory regions.

        Tag values are defined by each backend (sglang: weights / kv_cache /
        cuda_graph); cross-backend alignment of tag semantics is deferred to
        the vLLM round.
        """

    @abstractmethod
    async def onload(self, tags: Optional[list[str]] = None) -> None:
        """Restore backend-held GPU memory regions released by `offload`."""

    @abstractmethod
    async def abort_generation(self) -> None:
        """Abort all in-flight generations."""

    @abstractmethod
    async def get_running_state(self) -> RolloutEngineStats:
        """Report the backend's current occupancy / running statistics."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear the backend down (processes, connections)."""
