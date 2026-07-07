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

from typing import Any, Callable

import torch
from torch.distributed.tensor import DTensor

from rlinf.scheduler import CollectiveGroupOptions

from .base import RecvFn, SendFn, WeightSyncer

WorkerPointToPointSendFn = Callable[..., Any]


class VanillaWeightSyncer(WeightSyncer):
    """Simple full state dict sync.

    The sender point-to-point sends the actor ``state_dict`` to mapped rollout
    ranks. The receiver waits for a point-to-point ``recv``, loads weights with
    ``load_state_dict``, and releases the received dict.
    """

    def __init__(self):
        super().__init__()
        self._rollout_group_name: str | None = None
        self._weight_dst_ranks: list[int] | None = None
        self._worker_send: WorkerPointToPointSendFn | None = None

    @staticmethod
    def setup_weight_dst_ranks(
        rollout_world_size: int, actor_world_size: int, actor_rank: int
    ) -> list[int]:
        """Map one actor rank to its rollout destination ranks."""
        weight_dst_ranks: list[int] = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + actor_rank < rollout_world_size:
                weight_dst_ranks.append(i * actor_world_size + actor_rank)
        return weight_dst_ranks

    def configure_sender(
        self,
        *,
        rollout_group_name: str,
        weight_dst_ranks: list[int],
        worker_send: WorkerPointToPointSendFn,
    ) -> None:
        self._rollout_group_name = rollout_group_name
        self._weight_dst_ranks = list(weight_dst_ranks)
        self._worker_send = worker_send
        self._sender_initialized = True

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn | None = None,
        version: int | torch.Tensor | None = None,
    ) -> None:
        del send, version
        if (
            self._worker_send is None
            or self._rollout_group_name is None
            or self._weight_dst_ranks is None
        ):
            raise RuntimeError("VanillaWeightSyncer sender is not configured")

        comm_options: CollectiveGroupOptions | None = self.comm_options
        handles = []
        for rank in self._weight_dst_ranks:
            handles.append(
                self._worker_send(
                    state_dict,
                    self._rollout_group_name,
                    rank,
                    async_op=True,
                    options=comm_options,
                )
            )
        for handle in handles:
            await handle.async_wait()

    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int:
        param_state_dict = await recv()
        model.load_state_dict(param_state_dict)
        del param_state_dict
        return 0
