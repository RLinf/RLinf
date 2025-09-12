# Copyright 2025 The RLinf Authors.
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

import torch
from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker

from rlinf.scheduler import Worker as _RLinfWorker
from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import RankMapper

from . import weight_loader  # noqa all

logger = init_logger(__name__)


class VLLMWorker(_VllmInnerWorker):
    def __init__(
        self,
        # rlinf specific args
        rlinf_config: DictConfig,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        # vllm former args
        vllm_config: VllmConfig,
        distributed_init_method: str,
        local_rank: int,
        rank: int,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        # rlinf specific
        self.rlinf_config = rlinf_config
        self._rlinf_worker = _RLinfWorker(
            parent_address=parent_address,
            world_size=vllm_config.parallel_config.world_size,
            rank=rank,
        )
        self._actor_group_name = self.rlinf_config.actor.group_name
        self.placement_mode = placement.placement_mode
        rank_map = RankMapper.get_rollout_rank_to_actor_rank_map(placement=placement)
        self.actor_weight_rank = rank_map[
            self._rlinf_worker.get_parent_rank(), self.rank
        ]

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        if isinstance(kv_cache_config, list):
            assert (
                len(kv_cache_config) == self.vllm_config.parallel_config.world_size
            ), (
                f"Got {len(kv_cache_config)} KVCacheConfig, expected {self.vllm_config.parallel_config.world_size}"
            )
            kv_cache_config = kv_cache_config[self.rank]
        super().initialize_from_config(kv_cache_config)

    def offload_model_weights(self) -> None:
        super().sleep(level=2)

    def sync_hf_weight(self) -> None:
        use_cudagraph = not self.rlinf_config.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self._rlinf_worker.recv(
            src_group_name=self._actor_group_name, src_rank=self.actor_weight_rank
        )
        super().wake_up()

        model = self.model_runner.model
        if colocate:
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                list_args[6] = torch.cuda.current_device()
                new_weight: torch.Tensor = func(*list_args)
                model.load_weights([(name, new_weight)])
                del new_weight
        else:
            model.load_weights(state_dict.items())
        super().compile_or_warm_up_model()

    def use_sharded_weights(self) -> None:
        model = self.model_runner.model
        for _, param in model.named_parameters():
            setattr(param, "is_sharded_weight", True)

    def get_dp_rank(self) -> int:
        return self._rlinf_worker.get_parent_rank()
