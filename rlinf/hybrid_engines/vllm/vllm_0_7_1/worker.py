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

from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker

from rlinf.scheduler import Worker as _RLinfWorker
from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import DisaggRankMapper, HybridRankMapper


class VLLMWorker(_VllmInnerWorker, _RLinfWorker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rlinf_config: DictConfig,
        distributed_init_method: str,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        local_rank: int,
        rank: int,
        world_size: int,
    ):
        _RLinfWorker.__init__(
            self, parent_address=parent_address, world_size=world_size, rank=rank
        )
        super().__init__(vllm_config, local_rank, rank, distrubuted_init_method)

        self.rlinf_config = rlinf_config
        self._actor_group_name = self.rlinf_config.actor.group_name
        self.placement_mode = placement.placement_mode
        if self.placement_mode == PlacementMode.COLLOCATED:
            self.actor_weight_rank = (
                HybridRankMapper.get_rollout_rank_to_actor_rank_map(
                    self.cfg.actor.model.tensor_model_parallel_size,
                    self.cfg.actor.model.pipeline_model_parallel_size,
                    self.cfg.rollout.tensor_parallel_size,
                    self.cfg.cluster.num_nodes * self.cfg.cluster.num_gpus_per_node,
                )[self.get_parent_rank(), self._rank]
            )
        elif self.placement_mode == PlacementMode.DISAGGREGATED:
            rank_map = DisaggRankMapper.get_rollout_rank_to_actor_rank_map(
                actor_tp_size=self.cfg.actor.model.tensor_model_parallel_size,
                actor_pp_size=self.cfg.actor.model.pipeline_model_parallel_size,
                actor_world_size=placement.actor_world_size,
                rollout_tp_size=self.cfg.rollout.tensor_parallel_size,
                rollout_world_size=placement.rollout_world_size,
            )
            self.log_info(
                f"Rollout rank to actor rank mapping: {rank_map}, try to get {(self.get_parent_rank(), self._rank)}"
            )
            self.actor_weight_rank = rank_map[self.get_parent_rank(), self._rank]
        else:
            raise ValueError(f"Unsupported placement mode: {self.placement_mode}")

        self.log_info(
            f"Running VllmInnerWoker dp rank {self.get_parent_rank()}, tp_rank {rank}, corresponding actor weight rank = {self.actor_weight_rank}"
        )

    def _dispatch_loop(self):
        raise NotImplementedError("VLLMWorker._dispatch_loop is not implemented yet.")

    def sync_weight(self):
        raise NotImplementedError("VLLMWorker.sync_weight is not implemented yet.")

    def offload_model_weights(self):
        raise NotImplementedError(
            "VLLMWorker.offload_model_weights is not implemented yet."
        )
