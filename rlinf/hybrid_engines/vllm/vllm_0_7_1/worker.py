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

import json
import threading

import zmq
from omegaconf import DictConfig
from pydantic import TypeAdapter, ValidationError
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker

from rlinf.scheduler import Worker as _RLinfWorker
from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import DisaggRankMapper, HybridRankMapper
from rlinf.workers.rollout.vllm.io_struct import (
    OffloadModelWeightCommand,
    SyncHFWeightCommand,
    VLLMCommand,
)


class VLLMWorker(_VllmInnerWorker, _RLinfWorker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rlinf_config: DictConfig,
        distributed_init_method: str,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        worker_input_ipc_name: str,
        worker_output_ipc_name: str,
        local_rank: int,
        rank: int,
        world_size: int,
    ):
        _RLinfWorker.__init__(
            self, parent_address=parent_address, world_size=world_size, rank=rank
        )
        super().__init__(vllm_config, local_rank, rank, distributed_init_method)

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

        context = zmq.Context()

        self.send_to_executor = context.socket(zmq.PUSH)
        self.send_to_executor.connect(worker_output_ipc_name)

        self.recv_from_executor = context.socket(zmq.PULL)
        self.recv_from_executor.connect(worker_input_ipc_name)

        self.command_parser = TypeAdapter(VLLMCommand)

        self.command_handlers = {
            "sync_hf_weights": self.sync_hf_weight,
            "offload_model_weights": self.offload_model_weights,
        }

        self._dispatch_loop_handle = threading.Thread(
            target=self._dispatch_loop, daemon=True
        )
        self._dispatch_loop_handle.start()

        self.log_info(
            f"Running VllmInnerWoker dp rank {self.get_parent_rank()}, tp_rank {rank}, corresponding actor weight rank = {self.actor_weight_rank}"
        )

    def _dispatch_loop(self):
        while True:
            try:
                command_str = self.recv_from_executor.recv_string(flags=zmq.NOBLOCK)
                command = self.command_parser.validate_json(command_str)
                handler = self.command_handlers.get(command.command_type)
                if handler:
                    self.log_debug(f"Vllm inner worker dispatching command: {command}")
                    handler(command)
                else:
                    self.log_error(
                        f"Vllm inner worker received unknown command: {command}"
                    )
            except zmq.Again:
                continue
            except ValidationError as e:
                self.log_error(f"Failed to parse command: {e}")
            except json.JSONDecodeError as e:
                self.log_error(f"Failed to decode JSON: {e}")
            except Exception as e:
                self.log_error(f"Unexpected error in dispatch loop: {e}")

    def sync_hf_weight(self, command: SyncHFWeightCommand):
        use_cudagraph = not self.cfg.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self.recv(
            src_group_name=self._actor_group_name, src_rank=self.actor_weight_rank
        )

        model = self.model_runner.model

        if colocate:
            pass
        else:
            model.load_state_dict(state_dict)

    def offload_model_weights(self, command: OffloadModelWeightCommand):
        raise NotImplementedError(
            "VLLMWorker.offload_model_weights is not implemented yet."
        )
