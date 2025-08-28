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
import os
import signal
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing.process import BaseProcess
from typing import Dict, List

import cloudpickle
import psutil
import torch
import zmq
from omegaconf import DictConfig
from pydantic import TypeAdapter, ValidationError
from torch import nn
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.utils import GiB_bytes, get_mp_context
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker
from vllm.worker.worker_base import WorkerWrapperBase

from rlinf.scheduler import Worker as _RLinfWorker
from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import DisaggRankMapper, HybridRankMapper
from rlinf.workers.rollout.vllm.io_struct import (
    CollectiveRpcCommand,
    CollectiveRpcResponse,
    OffloadModelWeightCommand,
    SyncHFWeightCommand,
    VLLMCommand,
)

logger = init_logger(__name__)


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

        self.log_info(
            f"Running VllmInnerWoker dp rank {self.get_parent_rank()}, tp_rank {rank}, corresponding actor weight rank = {self.actor_weight_rank}"
        )

    def sync_hf_weight(self, command: SyncHFWeightCommand) -> CollectiveRpcResponse:
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

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput):
        output: ModelRunnerOutput = super().execute_model(scheduler_output)
        return CollectiveRpcResponse(
            rank=self._rank, data=output if self._rank == 0 else None, success=True
        )

    def get_kv_cache_spec(self) -> CollectiveRpcResponse:
        kv_cache_spec: KVCacheSpec = super().get_kv_cache_spec()
        return CollectiveRpcResponse(rank=self._rank, data=kv_cache_spec, success=True)

    def initialize_cache(self, kv_cache_config) -> CollectiveRpcResponse:
        result: None = super().initialize_cache(kv_cache_config)
        return CollectiveRpcResponse(rank=self._rank, data=result, success=True)

    def compile_or_warm_up_model(self) -> CollectiveRpcResponse:
        result: None = super().compile_or_warm_up_model()
        return CollectiveRpcResponse(rank=self._rank, data=result, success=True)

    @torch.inference_mode()
    def determine_available_memory(self) -> CollectiveRpcResponse:
        result: int = super().determine_available_memory()  # byte
        return CollectiveRpcResponse(rank=self._rank, data=result, success=True)

    def profile(self, is_start: bool = True) -> CollectiveRpcResponse:
        result: None = super().profile(is_start)
        return CollectiveRpcResponse(rank=self._rank, data=result, success=True)

    def check_health(self) -> CollectiveRpcResponse:
        result: None = super().check_health()
        return CollectiveRpcResponse(rank=self._rank, data=result, success=True)

    def offload_model_weights(
        self, command: OffloadModelWeightCommand
    ) -> CollectiveRpcResponse:
        free_bytes_before_offload = torch.cuda.mem_get_info()[0]
        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",))
        free_bytes_after_offload, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_before_offload - free_bytes_after_offload
        assert freed_bytes >= 0
        logger.info(
            "Vllm inner worker offload weights: offloaded %.2f GiB memory",
            freed_bytes / GiB_bytes,
        )
        return CollectiveRpcResponse(
            command_id=command.command_id,
            rank=self._rank,
            data={"freed_bytes": freed_bytes},
            success=True,
        )

    def get_model(self) -> CollectiveRpcResponse:
        result: nn.Module = super().get_model()
        return CollectiveRpcResponse(
            rank=self._rank, data=result if self._rank == 0 else None, success=True
        )


@dataclass
class ZmqWorkerProcHandle:
    proc: BaseProcess
    rank: int


class ZmqWorkerProc:
    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
        worker_input_ipc_name: str,
        worker_output_ipc_name: str,
        parent_address: WorkerAddress,
        distributed_init_method: str,
        placement: ModelParallelComponentPlacement,
    ):
        self.rank = rank
        self.vllm_config = vllm_config

        self.worker_input_ipc_name = worker_input_ipc_name
        self.worker_output_ipc_name = worker_output_ipc_name

        context = zmq.Context()

        self.send_to_executor = context.socket(zmq.PUSH)
        self.send_to_executor.connect(self.worker_output_ipc_name)

        self.recv_from_executor = context.socket(zmq.PULL)
        self.recv_from_executor.connect(self.worker_input_ipc_name)

        self.command_parser = TypeAdapter(VLLMCommand)

        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        all_kwargs: List[Dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "world_size": vllm_config.parallel_config.world_size,
            "parent_address": parent_address,
            "placement": placement,
        }
        wrapper.init_worker(all_kwargs=all_kwargs)

        self.worker = wrapper.worker

        pid = os.getpid()
        _add_prefix(sys.stdout, f"VllmZmqWorker[rank={rank}]", pid)
        _add_prefix(sys.stderr, f"VllmZmqWorker[rank={rank}]", pid)

    @staticmethod
    def worker_main(*args, **kwargs):
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        try:
            worker = ZmqWorkerProc(*args, **kwargs)

            worker.worker_busy_loop()
        except SystemExit:
            logger.debug("Vllm inner worker proc interrupted.")
        except Exception as e:
            logger.fatal(f"Vllm inner worker proc failed with exception: {e}")
            psutil.Process().parent().send_signal(signal.SIGUSR1)
            raise
        finally:
            if worker is not None:
                worker.shutdown()
                worker = None

    @staticmethod
    def make_worker_process(
        rank: int,
        worker_input_ipc_name: str,
        worker_output_ipc_name: str,
        distributed_init_method: str,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
    ) -> ZmqWorkerProcHandle:
        mp_context = get_mp_context()
        process_kwargs = {
            "rank": rank,
            "worker_input_ipc_name": worker_input_ipc_name,
            "worker_output_ipc_name": worker_output_ipc_name,
            "distributed_init_method": distributed_init_method,
            "parent_address": parent_address,
            "placement": placement,
        }

        proc = mp_context.Process(
            target=ZmqWorkerProc.worker_main, kwargs=process_kwargs, daemon=True
        )

        proc.start()

        return ZmqWorkerProcHandle(proc=proc, rank=rank)

    def shutdown(self):
        if getattr(self, "shutting_down", False):
            self.shutting_down = True

        self.send_to_executor.close()
        self.recv_from_executor.close()
        destroy_model_parallel()
        destroy_distributed_environment()

    def worker_busy_loop(self):
        while True:
            try:
                command_str: str = self.recv_from_executor.recv_string(
                    flags=zmq.NOBLOCK
                )
                command: CollectiveRpcCommand = self.command_parser.validate_json(
                    command_str
                )
                if isinstance(command.method, str):
                    func = getattr(self.worker, command.method, None)
                elif isinstance(command.method, bytes):
                    func = partial(cloudpickle.loads(command.method), self.worker)
                else:
                    logger.error(
                        f"Unknown method type recevied in zmq worker's worker_busy_loop: {type(command.method)}"
                    )
                    func = None
                if func:
                    logger.debug(f"Vllm inner worker dispatching command: {command}")
                    response = func(*command.args, **command.kwargs)
                else:
                    logger.error(
                        f"Vllm inner worker received unknown command: {command}"
                    )
                    response = CollectiveRpcResponse(
                        command_id=command.command_id,
                        rank=self._rank,
                        data=None,
                        success=False,
                        error=f"Unknown command type: {command.command_type}",
                    )
                assert isinstance(response, CollectiveRpcResponse), (
                    "Response is not a CollectiveRpcResponse. Check if there is some function you don't reimplement"
                )
                self.send_to_executor.send_string(response.model_dump_json())
            except zmq.Again:
                continue
            except ValidationError as e:
                logger.error(f"Failed to parse command: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in dispatch loop: {e}")
