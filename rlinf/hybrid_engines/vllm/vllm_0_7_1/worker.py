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

import os
import pickle
import signal
import sys
from typing import Dict, List

import psutil
import torch
import zmq
from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import (
    CuMemAllocator,
)
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.utils import get_mp_context, get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.v1.executor.multiproc_executor import WorkerProc, WorkerProcHandle
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker
from vllm.worker.worker_base import WorkerWrapperBase

from rlinf.scheduler import Worker as _RLinfWorker
from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import DisaggRankMapper, HybridRankMapper
from rlinf.workers.rollout.vllm.io_struct import (
    OffloadModelWeightCommand,
    SyncHFWeightCommand,
)

from . import weight_loader  # noqa all

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
    ):
        _RLinfWorker.__init__(
            self,
            parent_address=parent_address,
            world_size=vllm_config.parallel_config.world_size,
            rank=rank,
        )
        super().__init__(vllm_config, local_rank, rank, distributed_init_method)

        self.rlinf_config = rlinf_config
        self._actor_group_name = self.rlinf_config.actor.group_name
        self.placement_mode = placement.placement_mode
        if self.placement_mode == PlacementMode.COLLOCATED:
            self.actor_weight_rank = (
                HybridRankMapper.get_rollout_rank_to_actor_rank_map(
                    self.rlinf_config.actor.model.tensor_model_parallel_size,
                    self.rlinf_config.actor.model.pipeline_model_parallel_size,
                    self.rlinf_config.rollout.tensor_parallel_size,
                    self.rlinf_config.cluster.num_nodes
                    * self.rlinf_config.cluster.num_gpus_per_node,
                )[self.get_parent_rank(), self._rank]
            )
        elif self.placement_mode == PlacementMode.DISAGGREGATED:
            rank_map = DisaggRankMapper.get_rollout_rank_to_actor_rank_map(
                actor_tp_size=self.rlinf_config.actor.model.tensor_model_parallel_size,
                actor_pp_size=self.rlinf_config.actor.model.pipeline_model_parallel_size,
                actor_world_size=placement.actor_world_size,
                rollout_tp_size=self.rlinf_config.rollout.tensor_parallel_size,
                rollout_world_size=placement.rollout_world_size,
            )
            self.log_info(
                f"Rollout rank to actor rank mapping: {rank_map}, try to get {(self.get_parent_rank(), self._rank)}"
            )
            self.actor_weight_rank = rank_map[self.get_parent_rank(), self._rank]
        else:
            raise ValueError(f"Unsupported placement mode: {self.placement_mode}")

        self.log_info(
            f"Running VllmInnerWoker dp rank {self.get_parent_rank()}, tp_rank {self._rank}, corresponding actor weight rank = {self.actor_weight_rank}"
        )

    def sync_hf_weight(self, command: SyncHFWeightCommand) -> None:
        use_cudagraph = not self.rlinf_config.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self.recv(
            src_group_name=self._actor_group_name, src_rank=self.actor_weight_rank
        )

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up()
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

        self.restore_named_buffers()
        torch.cuda.synchronize()
        super().compile_or_warm_up_model()

    def offload_model_weights(self, command: OffloadModelWeightCommand) -> None:
        self.save_named_buffers()
        free_bytes_before_offload = torch.cuda.mem_get_info()[0]
        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",))

        free_bytes_after_offload = torch.cuda.mem_get_info()[0]
        freed_bytes = free_bytes_after_offload - free_bytes_before_offload
        assert freed_bytes >= 0, (
            f"before offload:{free_bytes_before_offload}, after offload:{free_bytes_after_offload}"
        )

    def save_named_buffers(self) -> None:
        model = self.model_runner.model
        self.saved_buffers = {
            name: buffer.cpu().clone() for name, buffer in model.named_buffers()
        }

    def restore_named_buffers(self) -> None:
        model = self.model_runner.model
        for name, buffer in model.named_buffers():
            if name in self.saved_buffers:
                buffer.data.copy_(self.saved_buffers[name].data)
        self.saved_buffers = {}
        logger.info("vllm worker restored named buffers from saved buffers")

    def use_sharded_weights(self) -> None:
        model = self.model_runner.model
        for _, param in model.named_parameters():
            setattr(param, "is_sharded_weight", True)


class VLLMWorkerProc(WorkerProc):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        parent_address: WorkerAddress,
        distributed_init_method: str,
        input_shm_handle: Handle,
        ready_path: str,
        placement: ModelParallelComponentPlacement,
    ):
        self.rank = rank  # global rank in 2D parallel (tp,pp)

        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        all_kwargs: List[Dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "parent_address": parent_address,
            "placement": placement,
        }
        wrapper.init_worker(all_kwargs=all_kwargs)

        self.worker = wrapper.worker

        pid = os.getpid()
        _add_prefix(
            sys.stdout,
            f"VllmWorkerProc[dp_rank={self.worker.get_parent_rank()},tp_rank={self.rank}]",
            pid,
        )
        _add_prefix(
            sys.stderr,
            f"VllmWorkerProc[dp_rank={self.worker.get_parent_rank()},tp_rank={self.rank}]",
            pid,
        )
        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
        worker_response_mq_handle = self.worker_response_mq.export_handle()

        # Send Readiness signal to EngineCore process.
        with zmq_socket_ctx(ready_path, zmq.constants.PUSH) as ready_socket:
            payload = pickle.dumps(
                worker_response_mq_handle, protocol=pickle.HIGHEST_PROTOCOL
            )
            ready_socket.send_string(WorkerProc.READY_STR)
            ready_socket.send(payload)

        self.worker.init_device()
        self.worker.load_model()
        # after load_model, we should save it's named_buffers to implement sync weight
        self.worker.use_sharded_weights()

    @staticmethod
    def make_worker_process(
        rank: int,
        local_rank: int,
        distributed_init_method: str,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        vllm_config: VllmConfig,
        input_shm_handle: Handle,
    ) -> WorkerProcHandle:
        """
        Note:
        This function is modified from vllm's raw implementation. because it in vllm hardcodes that
        it will launch `WorkerProc` which can't be selected, we can only copy and modify it here.
        """
        context = get_mp_context()

        # ZMQ path for worker to send ready message and shm_broadcast handle
        # back to core process.
        ready_path = get_open_zmq_ipc_path()
        process_kwargs = {
            "rank": rank,
            "local_rank": local_rank,
            "distributed_init_method": distributed_init_method,
            "parent_address": parent_address,
            "placement": placement,
            "vllm_config": vllm_config,
            "input_shm_handle": input_shm_handle,
            "ready_path": ready_path,
        }

        proc = context.Process(
            target=VLLMWorkerProc.worker_main, kwargs=process_kwargs, daemon=True
        )

        proc.start()
        # Wait for startup
        worker_response_mq_handle = WorkerProc.wait_for_startup(proc, ready_path)

        worker_response_mq = MessageQueue.create_from_handle(
            worker_response_mq_handle, 0
        )

        return WorkerProcHandle(proc, rank, ready_path, worker_response_mq)

    @staticmethod
    def worker_main(*args, **kwargs):
        """
        Note:
        This function is modified from vllm's raw implementation. because it in vllm hardcodes that
        it will launch `WorkerProc` which can't be selected, we can only copy and modify it here.
        """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        try:
            worker = VLLMWorkerProc(*args, **kwargs)

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()

            worker.worker_busy_loop()

        except SystemExit:
            logger.debug("Worker interrupted.")

        except Exception:
            # worker_busy_loop sends exceptions exceptons to Executor
            # for shutdown, but if there is an error in startup or an
            # error with IPC itself, we need to alert the parent.
            psutil.Process().parent().send_signal(signal.SIGUSR1)
            raise

        finally:
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
                worker = None
