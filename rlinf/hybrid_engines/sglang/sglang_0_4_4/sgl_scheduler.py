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

import faulthandler
import logging
import os
import signal
from typing import Optional

import psutil
import setproctitle
import torch
from omegaconf import DictConfig
from sglang.srt.managers.io_struct import (
    AbortReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler import Scheduler as _Scheduler
from sglang.srt.managers.scheduler import logger
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

from rlinf.scheduler import Worker, WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.rollout.utils import (
    DisaggRankMapper,
    HybridRankMapper,
    get_module_from_name,
    rebind_param_attr,
    swap_tensor_pointer,
)

from .io_struct import (
    OffloadReqInput,
    OffloadReqOutput,
    SyncHFWeightInput,
    SyncHFWeightOutput,
    SyncWeightInput,
    SyncWeightOutput,
    TaskMethodInput,
    TaskMethodOutput,
)

logger.setLevel(logging.WARNING)


class Scheduler(_Scheduler, Worker):
    """
    Overridden class of SGLang's TP worker class _Scheduler.
    A Scheduler is a Task that manages the TP worker, and performs necessary weight synchronization with actor and weight offloading.
    """

    def __init__(
        self,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        config: DictConfig,
        world_size: int,
        rank: int,
        server_args: ServerArgs,
        port_args,
        gpu_id,
        tp_rank,
        dp_rank,
    ):
        Worker.__init__(
            self, parent_address=parent_address, world_size=world_size, rank=rank
        )

        _Scheduler.__init__(self, server_args, port_args, gpu_id, tp_rank, dp_rank)
        # `TpModelWorkerClient` is used when ServerArgs.enable_overlap=True, and it has 'worker' attribute.
        # But in early SGLang version, `TpModelWorker` doesn't have 'worker' attribute.
        if not hasattr(self.tp_worker, "worker"):
            self.tp_worker.worker = self.tp_worker

        self._request_dispatcher._mapping.extend(
            [
                (TaskMethodInput, self.run_task_method),
                (OffloadReqInput, self.offload_model_weights),
                (SyncWeightInput, self.sync_weight),
                (SyncHFWeightInput, self.sync_hf_weight),
            ]
        )
        self.cfg = config
        self.binded_attr = {}

        self._actor_group_name = self.cfg.actor.group_name
        self.placement_mode = placement.placement_mode
        if self.placement_mode == PlacementMode.COLLOCATED:
            self.actor_weight_rank = (
                HybridRankMapper.get_rollout_rank_to_actor_rank_map(
                    self.cfg.actor.model.tensor_model_parallel_size,
                    self.cfg.actor.model.pipeline_model_parallel_size,
                    self.cfg.rollout.tensor_parallel_size,
                    self.cfg.cluster.num_nodes * self.cfg.cluster.num_gpus_per_node,
                )[(self.get_parent_rank(), self._rank)]
            )
        else:
            assert self.placement_mode == PlacementMode.DISAGGREGATED, (
                f"Unsupported placement mode: {self.placement_mode}"
            )
            rank_map = DisaggRankMapper.get_rollout_rank_to_actor_rank_map(
                actor_tp_size=self.cfg.actor.model.tensor_model_parallel_size,
                actor_pp_size=self.cfg.actor.model.pipeline_model_parallel_size,
                actor_world_size=placement.actor_world_size,
                rollout_tp_size=self.cfg.rollout.tensor_parallel_size,
                rollout_world_size=placement.rollout_world_size,
            )
            self._logger.info(
                f"Rollout rank to actor rank mapping: {rank_map}, try to get {(self.get_parent_rank(), self._rank)}"
            )

            self.actor_weight_rank = rank_map[(self.get_parent_rank(), self._rank)]

        # it's important to use load_weight to load resharded weight from megatron
        for _, module in self.tp_worker.worker.model_runner.model.named_modules():
            if hasattr(module, "use_presharded_weights"):
                module.use_presharded_weights = True

        self._logger.info(
            f"Running Scheduler dp rank {self.get_parent_rank()}, tp rank {self.tp_rank}, corresponding actor weight rank = {self.actor_weight_rank}"
        )

    def sync_in_tp(self, fn: str = ""):
        broadcast_pyobj(
            [], self.tp_rank, self.tp_worker.worker.model_runner.tp_group.cpu_group
        )
        # logger.info(f"{fn}: Sync in tp success!")

    def cuda_info(self, text: str = ""):
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        free_gpu_memory /= 2**30
        total_gpu_memory /= 2**30

        memory_allocated = torch.cuda.memory_allocated() / 2**30
        memory_reserved = torch.cuda.memory_reserved() / 2**30

        self._logger.info(
            f"[dp {self.get_parent_rank()}-tp {self.tp_rank}] {text} "
            f"{memory_allocated=:.2f} GiB, {memory_reserved=:.2f} GiB, "
            f"{free_gpu_memory=:.2f} GiB, {total_gpu_memory=:.2f} GiB"
        )

    def offload_model_weights(self, recv_req: OffloadReqInput):
        use_cudagraph = not self.cfg.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        if not colocate:
            assert use_cudagraph, "If not colocate, use_cudagraph must be True now."

        if use_cudagraph or not colocate:
            self.release_memory_occupation(ReleaseMemoryOccupationReqInput())
            # self.cuda_info("After offload Model weights and kv cache")
            return OffloadReqOutput()

        # manually offload
        self.named_buffers = {
            n: buf.clone()
            for n, buf in self.tp_worker.worker.model_runner.model.named_buffers()
        }

        self.binded_attr = {
            name: param.__dict__
            for name, param in self.tp_worker.worker.model_runner.model.named_parameters()
        }

        # offload parameters
        self.tp_worker.worker.model_runner.model.to("meta")

        # offload kv cache
        self.tp_worker.worker.model_runner.token_to_kv_pool._clear_buffers()

        self.flush_cache()
        self.sync_in_tp("offload_model_weights")
        # self.cuda_info("After offload Model weights and kv cache")
        return OffloadReqOutput()

    def sync_hf_weight(self, recv_req: SyncHFWeightInput):
        use_cudagraph = not self.cfg.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED

        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self.recv(
            src_group_name=self._actor_group_name,
            src_rank=self.actor_weight_rank,
        )

        model = self.tp_worker.worker.model_runner.model

        if colocate:
            self.resume_memory_occupation(ResumeMemoryOccupationReqInput())
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                # NOTE: the key is to change device id to the current device id
                # in case two processes have different CUDA_VISIBLE_DEVICES
                list_args[6] = torch.cuda.current_device()
                new_weight = func(*list_args)

                model.load_weights([(name, new_weight)])
                del new_weight
        else:
            # disaggregate mode, recv tensor directly
            for name, tensor in state_dict.items():
                model.load_weights([(name, tensor)])
        self.sync_in_tp("sync_hf_weight")
        return SyncHFWeightOutput()

    def sync_weight(self, recv_req: SyncWeightInput):
        use_cudagraph = not self.cfg.rollout.enforce_eager
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        if not colocate:
            assert use_cudagraph, "If not colocate, use_cudagraph must be True now."

        state_dict = self.recv(
            src_group_name=self._actor_group_name,
            src_rank=self.actor_weight_rank,
        )
        model = self.tp_worker.worker.model_runner.model

        if use_cudagraph and colocate:
            self.resume_memory_occupation(ResumeMemoryOccupationReqInput())

        if colocate:
            if use_cudagraph:
                for name, handle in state_dict.items():
                    func, args = handle
                    list_args = list(args)
                    # NOTE: the key is to change device id to the current device id
                    # in case two processes have different CUDA_VISIBLE_DEVICES
                    list_args[6] = torch.cuda.current_device()
                    new_weight = func(*list_args)

                    self.tp_worker.worker.model_runner.update_weights_from_tensor(
                        [(name, new_weight)], load_format="direct"
                    )
                    del new_weight

            else:
                named_params = dict(model.named_parameters())
                for name, handle in state_dict.items():
                    rebind_param_attr(model, name, self.binded_attr, materialize=False)
                    func, args = handle
                    list_args = list(args)
                    list_args[6] = torch.cuda.current_device()
                    new_weight = func(*list_args)
                    vllm_weight = named_params[name]
                    assert vllm_weight.shape == new_weight.shape, (
                        f"{name}: {vllm_weight.shape=}, {new_weight.shape=}"
                    )
                    assert vllm_weight.dtype == new_weight.dtype, (
                        f"{name}: {vllm_weight.dtype=}, {new_weight.dtype=}"
                    )

                    swap_tensor_pointer(vllm_weight, new_weight)
                    del new_weight

                for name, buffer in self.named_buffers.items():
                    vllm_buffer = get_module_from_name(model, name)
                    assert vllm_buffer.shape == buffer.shape
                    assert vllm_buffer.dtype == buffer.dtype
                    swap_tensor_pointer(vllm_buffer, buffer)

                self.named_buffers = {}

                self.tp_worker.worker.model_runner.token_to_kv_pool._create_buffers()
        else:
            # disaggregate mode, recv tensor directly
            named_tensors = [(n, p) for n, p in state_dict.items()]
            self.tp_worker.worker.model_runner.update_weights_from_tensor(
                named_tensors, load_format="direct"
            )
        self.sync_in_tp("sync_weight")

        return SyncWeightOutput()

    def run_task_method(self, obj: TaskMethodInput):
        """
        Run a CommTask method with the given name and arguments.
        NOTE: will call wait() if async_op is True.
        """
        result = getattr(self, obj.method_name)(*obj.args, **obj.kwargs)
        if "async_op" in obj.kwargs and obj.kwargs["async_op"]:
            result = result.wait()
        return TaskMethodOutput(method_name=obj.method_name, result=result)

    def abort_request(self, recv_req: AbortReq):
        # Compared to the original SGLang implementation, we will remove all requests that start with the given rid.
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in sorted(to_del, reverse=True):
            req = self.waiting_queue.pop(i)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete requests in the running batch
        for req in self.running_batch.reqs:
            if req.rid.startswith(recv_req.rid) and not req.finished():
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True


# only modifiy Scheduler's initialization parameters
def run_scheduler_process(
    parent_address: WorkerAddress,
    placement: ModelParallelComponentPlacement,
    config: DictConfig,
    world_size: int,
    rank: int,
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Generate the prefix
    if dp_rank is None:
        prefix = f" TP{tp_rank}"
    else:
        prefix = f" DP{dp_rank} TP{tp_rank}"
        dp_rank = None

    # Config the process
    # kill_itself_when_parent_died()  # This is disabled because it does not work for `--dp 2`
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            parent_address,
            placement,
            config,
            world_size,
            rank,
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            dp_rank,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )
        if scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
