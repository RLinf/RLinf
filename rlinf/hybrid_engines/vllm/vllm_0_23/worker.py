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

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker

from rlinf.scheduler import Worker as _RLinfWorker
from rlinf.utils.placement import PlacementMode
from rlinf.workers.rollout.utils import RankMapper

from . import weight_loader  # noqa all
from .engine_args import RLinfEngineArgs

logger = init_logger(__name__)


class VLLMWorker(_VllmInnerWorker):
    """vllm worker with RLinf weight synchronisation.

    Unlike the vllm 0.8.5 variant, this takes vllm's own constructor arguments
    only; the RLinf arguments arrive through ``vllm_config.additional_config``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        rlinf_args = vllm_config.additional_config
        assert isinstance(rlinf_args, RLinfEngineArgs), (
            f"vllm_config.additional_config must be RLinfEngineArgs, got {type(rlinf_args)}"
        )
        self.rlinf_config = rlinf_args.rlinf_config
        self.using_sharded_weight = (
            False if self.rlinf_config.actor.training_backend == "fsdp" else True
        )
        self._rlinf_worker = _RLinfWorker(
            parent_address=rlinf_args.parent_address,
            world_size=vllm_config.parallel_config.world_size,
            rank=rank,
        )
        self._actor_group_name = self.rlinf_config.actor.group_name
        self.placement_mode = rlinf_args.placement.placement_mode
        rank_map = RankMapper.get_rollout_rank_to_actor_rank_map(
            placement=rlinf_args.placement
        )
        self.actor_weight_rank = rank_map[
            self._rlinf_worker.get_parent_rank(), self.rank
        ]
        self.is_weight_offloaded = False
        self.offloaded_tags = {"weights": False, "kv_cache": False}

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
        self.offloaded_tags = {"weights": True, "kv_cache": True}
        self.is_weight_offloaded = True

    def batch_load_hf_weight(self, state_dict: dict[str, Any]) -> Any:
        model = self.model_runner.model
        colocate = self.placement_mode == PlacementMode.COLLOCATED
        batch_weight = []
        if colocate:
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                # NOTE: the key is to change device id to the current device id
                # in case two processes have different CUDA_VISIBLE_DEVICES
                list_args[6] = torch.cuda.current_device()
                new_weight = func(*list_args)
                batch_weight.append((name, new_weight))
            model.load_weights(batch_weight)
        else:
            # disaggregate mode, recv tensor directly
            model.load_weights(state_dict.items())

        for name, weight in batch_weight:
            del weight
        batch_weight.clear()

    def sync_hf_weight(self) -> None:
        use_cudagraph = not self.rlinf_config.rollout.enforce_eager
        assert use_cudagraph, "use_cudagraph must be True now."

        state_dict = self._rlinf_worker.recv(
            src_group_name=self._actor_group_name,
            src_rank=self.actor_weight_rank,
        )

        bucket_length = state_dict.get("bucket_length", None)

        if bucket_length is None:
            # recv from the fsdp backend
            # fsdp just send a bucket and don't have the key bucket_length
            bucket_length = 1
        else:
            # recv from the Megatron backend
            # Megatron use weight bucket to sync weight, the bucket length in dict of bucket 0, bucket_length
            state_dict.pop("bucket_length")
            if isinstance(bucket_length, torch.Tensor):
                # The actor sends bucket_length as a GPU tensor
                bucket_length = bucket_length.item()

        if self.is_weight_offloaded:
            # Only wake weights here; KV cache + cudagraph stay offloaded until onload_kv_cudagraph
            super().wake_up(tags=["weights"])
            self.offloaded_tags["weights"] = False
            self.is_weight_offloaded = False

        assert bucket_length > 0, f"bucket_length {bucket_length} is invalid"

        self.batch_load_hf_weight(state_dict)
        if bucket_length > 1:
            recv_handle = self._rlinf_worker.recv(
                src_group_name=self._actor_group_name,
                src_rank=self.actor_weight_rank,
                async_op=True,
            )

            for _ in range(bucket_length - 2):
                next_recv_handle = self._rlinf_worker.recv(
                    src_group_name=self._actor_group_name,
                    src_rank=self.actor_weight_rank,
                    async_op=True,
                )
                state_dict = recv_handle.wait()
                self.batch_load_hf_weight(state_dict)
                recv_handle = next_recv_handle

            state_dict = recv_handle.wait()
            self.batch_load_hf_weight(state_dict)

    def onload_kv_cudagraph(self) -> None:
        """Onload KV cache + cuda graph deferred from sync_hf_weight.
        sync_hf_weight only woke weights for load_weights;
        KV cache + cudagraph are restored here, vLLM has no cuda_graph tag, so the
        cuda graph is re-captured via compile_or_warm_up_model.
        """
        assert self.offloaded_tags["kv_cache"], (
            "Cannot onload kv_cache: not offloaded (already onloaded). "
            f"offload_state={self.offloaded_tags}"
        )
        super().wake_up(tags=["kv_cache"])
        self.offloaded_tags["kv_cache"] = False
        super().compile_or_warm_up_model()

    def use_sharded_weights(self) -> None:
        model = self.model_runner.model
        for _, param in model.named_parameters():
            setattr(param, "is_sharded_weight", self.using_sharded_weight)

    def get_dp_rank(self) -> int:
        return self._rlinf_worker.get_parent_rank()
