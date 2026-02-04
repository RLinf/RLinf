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
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.tensor import DTensor

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import (
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.actor.fsdp_actor_worker import FSDPActor


class FSDPInference(FSDPActor):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        """
        FSDP Inference worker used in pipeline mode.

        Args:
            cfg (DictConfig): The global yaml config.
            placement (ModelParallelComponentPlacement): The accelerator placement for inference worker.
        """
        super().__init__(cfg, placement, cfg.inference)

        self._actor_group_name = cfg.actor.group_name
        self._actor_world_size = self._component_placement.get_world_size("actor")
        # here key is actor ranks, value is dict of param name to (actor_shard_offset,inference_shard_offset,needed_size)
        self._actor_dst_map: dict[int, dict[str, tuple[int, int, int]]] = {}

    def init_worker(self) -> None:
        """
        Init the FSDP inference worker. It will build the model and use
        FSDP to wrap it. If needed, it will also retrieve the reference model's state_dict from CPU.
        And finally, it will determine which actor ranks will send their params to this inference rank
        by do a All-to-All handshake, swapping their shard tensors' metadata.
        """
        # create and wrap model with FSDP's strategy
        model = self.model_provider_func()
        self.model = self._strategy.wrap_model(
            model=model, device_mesh=self._device_mesh
        )

        # Get Ref model if needed.
        ref_policy_state_dict = None
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
        self.ref_policy_state_dict = ref_policy_state_dict

    @torch.no_grad()
    def load_from_actors_by_intersection(
        self, cur_state_dict: dict[str, torch.Tensor | DTensor | ShardedTensor]
    ) -> None:
        """
        Synchronize the model weights from actor workers to the inference workers according former All-to-All
        handshake with actor workers.

        Args:
            cur_state_dict(dict[str, torch.Tensor|DTensor|ShardedTensor]): The current rank's state_dict to be updated.
        """

        if not self._actor_dst_map:
            self._strategy.setup_inference_sync_actor_ranks(self)

        needed_actor_ranks = list(self._actor_dst_map.keys())
        receiving_jobs = [
            self.recv(
                src_rank=rank, src_group_name=self._actor_group_name, async_op=True
            )
            for rank in needed_actor_ranks
        ]
        received_state_dicts: list[dict[str, torch.Tensor]] = [
            job.wait() for job in receiving_jobs
        ]

        for k, cur_tensor in cur_state_dict.items():
            inference_local = (
                cur_tensor.to_local() if isinstance(cur_tensor, DTensor) else cur_tensor
            )

            inference_flat = inference_local.view(-1)

            for actor_rank, src_dict in zip(needed_actor_ranks, received_state_dicts):
                # ranks is setup in setup_inference_sync_actor_ranks, so
                # every src_dict should contain k, or need to check implementation.
                assert k in src_dict, (
                    f"Key {k} not found in received state dict from actors."
                )
                actor_flat = src_dict[k].view(-1)

                assert actor_rank in self._actor_dst_map, (
                    f"Actor rank {actor_rank} not found in actor_dst_map."
                )
                assert k in self._actor_dst_map[actor_rank], (
                    f"Key {k} not found in actor_dst_map for actor rank {actor_rank}."
                )
                actor_shard_off, inference_shard_off, need_size = self._actor_dst_map[
                    actor_rank
                ][k]
                inference_flat[
                    inference_shard_off : inference_shard_off + need_size
                ].copy_(
                    actor_flat[actor_shard_off : actor_shard_off + need_size],
                    non_blocking=True,
                )

        torch.cuda.synchronize()
        torch.distributed.barrier()

    def sync_model_from_actor(self) -> None:
        """
        Sync the model weights from actor workers to the inference workers.
        In former All-to-All setup communication, each inference rank only receives needed shards from actor ranks.
        So here we first get the current rank's state_dict, then load the needed shards from actor ranks,
        and then set the updated state_dict back to the model.
        """
        opts = StateDictOptions(cpu_offload=False, full_state_dict=False)
        current_rank_state_dict = get_model_state_dict(model=self.model, options=opts)
        self.load_from_actors_by_intersection(cur_state_dict=current_rank_state_dict)
        set_model_state_dict(
            model=self.model, model_state_dict=current_rank_state_dict, options=opts
        )
