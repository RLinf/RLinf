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
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import retrieve_model_state_dict_in_cpu
from rlinf.workers.actor.fsdp_actor_worker import FSDPActor


class FSDPInference(FSDPActor):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self._actor_group_name = cfg.actor.group_name

        # algorithms
        self.kl_beta = cfg.algorithm.get("kl_beta", 0)
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

    def init_worker(self) -> None:
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

    def sync_model_from_actor(self) -> None:
        if self._rank == 0:
            state_dict = self.recv(
                src_group_name=self._actor_group_name,
                src_rank=0,
            )
        else:
            state_dict = None

        option = StateDictOptions(
            cpu_offload=False, full_state_dict=True, broadcast_from_rank0=True
        )
        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=option,
        )
