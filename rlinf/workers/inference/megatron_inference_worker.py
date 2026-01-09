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

import copy
from omegaconf import DictConfig, open_dict

from rlinf.utils.placement import ComponentPlacement
from rlinf.utils.utils import retrieve_model_state_dict_in_cpu

from ..actor.megatron_actor_worker import MegatronActor
from ..critic.megatron_critic_worker import MegatronCritic

def make_megatron_inference(base_class):
    class MegatronInference(base_class):
        """The class for running inference using Megatron.

        This class is only used for disaggregated mode, where the model is not trained in the same process as the inference.
        The inference model is loaded from the checkpoint, and sync weights with the training model after a iteration of training is done.
        """

        def __init__(
            self, cfg: DictConfig, placement: ComponentPlacement, train_role: str = ""
        ):
            """Initialize the Megatron inference task.

            Args:
                cfg (DictConfig): Configuration for the inference task, including model parameters and other settings.
            """

            self.cfg = cfg
            self.train_role = train_role if len(train_role) > 0 else 'actor'
            self.train_role_cfg = getattr(self.cfg, self.train_role)
            self.role = '_'.join([train_role, "inference"])
            self.role_cfg = getattr(self.cfg, self.role)

            self._build_inference_cfg()
            super().__init__(self.cfg, placement, role=train_role)
            self._iteration = 0

            # Actor information
            self._train_group_name = self.train_role_cfg.group_name
            self._weight_sync_train_src_rank = self._rank
            self.offload_weight = False
            self.offload_optimizer = False

        def init_worker(self):
            self.setup_model_and_optimizer()
            self.optimizer, self.lr_scheduler = None, None

            ref_policy_state_dict = None
            # only need this if we are running with inital kl penalty & full-parameter tuning
            if (
                self.cfg.algorithm.kl_beta > 0
                or self.cfg.algorithm.get("reinpp_kl_beta", 0) > 0
            ) and self.train_role_cfg.get("combine_reference_model", True):
                ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
            self.ref_policy_state_dict = ref_policy_state_dict

            self._weight_dst_rank_in_inference = self.get_inference_weight_dst_ranks(
                self.role_cfg.model.tensor_model_parallel_size,
                self.role_cfg.model.pipeline_model_parallel_size,
            )

        def _build_inference_cfg(self):
            """Build the configuration for inference based on the actor config."""
            inference_cfg = self.role_cfg
            train_cfg = self.train_role_cfg
            merged_cfg = copy.deepcopy(train_cfg)
            with open_dict(merged_cfg):
                # Override with inference configs
                merged_cfg.group_name = inference_cfg.group_name
                merged_cfg.load_from_actor = inference_cfg.load_from_actor
                merged_cfg.model.tensor_model_parallel_size = (
                    inference_cfg.model.tensor_model_parallel_size
                )
                merged_cfg.model.pipeline_model_parallel_size = (
                    inference_cfg.model.pipeline_model_parallel_size
                )
                merged_cfg.model.sequence_parallel = inference_cfg.model.sequence_parallel

            with open_dict(self.cfg):
                self.role_cfg = merged_cfg

        def sync_model_from_actor(self):
            # sync_from_actor_start_time = time.time()
            if self.is_weight_offloaded:
                self.onload_model_weights_and_grad(load_grad=False)
                self.is_weight_offloaded = False
            for rank in self._weight_dst_rank_in_inference:
                if self._rank == rank:
                    state_dict = self.recv(
                        src_group_name=self._train_group_name,
                        src_rank=rank,
                    )
                    self.load_state_dict(state_dict, strict=False)

            for ddp_model in self.model:
                ddp_model.broadcast_params()

            self.log_debug("Inference sync_model_from_actor: resharding done")

    return MegatronInference

MegatronInference = make_megatron_inference(MegatronActor)
MegatronCriticInference = make_megatron_inference(MegatronCritic)
