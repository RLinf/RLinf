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


class MegatronInference(MegatronActor):
    """The class for running inference using Megatron.

    This class is only used for disaggregated mode, where the model is not trained in the same process as the inference.
    The inference model is loaded from the checkpoint, and sync weights with the training model after a iteration of training is done.
    """

    def __init__(
        self, cfg: DictConfig, placement: ComponentPlacement, role="inference"
    ):
        """Initialize the Megatron inference task.

        Args:
            cfg (DictConfig): Configuration for the inference task, including model parameters and other settings.
        """

        self.cfg = cfg
        self._build_inference_cfg()
        super().__init__(self.cfg, placement, role=role)
        self._iteration = 0

        # Actor information
        self._actor_group_name = self.cfg.actor.group_name
        self._weight_sync_actor_src_rank = self._rank
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
        ) and self.cfg.actor.get("combine_reference_model", True):
            ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
        self.ref_policy_state_dict = ref_policy_state_dict

        self._weight_dst_rank_in_inference = self.get_inference_weight_dst_ranks(
            self.cfg.inference.model.tensor_model_parallel_size,
            self.cfg.inference.model.pipeline_model_parallel_size,
        )

    def _build_inference_cfg(self):
        """Build the configuration for inference based on the actor config."""
        inference_cfg = self.cfg.inference
        actor_cfg = self.cfg.actor
        merged_cfg = copy.deepcopy(actor_cfg)
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
            self.cfg.inference = merged_cfg

    def sync_model_from_actor(self):
        if self.is_weight_offloaded:
            self.onload_model_weights_and_grad(load_grad=False)
            self.is_weight_offloaded = False
        for rank in self._weight_dst_rank_in_inference:
            if self._rank == rank:
                state_dict = self.recv(
                    src_group_name=self._actor_group_name,
                    src_rank=rank,
                )
                self.load_state_dict(state_dict, strict=False)

        for ddp_model in self.model:
            ddp_model.broadcast_params()

        self.log_debug("Inference sync_model_from_actor: resharding done")
