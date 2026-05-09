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

from rlinf.utils.placement import (
    ComponentPlacement,
    PlacementMode,
)
from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
from rlinf.utils.resharding.reshard_config import ReshardConfig
from rlinf.utils.utils import retrieve_model_state_dict_in_cpu, clear_memory

from ..actor.megatron_actor_worker import MegatronActor, enable_ptrace


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
        # enable_ptrace()
        self.setup_model_and_optimizer()
        self.offload_megatron_optimizer()
        self.optimizer, self.lr_scheduler = None, None
        if self.use_auto_scheduler and self.cfg.cluster.rollout_use_all_gpus:
            self.offload_model_weights_and_grad(offload_weight=False, offload_grad=True)
        clear_memory()
        self.cuda_info("after set opt to None")

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

        rollout_reshard_config = ReshardConfig(
            model_arch=self.cfg.rollout.model_arch,
            model_config=self.transformer_config,
            reshard_tp_size=self.cfg.rollout.tensor_parallel_size,
            reshard_pp_size=self.cfg.rollout.pipeline_parallel_size,
        )
        self.rollout_weights_reshard = MegatronCoreWeightReshard(rollout_reshard_config)
        self._setup_rollout_weight_dst_ranks()

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
            self.log_on_first_rank("before sync from actor, onload weight...")
            self.onload_model_weights_and_grad(load_grad=False)
            self.is_weight_offloaded = False
        for rank in self._weight_dst_rank_in_inference:
            if self._rank == rank:
                state_dict = self.recv(
                    src_group_name=self._actor_group_name,
                    src_rank=rank,
                )
                self.load_state_dict(state_dict, strict=False)

        if len(self._weight_dst_rank_in_inference) < self.component_placement.inference_world_size:
            self.log_info("boradcast parameters among inference dp...")
            for ddp_model in self.model:
                ddp_model.broadcast_params()

        self.log_debug("Inference sync_model_from_actor: resharding done")

    def get_model_state_and_offload(self):
        """Send the model weights to the destination ranks in the rollout task.

        When in COLLOCATED mode or when `use_pre_process_policy` is True, first offload the optimizer and gradients.
        Then call `_get_rollout_model_state_dict()`, and finally offload the model weights.
        """
        if not self.is_running:
            return

        assert self.component_placement._placement_mode in [
            PlacementMode.DISAGGREGATED,
            PlacementMode.AUTO,
        ], "Unsupported placement mode for sending weights."
        assert isinstance(self._weight_dst_rank_in_rollout, list), (
            f"In disaggregated mode, weight_dst_rank_in_rollout should be a list of ranks, got {type(self._weight_dst_rank_in_rollout)}"
        )
        self.reshard_state_dict = self._get_rollout_model_state_dict()

        if self.use_auto_scheduler and self.use_pre_process_policy:
            if self.cfg.cluster.rollout_use_all_gpus:
                self.offload_model_weights_and_grad()
            clear_memory()

    def sync_model_to_rollout(self):
        """Send the model weights to the destination ranks in the rollout task."""
        # if self.recreate_nccl_groups:
        #     nccl_group_recreate()
        if not self.is_running:
            return
        self.get_model_state_and_offload()

        self.cuda_info("inference: before send to rollout")
        assert (
            not self.component_placement._placement_mode == PlacementMode.COLLOCATED
        ), (
            "Inference Worker's sync_model_to_rollout() should not be called under collocated mode."
        )
        for weight_dst_rank in self._weight_dst_rank_in_rollout:
            self.send(
                self.reshard_state_dict,
                self.rollout_group_name,
                weight_dst_rank,
            )
