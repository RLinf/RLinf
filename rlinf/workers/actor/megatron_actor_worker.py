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

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.training.training import unwrap_model
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig
from torch.multiprocessing.reductions import reduce_tensor

from rlinf.algorithms.registry import policy_loss
from rlinf.algorithms.utils import kl_penalty

from rlinf.utils.distributed import (
    vocab_parallel_entropy_and_log_probs,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.resharding.mcore_weight_reshard import MegatronCoreWeightReshard
from rlinf.utils.resharding.reshard_config import ReshardConfig
from rlinf.utils.utils import retrieve_model_state_dict_in_cpu
from rlinf.workers.rollout.utils import RankMapper
from rlinf.workers.megatron_worker import MegatronWorker

try:
    from params_resharding import nccl_group_recreate, resharding_init

    HAVE_RESHARDING = True
except ImportError:
    HAVE_RESHARDING = False


class MegatronActor(MegatronWorker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role='actor'
    ):
        """Initialize the MegatronWorker.

        Args:
            cfg (DictConfig): The configuration for the actor.
        """
        super().__init__(cfg, placement, role)

        self.dst_tp_rank = self._rank % placement.rollout_tp_size
        assert placement.rollout_tp_size <= placement.actor_tp_size, (
            f" rollout tensor parallel size {placement.rollout_tp_size} must be less than or equal to actor tensor parallel size {placement.actor_tp_size}."
        )

        # Algo configurations
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        clip_ratio = self.cfg.algorithm.ratio_clip_eps
        self.clip_ratio_low = (
            self.cfg.algorithm.get("clip_ratio_low")
            if self.cfg.algorithm.get("clip_ratio_low") is not None
            else clip_ratio
        )
        self.clip_ratio_high = (
            self.cfg.algorithm.get("clip_ratio_high")
            if self.cfg.algorithm.get("clip_ratio_high") is not None
            else clip_ratio
        )

        self.clip_ratio_c = self.cfg.algorithm.clip_ratio_c

        self.ref_policy_state_dict = None

        # Rollout configurations
        self.rollout_group_name = self.cfg.rollout.group_name

    def init_worker(self):

        super().init_worker()

        # only need this if we are running with inital kl penalty & full-parameter tuning
        if (
            self.cfg.algorithm.kl_beta > 0
            or self.cfg.algorithm.get("reinpp_kl_beta", 0) > 0
        ) and self.cfg.actor.get("combine_reference_model", True):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
            self.offload_model_buffer = {}

        rollout_reshard_config = ReshardConfig(
            model_type=self.cfg.rollout.model.model_type,
            model_config=self.transformer_config,
            reshard_tp_size=self.cfg.rollout.tensor_parallel_size,
            reshard_pp_size=self.cfg.rollout.pipeline_parallel_size,
            mg_ep_size=self.role_cfg.model.expert_model_parallel_size,
            mg_tpe_size=self.role_cfg.model.expert_tensor_parallel_size,
            moe_grouped_gemm=self.role_cfg.model.get("moe_grouped_gemm", None),
        )
        self.rollout_weights_reshard = MegatronCoreWeightReshard(rollout_reshard_config)
        self._setup_rollout_weight_dst_ranks()

    def get_forward_step_func(self):
        """Acquire the forward step function for the model."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            batch = {key: val.cuda() for key, val in batch.items()}

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            response_len = self.response_len
            responses = input_ids[:, -response_len:]
            label = copy.deepcopy(position_ids)
            label[:, -response_len - 1 : -1] = responses
            label_mask = copy.deepcopy(attention_mask)
            label_mask[:, : -response_len - 1] = False
            label_mask[:, -1] = False

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape

                if self.calculate_entropy:
                    entropy, log_probs = vocab_parallel_entropy_and_log_probs(
                        logits,
                        label,
                        calculate_entropy_loss=self.calculate_entropy_loss,
                    )
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret = {"log_probs": log_probs, "entropy": entropy}
                else:
                    log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret = {"log_probs": log_probs}

                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}

            output = self.custom_forward(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                temperature=self.cfg.algorithm.sampling_params.temperature,
            )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output

                # in last stage need to get the log_probs from the output
                if unwrap_model(model).post_process:
                    mask = batch["response_mask"][:, -response_len:]
                    output = output["log_probs"][:, -response_len - 1 : -1].contiguous()
                    output = output * mask

                return output, id_func

            def loss_func(output):
                curr_logprobs = output["log_probs"][
                    :, -response_len - 1 : -1
                ].contiguous()

                advantages = batch["advantages"]
                prev_logprobs = batch["prev_logprobs"]
                ref_logprobs = None
                if "ref_logprobs" in batch:
                    ref_logprobs = batch["ref_logprobs"]

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                mask = batch["response_mask"][:, -response_len:]

                loss, metrics_data = policy_loss(
                    task_type=self.cfg.runner.task_type,
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=curr_logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_c=self.clip_ratio_c,
                    clip_ratio_low=self.clip_ratio_low,
                    clip_ratio_high=self.clip_ratio_high,
                    loss_mask=mask,
                )

                entropy_loss = torch.zeros(1, device=loss.device)
                if self.calculate_entropy:
                    entropy = output["entropy"][:, -response_len - 1 : -1].contiguous()
                    entropy_loss = self.loss_agg_func(entropy, mask=mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(curr_logprobs, ref_logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, mask)
                    loss = loss + kl_loss * self.kl_beta

                # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
                _imp = metrics_data["actor/ratio"]
                torch.distributed.all_reduce(
                    _imp, group=parallel_state.get_data_parallel_group()
                )
                _n_valid_tokens = mask.count_nonzero().clone()
                torch.distributed.all_reduce(
                    _n_valid_tokens, group=parallel_state.get_data_parallel_group()
                )
                _imp /= _n_valid_tokens

                # Early stopping.
                if (
                    self.cfg.algorithm.early_stop_imp_ratio is not None
                    and _imp > self.cfg.algorithm.early_stop_imp_ratio
                ):
                    self.log_warning(
                        f"Current importance ratio {_imp.item():.4f} is larger "
                        f"than early stop threshold {self.cfg.algorithm.early_stop_imp_ratio}. Abandon this microbatch."
                    )
                    loss = loss * 0.0

                if self.cfg.algorithm.use_valid_token_scale:
                    loss_scale = (
                        mask.sum()
                        / self.global_valid_token
                        * parallel_state.get_data_parallel_world_size()
                        * self.num_microbatches
                    )
                    loss *= loss_scale.item()

                # add to log
                metrics_data.update(
                    {
                        "actor/final_loss": loss.detach(),
                        "actor/entropy_loss": entropy_loss.detach(),
                        "actor/kl_loss": kl_loss.detach(),
                    }
                )

                for k, v in metrics_data.items():
                    if v is not None:
                        metrics_data[k] = average_losses_across_data_parallel_group([v])

                return loss, metrics_data

            return output, loss_func

        return forward_output_and_loss_func

    def _get_rollout_model_state_dict(self, bucket_weight):
        """Get the state dictionary of the model for rollout."""
        return self.rollout_weights_reshard.gather_and_reshard_model(
            bucket_weight, self.dst_tp_rank
        )

    def _setup_rollout_weight_dst_ranks(self):
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self.component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def divide_model_to_bucket(self):
        model_bucket_list = self.rollout_weights_reshard.divide_model_to_bucket(
            self.model
        )
        return model_bucket_list

    def sync_model_to_rollout(self):
        """Send the model weights to the destination ranks in the rollout task."""
        if self.recreate_nccl_groups:
            nccl_group_recreate()
        if not self.is_running:
            return

        # ensure weights are on GPU before reshard
        self._load_weight()

        model_bucket_list = self.divide_model_to_bucket()
        if not hasattr(self, "sync_model_bucket_length"):
            self.sync_model_bucket_length = len(model_bucket_list)
        else:
            assert self.sync_model_bucket_length == len(model_bucket_list), (
                f"last sync_model_bucket_length {self.sync_model_bucket_length} don't equal now the len(model_bucket_list) {len(model_bucket_list)}"
            )
            assert self.sync_model_bucket_length != 0, (
                "error the self.sync_model_bucket_length is 0"
            )

        self.model_state_offload_optimizer_and_grad()

        # send bucket size
        if len(self._weight_dst_rank_in_rollout) > 0:
            if self.placement_mode == PlacementMode.COLLOCATED:
                send_handle = None
                for bucket_weight in model_bucket_list:
                    reshard_state_dict = self._get_rollout_model_state_dict(
                        bucket_weight
                    )
                    buffer = {
                        k: reduce_tensor(v) for k, v in reshard_state_dict.items()
                    }
                    if send_handle is not None:
                        send_handle.wait()
                    else:
                        # add the bucket_length message in bucket 0
                        buffer["bucket_length"] = len(model_bucket_list)
                    send_handle = self.send(
                        buffer,
                        self.rollout_group_name,
                        self._weight_dst_rank_in_rollout,
                        async_op=True,
                    )
                    del reshard_state_dict
                send_handle.wait()
            else:
                send_handle_bucket = []
                for bucket_weight in model_bucket_list:
                    reshard_state_dict = self._get_rollout_model_state_dict(
                        bucket_weight
                    )

                    if len(send_handle_bucket) != 0:
                        for send_handle in send_handle_bucket:
                            send_handle.wait()
                        send_handle_bucket = []
                    else:
                        # add the bucket_length message in bucket 0
                        reshard_state_dict["bucket_length"] = len(model_bucket_list)

                    for weight_dst_rank in self._weight_dst_rank_in_rollout:
                        send_handle = self.send(
                            reshard_state_dict,
                            self.rollout_group_name,
                            weight_dst_rank,
                            async_op=True,
                        )
                        send_handle_bucket.append(send_handle)

                if len(send_handle_bucket) != 0:
                    for send_handle in send_handle_bucket:
                        send_handle.wait()

        if (
            self.placement_mode == PlacementMode.COLLOCATED
            or self.use_pre_process_policy
        ):
            if self.offload_weight:
                self.offload_model_weights_and_grad(
                    offload_grad=False, offload_weight=True
                )
                self.is_weight_offloaded = True

    def model_state_offload_optimizer_and_grad(self):
        if not self.is_running:
            return
        if (
            self.placement_mode == PlacementMode.COLLOCATED
            or self.use_pre_process_policy
        ):
            if self.offload_optimizer:
                self.offload_megatron_optimizer()
                self.is_optimizer_offloaded = True
            self.offload_model_weights_and_grad(
                offload_grad=self.offload_grad, offload_weight=False
            )
        else:
            assert self.placement_mode in [
                PlacementMode.DISAGGREGATED,
                PlacementMode.AUTO,
            ], "Unsupported placement mode for sending weights."
            assert isinstance(self._weight_dst_rank_in_rollout, list), (
                f"In disaggregated mode, weight_dst_rank_in_rollout should be a list of ranks, got {type(self._weight_dst_rank_in_rollout)}"
            )
