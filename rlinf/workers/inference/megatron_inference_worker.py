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

import contextlib
import copy
import time
from typing import Dict

import torch
from megatron.core import mpu
from omegaconf import DictConfig, open_dict

from rlinf.data.io_struct import RolloutResult
from rlinf.utils.placement import ComponentPlacement
from rlinf.utils.utils import (
    clear_memory,
)

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

        # Inference configuration
        self._response_len = (
            self.cfg.inference.model.encoder_seq_length
            - self.cfg.data.max_prompt_length
        )
        self._logprob_forward_micro_batch_size = (
            self.cfg.algorithm.logprob_forward_micro_batch_size
        )
        self._enable_dynamic_batch_size = self.cfg.runner.enable_dynamic_batch_size
        self._max_tokens_per_mbs = self.cfg.runner.max_tokens_per_mbs

        # Actor information
        self._actor_group_name = self.cfg.actor.group_name
        self._weight_sync_actor_src_rank = self._rank

        # Dataserver setup
        self._data_server = self.connect_channel(
            channel_name=self.cfg.actor.channel.name
        )
        self._output_queue_name: str = self.cfg.inference.channel.queue_name
        self._input_queue_name: str = (
            f"{self._output_queue_name}_{mpu.get_data_parallel_rank()}"
        )
        self._data_server.create_queue(self._output_queue_name)

        # Inference state
        self._timer_names = []
        self._is_model_rank_0 = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        )
        self._num_rollout_results_per_step = (
            self.cfg.data.rollout_batch_size
            // self.component_placement.inference_dp_size
        )

    def init_worker(self):
        self.setup_model_and_optimizer()
        self.optimizer, self.lr_scheduler = None, None

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
            merged_cfg.channel = inference_cfg.channel
            merged_cfg.load_from_actor = inference_cfg.load_from_actor
            merged_cfg.model.tensor_model_parallel_size = (
                inference_cfg.model.tensor_model_parallel_size
            )
            merged_cfg.model.pipeline_model_parallel_size = (
                inference_cfg.model.pipeline_model_parallel_size
            )
            merged_cfg.model.sequence_parallel = inference_cfg.model.sequence_parallel
            merged_cfg.megatron.transformer_impl = "transformer_engine"
            merged_cfg.megatron.swiglu = True
            merged_cfg.megatron.untie_embeddings_and_output_weights = True
            merged_cfg.megatron.padded_vocab_size = merged_cfg.model.override_vocab_size
            merged_cfg.megatron.make_vocab_size_divisible_by = (
                merged_cfg.model.make_vocab_size_divisible_by
            )

            if self.cfg.inference.load_from_actor:
                merged_cfg.megatron.load = None
                merged_cfg.megatron.pretrained_checkpoint = None

        with open_dict(self.cfg):
            self.cfg.inference = merged_cfg

    def _log_on_model0(self, message: str, debug: bool = False):
        if self._is_data_io_rank:
            if debug:
                self.log_debug(message)
            else:
                self.log_info(message)

    @contextlib.contextmanager
    def _timer(self, name: str):
        """Context manager for timing code execution."""
        start_time = time.perf_counter_ns()
        yield
        end_time = time.perf_counter_ns()
        elapsed_time = (end_time - start_time) / 1e9
        self._log_on_model0(
            f"Inference iter {self._iteration % self._num_rollout_results_per_step}: {name} took {elapsed_time:.2f} seconds",
            debug=True,
        )
        timer_collection_name = f"_{name}_timer"
        if not hasattr(self, timer_collection_name):
            setattr(self, timer_collection_name, [])
            self._timer_names.append(timer_collection_name)
        getattr(self, timer_collection_name).append(elapsed_time)

    def _clear_timers(self):
        """Clear the timers after each iteration."""
        for timer_name in self._timer_names:
            setattr(self, timer_name, [])

    def _inference(
        self, rollout_result: RolloutResult, should_compute_ref_logprobs: bool
    ) -> Dict:
        """Run inference on the given rollout result."""
        rollout_batch = rollout_result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.runner.seq_length,
            pad_token=self.tokenizer.eos_token_id,
        )

        batch = {
            "input_ids": rollout_batch["input_ids"],
            "attention_mask": rollout_batch["attention_mask"],
            "position_ids": rollout_batch["position_ids"],
        }
        # Compute the logprobs for the inference policy.
        rollout_batch["prev_logprobs"] = self.inference_step(batch)

        # TODO: support calculate ref logprobs
        # Compute the reference policy logprobs if needed.
        # if should_compute_ref_logprobs:
        #     rollout_batch["ref_logprobs"] = self.compute_ref_logprobs(batch)
        clear_memory()

        return rollout_batch

    def sync_model_from_actor(self):
        for rank in self._weight_dst_rank_in_inference:
            if self._rank == rank:
                state_dict = self.recv(
                    src_group_name=self._actor_group_name,
                    src_rank=rank,
                )
                self.load_state_dict(state_dict, strict=False)

        for ddp_model in self.model:
            ddp_model.broadcast_params()

        self.log_info("Inference sync_model_from_actor: resharding done")

    def _recv_rollout_results(self):
        """Receive rollout results."""
        if self._is_data_io_rank:
            recv_result: RolloutResult = self._data_server.get(
                queue_name=self._input_queue_name
            )
            torch.distributed.broadcast_object_list(
                [recv_result],
                src=mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
        else:
            res = [None]
            torch.distributed.broadcast_object_list(
                res,
                src=mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            recv_result: RolloutResult = res[0]

        return recv_result

    def run_inference(self, should_compute_ref_logprobs: bool = False):
        """Run inference in a pipeline manner.

        This is used for disaggregated mode where the model is not trained in the same process as the inference.
        """
        for _ in range(self._num_rollout_results_per_step):
            with self._timer("inference_iter"):
                with self._timer("recv_rollout_results"):
                    recv_result = self._recv_rollout_results()

                with self._timer("inference"):
                    rollout_batch = self._inference(
                        recv_result, should_compute_ref_logprobs
                    )

                if self._is_data_io_rank:
                    with self._timer("put_inference_results"):
                        recv_result.prev_logprobs = rollout_batch["prev_logprobs"].cpu()
                        if rollout_batch.get("ref_logprobs", None) is not None:
                            recv_result.ref_logprobs = rollout_batch[
                                "ref_logprobs"
                            ].cpu()
                        self._data_server.put(
                            recv_result, queue_name=self._output_queue_name
                        )

                self._iteration += 1

            if self._iteration % self._num_rollout_results_per_step == 0:
                self._log_on_model0(
                    f"Processed {self._iteration // self._num_rollout_results_per_step} * {self._num_rollout_results_per_step} batches in inference pipeline. Sync weight from actor."
                )
                self._clear_timers()
