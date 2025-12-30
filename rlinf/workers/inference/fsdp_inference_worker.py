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

from rlinf.data.io_struct import RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.scheduler import Worker
from rlinf.scheduler.channel import Channel
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import (
    compute_logprobs_from_logits,
    cpu_weight_swap,
    retrieve_model_state_dict_in_cpu,
)


class FSDPInference(FSDPModelManager, Worker):
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
        Worker.__init__(self)
        super().__init__(cfg.inference, self._world_size, self._rank)
        self.cfg = cfg
        self._actor_group_name = cfg.actor.group_name
        self._component_placement = placement
        # algorithms
        self.kl_beta = cfg.algorithm.get("kl_beta", 0)
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )
        self._actor_world_size = self._component_placement.get_world_size("actor")
        # here key is actor ranks, value is dict of param name to (actor_shard_offset,inference_shard_offset,needed_size)
        self._actor_dst_map: dict[int, dict[str, tuple[int, int, int]]] = {}
        # logprobs computation op type("torch", "flash_attn", "liger_kernel")
        self._entropy_op_type = cfg.algorithm.get("entropy_op_type", "liger_kernel")

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

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(
            logits, responses, op_type=self._entropy_op_type
        )
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
