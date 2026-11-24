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

import torch
from megatron.core import parallel_state
from megatron.training.training import unwrap_model
from omegaconf import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.schema.actor_batch import DynamicActorBatch
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.hybrid_engines.megatron.utils import (
    postprocess_ragged_seqs,
    preprocess_ragged_seqs,
)
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics_dynamic,
    masked_stats,
    normalize_from_stats,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import clear_memory, configure_batch_sizes, cpu_dict
from rlinf.workers.critic.megatron_critic_worker import MegatronCritic
from rlinf.workers.megatron_worker import DynamicMegatronBatchMixin


class MAMegatronCritic(DynamicMegatronBatchMixin, MegatronCritic):
    """Megatron value inference and updates for complete multi-turn agent episodes."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        role: str = "critic",
    ) -> None:
        super().__init__(cfg, placement, role)
        assert cfg.agentloop.is_dynamic_rollout_batch and not self.is_pipeline, (
            "Multi-agent critic requires dynamic rollout batches and synchronous execution."
        )
        assert self.enable_dp_load_balance, (
            "Multi-agent critic requires enable_dp_load_balance=True."
        )
        self.pack_traj = cfg.actor.get("pack_traj", True)

    def get_forward_step_func(self):
        """Align full-sequence state values with generated action tokens."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            rows = [row.to(self.torch_device_type) for row in batch["input_ids"]]
            lengths = list(map(len, rows))
            input_ids, packed_params = preprocess_ragged_seqs(rows)
            position_ids, _ = preprocess_ragged_seqs(
                [torch.arange(n, device=input_ids.device) for n in lengths]
            )
            output = self.custom_forward(
                model,
                input_ids,
                None,
                position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                packed_seq_params=packed_params,
            )
            if unwrap_model(model).post_process:
                value_rows = [
                    torch.nn.functional.pad(row.squeeze(-1).float()[:-1], (1, 0))
                    for row in postprocess_ragged_seqs(output, packed_params, lengths)
                ]
                if not self.return_loss:
                    values = [
                        (row * mask.to(row.device)).detach().cpu()
                        for row, mask in zip(value_rows, batch["response_mask"])
                    ]
                    return output, lambda output, non_loss_data=True: values
                width = self.role_cfg.model.encoder_seq_length
                output = DynamicRolloutResult.materialize_actor_batch(
                    {"values": value_rows}, width, 0, input_ids.device
                )["values"]
                batch = DynamicRolloutResult.materialize_actor_batch(
                    {key: batch[key] for key in ("values", "returns", "response_mask")},
                    width,
                    0,
                    input_ids.device,
                )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output

                return output, id_func

            def loss_func(output):
                return self._compute_value_loss(output, batch, batch["response_mask"])

            return output, loss_func

        return forward_output_and_loss_func

    def _setup_valid_token_scale(self, batch: DynamicActorBatch) -> None:
        self.global_valid_token = torch.tensor(
            sum(int(row.sum()) for row in batch["response_mask"]),
            device=self.torch_device_type,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            self.global_valid_token, group=parallel_state.get_data_parallel_group()
        )

    def _compute_rollout_metrics(self, batch: DynamicActorBatch):
        group = self.get_rollout_metrics_group(batch)
        if group is None:
            return None
        metrics, _, _ = compute_rollout_metrics_dynamic(
            batch, self.cfg.data.max_prompt_length, self.response_len, group
        )
        return cpu_dict(metrics)

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, batch: DynamicActorBatch
    ) -> DynamicActorBatch:
        """Compute fixed episode targets before packing or updating the critic."""
        if batch.get("advantages") is not None:
            return batch
        advantages, returns = calculate_adv_and_returns(
            task_type=self.cfg.runner.task_type,
            adv_type=self.cfg.algorithm.adv_type,
            rewards=batch["rewards"],
            values=batch["values"],
            loss_mask=batch["response_mask"],
            idx_to_traj=batch["idx_to_traj"],
            prompt_lengths=batch["prompt_lengths"],
            group_size=self.cfg.algorithm.group_size,
            gamma=self.cfg.algorithm.gamma,
            gae_lambda=self.cfg.algorithm.gae_lambda,
            normalize_advantages=False,
        )
        if self.cfg.algorithm.normalize_advantages:
            stats = masked_stats(
                torch.cat(advantages).to(self.torch_device_type),
                torch.cat(batch["response_mask"]).to(self.torch_device_type),
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    stats, group=parallel_state.get_data_parallel_group()
                )
            advantages = [
                normalize_from_stats(row, stats.cpu()).masked_fill(~mask, 0.0)
                for row, mask in zip(advantages, batch["response_mask"])
            ]
        batch["advantages"] = advantages
        batch["returns"] = returns
        return batch

    def run_training(
        self,
        input_channel: Channel,
        output_channel: Channel | None = None,
        do_offload: bool = True,
        compute_rollout_metrics: bool = True,
    ):
        """Pack and balance the rollout batch, then update fixed value targets."""
        batch, results = self.get_step_batch(input_channel)
        assert "values" in batch, (
            "Multi-agent critic training requires fixed old values."
        )
        batch = self.compute_advantages_and_returns(batch)
        if output_channel is not None:
            offset = 0
            for result in results:
                end = offset + result.num_sequence
                result.advantages = [
                    row.clone() for row in batch["advantages"][offset:end]
                ]
                result.values = result.returns = None
                offset = end
        rollout_metrics = (
            self._compute_rollout_metrics(batch) if compute_rollout_metrics else None
        )
        if self.pack_traj:
            batch = DynamicRolloutResult.pack_traj_batch({}, batch)
        for key in list(batch.keys()):
            if key == "idx_to_traj" or key.startswith("extra:"):
                batch.pop(key, None)

        self._load_weight_and_optimizer()
        self._training_setup()

        batch = self._pad_ragged_rows(batch)
        if self.use_profiler:
            self.profiler.init_fwd_bwd_schedule(self.cfg.algorithm.n_minibatches)

        global_batches = get_iterator_k_split(
            batch,
            num_splits=self.num_train_steps,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.role_cfg.seed,
        )
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for global_batch in global_batches:
                if self.cfg.algorithm.use_valid_token_scale:
                    self._setup_valid_token_scale(global_batch)
                assert len(global_batch["input_ids"]) > 0, (
                    "Dynamic DP load balancing must produce nonempty critic minibatches."
                )
                dp_size = parallel_state.get_data_parallel_world_size()
                configure_batch_sizes(
                    rank=torch.distributed.get_rank(),
                    mbs=self.role_cfg.micro_batch_size,
                    gbs=len(global_batch["input_ids"]) * dp_size,
                    dp=dp_size,
                )
                training_metrics = self.training_step(global_batch)
                training_metrics_list.append(training_metrics)

        self._gather_weights_among_dp()
        if output_channel is not None:
            for result in results:
                self.put_result(result, output_channel)
        if do_offload:
            self._offload_weight_and_optimizer()
        # Shared training devices must be able to reuse the activation cache.
        clear_memory()
        return rollout_metrics, training_metrics_list

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool = False,
        do_offload: bool = True,
    ) -> None:
        """Compute token values and return each episode group through the channel."""
        assert not compute_ref_logprobs, "Critic inference does not compute logprobs."
        batch, results = self.get_step_batch(input_channel)
        self._load_weight_and_optimizer()
        with self.worker_timer():
            values = self.inference_step(batch)
        offset = 0
        for result in results:
            end = offset + result.num_sequence
            result.values = [row.clone() for row in values[offset:end]]
            self.put_result(result, output_channel)
            offset = end
        self.scheduler_offload_sync()
        if do_offload:
            self._offload_weight_and_optimizer()
