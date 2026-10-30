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

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_seqlen_balanced_partitions,
)
from rlinf.utils.distributed import RolloutDataBalance, compute_rollout_metrics_dynamic
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import clear_memory, configure_batch_sizes, cpu_dict
from rlinf.workers.critic.megatron_critic_worker import MegatronCritic


class MAMegatronCritic(MegatronCritic):
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
        self.pack_traj = cfg.actor.get("pack_traj", True)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], DynamicRolloutResult]:
        """Receive one episode group and broadcast it across model parallel ranks."""
        if channel.is_local:
            result = channel.get()
        else:
            result = channel.get() if self.is_data_io_rank else None
            for ranks in (
                parallel_state._MODEL_PARALLEL_GLOBAL_RANKS,
                parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS,
            ):
                result = self.broadcast(result, groups=[(self._group_name, ranks)])
        batch = result.to_actor_batch(
            self.role_cfg.model.encoder_seq_length, self.tokenizer.eos_token_id
        )
        return batch, result

    def get_forward_step_func(self):
        """Align full-sequence state values with generated action tokens."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = {
                key: val.to(self.torch_device_type) if torch.is_tensor(val) else val
                for key, val in next(dataloader_iter).items()
            }
            output = self.custom_forward(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["position_ids"],
                sequence_parallel=self.transformer_config.sequence_parallel,
            )

            def action_values(output):
                # Token t uses the state before that token, including tool observations.
                return torch.nn.functional.pad(
                    output.squeeze(-1).float()[:, :-1], (1, 0)
                )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output

                if unwrap_model(model).post_process:
                    output = action_values(output) * batch["response_mask"]
                return output, id_func

            def loss_func(output):
                return self._compute_value_loss(
                    action_values(output), batch, batch["response_mask"]
                )

            return output, loss_func

        return forward_output_and_loss_func

    def _setup_valid_token_scale(self, batch: dict[str, torch.Tensor]) -> None:
        self.global_valid_token = (
            batch["response_mask"]
            .to(device=self.torch_device_type, dtype=torch.float32)
            .sum()
        )
        torch.distributed.all_reduce(
            self.global_valid_token, group=parallel_state.get_data_parallel_group()
        )

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        return RolloutDataBalance.from_rollout_batches_dynamic(
            rollout_batches=batch,
            dp_world_size=parallel_state.get_data_parallel_world_size(),
            dp_rank=parallel_state.get_data_parallel_rank(),
            dp_group=parallel_state.get_data_parallel_group(),
            rollout_batch_pad=DynamicRolloutResult.get_batch_pad(
                self.role_cfg.model.encoder_seq_length, batch.keys()
            ),
            split_fix_chunk=self.num_train_steps * self.role_cfg.micro_batch_size,
            partitioning_tool=get_seqlen_balanced_partitions,
        )

    def _compute_rollout_metrics(self, batch: dict[str, torch.Tensor]):
        group = self.get_rollout_metrics_group(batch)
        if group is None:
            return None
        metrics, _, _ = compute_rollout_metrics_dynamic(
            batch, self.cfg.data.max_prompt_length, self.response_len, group
        )
        return cpu_dict(metrics)

    def run_training(
        self,
        input_channel: Channel,
        output_channel: Channel | None = None,
        do_offload: bool = True,
        compute_rollout_metrics: bool = True,
    ):
        """Pack and balance the rollout batch, then update fixed value targets."""
        batches = []
        results = []
        total_result_size_per_dp: int = (
            self.cfg.data.rollout_batch_size
            // parallel_state.get_data_parallel_world_size()
        )
        for _ in range(total_result_size_per_dp):
            batch, result = self.get_batch(input_channel)
            batches.append(batch)
            results.append(result)
        batch = DynamicRolloutResult.merge_batches(
            batches, self.cfg.algorithm.group_size
        )
        assert "values" in batch and "returns" in batch, (
            "Multi-agent critic training requires fixed old values and returns."
        )
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

        if self.enable_dp_load_balance:
            batch = self._dp_load_balance(batch)
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
                dp_size = parallel_state.get_data_parallel_world_size()
                configure_batch_sizes(
                    rank=torch.distributed.get_rank(),
                    mbs=self.role_cfg.micro_batch_size,
                    gbs=global_batch["input_ids"].shape[0] * dp_size,
                    dp=dp_size,
                )
                training_metrics = self.training_step(global_batch)
                training_metrics_list.append(training_metrics)

        self._gather_weights_among_dp()
        if output_channel is not None:
            self.put_result(
                DynamicRolloutResult.merge_result_list(results), output_channel
            )
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
        batches = []
        rollout_results = []
        total_batch_size_per_dp: int = (
            self.cfg.data.rollout_batch_size
            // parallel_state.get_data_parallel_world_size()
        )
        for _ in range(total_batch_size_per_dp):
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            rollout_results.append(rollout_result)
        merged_batch, num_sequence_per_group = DynamicRolloutResult.merge_batches(
            batches, adjust_traj_indices=False, return_num_sequence_per_group=True
        )
        merged_batch.pop("idx_to_traj")

        rollout_result = DynamicRolloutResult.merge_result_list(rollout_results)

        self._load_weight_and_optimizer()
        with self.worker_timer():
            self.process_inference_output(
                rollout_result, self.inference_step(merged_batch).cpu()
            )
        for rollout_result in DynamicRolloutResult.split_results(
            rollout_result, num_sequence_per_group
        ):
            self.put_result(rollout_result, output_channel)
        self.scheduler_offload_sync()
        if do_offload:
            self._offload_weight_and_optimizer()
