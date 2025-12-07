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

import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel
from rlinf.data.io_struct import RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
    seq_mean_token_mean,
    seq_mean_token_sum,
)
from rlinf.workers.rollout.utils import RankMapper


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(self._world_size,), mesh_dim_names=["fsdp"]
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self._weight_dst_rank_in_rollout = self._rank
        if self._weight_dst_rank_in_rollout >= self._component_placement.get_world_size(
            "rollout"
        ):
            self._weight_dst_rank_in_rollout = None

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self):
        if self.cfg.actor.get("enable_offload", False):
            self.offload_optimizer()

        if next(self.model.parameters()).is_cpu:
            if self.cfg.actor.get("enable_offload", False):
                self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict()
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()

    def recv_rollout_batch(self, input_channel: Channel) -> None:
        """
        Receive rollout batch from rollout workers.
        """
        send_num = (
            self._component_placement.get_world_size("rollout")
            * self.num_pipeline_stages
        )
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(input_channel.get())

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        for key in recv_list[0].keys():
            self.rollout_batch[key] = torch.cat(
                [recv_list[i][key] for i in range(split_num)], dim=1
            )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        for key, value in rollout_batch.items():
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            rollout_batch[key] = new_value

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if self.rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if self.rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & self.rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def compute_advantages_and_returns(self):
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        self.rollout_batch.update(
            {
                "loss_mask": kwargs["loss_mask"],
                "loss_mask_sum": kwargs["loss_mask_sum"],
            }
        )
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self, input_channel: Channel):
        self.recv_rollout_batch(input_channel)
        with self.worker_timer():
            rollout_metrics = self.compute_advantages_and_returns()

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            rollout_size = (
                self.rollout_batch["prev_logprobs"].shape[0]
                * self.rollout_batch["prev_logprobs"].shape[1]
            )
            g = torch.Generator()
            g.manual_seed(self.cfg.actor.seed + self._rank)
            shuffle_id = torch.randperm(rollout_size, generator=g)

            with torch.no_grad():
                for key, value in self.rollout_batch.items():
                    if key in ["dones", "prev_values"]:
                        value = value[:-1]
                    if "env_info" in key:
                        continue
                    if value is None:
                        continue
                    value = value.reshape(rollout_size, *value.shape[2:])
                    self.rollout_batch[key] = value[shuffle_id]

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            # Split to make minibatch iterator for updating the actor
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            rollout_size = self.rollout_batch["prev_logprobs"].size(0)
            batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
            assert rollout_size % batch_size_per_rank == 0, (
                f"{rollout_size=} is not divisible by {batch_size_per_rank=}"
            )
            metrics = {}
            update_epoch = self.cfg.algorithm.get("update_epoch", 1)
            for _ in range(update_epoch):
                rollout_dataloader_iter = get_iterator_k_split(
                    self.rollout_batch,
                    rollout_size // batch_size_per_rank,
                )
                for train_global_batch in rollout_dataloader_iter:
                    # split batch into micro_batches
                    train_global_batch_size = train_global_batch["prev_logprobs"].shape[
                        0
                    ]
                    assert (
                        train_global_batch_size
                        == self.cfg.actor.global_batch_size
                        // torch.distributed.get_world_size()
                    )
                    assert (
                        train_global_batch_size % self.cfg.actor.micro_batch_size == 0
                    ), f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                    train_micro_batch = get_iterator_k_split(
                        train_global_batch,
                        train_global_batch_size // self.cfg.actor.micro_batch_size,
                    )

                    self.optimizer.zero_grad()
                    for idx, data in enumerate(train_micro_batch):
                        for k, v in data.items():
                            data[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")
                        backward_ctx = self.before_micro_batch(
                            self.model,
                            is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                        )
                        advantages = data["advantages"]
                        prev_logprobs = data["prev_logprobs"]
                        returns = data.get("returns", None)
                        prev_values = data.get("prev_values", None)
                        loss_mask = data.get("loss_mask", None)
                        loss_mask_sum = data.get("loss_mask_sum", None)

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        data["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                        compute_values = (
                            True if self.cfg.algorithm.adv_type == "gae" else False
                        )

                        with self.amp_context:
                            output_dict = self.model(
                                data=data,
                                compute_logprobs=True,
                                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                                compute_values=compute_values,
                                use_cache=False,
                            )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T
                    ]:
                        prev_logprobs = output_dict["prev_logprobs"]

                        kwargs = {
                            "loss_type": self.cfg.algorithm.loss_type,
                            "logprob_type": self.cfg.algorithm.logprob_type,
                            "reward_type": self.cfg.algorithm.reward_type,
                            "single_action_dim": self.cfg.actor.model.get(
                                "action_dim", 7
                            ),
                            "logprobs": output_dict["logprobs"],
                            "values": output_dict.get("values", None),
                            "old_logprobs": prev_logprobs,
                            "advantages": advantages,
                            "returns": returns,
                            "prev_values": prev_values,
                            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                            "value_clip": self.cfg.algorithm.get("value_clip", None),
                            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                            "loss_mask": loss_mask,
                            "loss_mask_sum": loss_mask_sum,
                            "max_episode_steps": self.cfg.env.train.max_episode_steps,
                            "task_type": self.cfg.runner.task_type,
                            "critic_warmup": self.optimizer_steps
                            < self.critic_warmup_steps,
                        }
                        loss, metrics_data = policy_loss(**kwargs)

                        entropy_loss = torch.tensor(
                            0.0, device=torch.cuda.current_device()
                        )
                        if (
                            self.cfg.algorithm.entropy_bonus > 0
                            and not kwargs["critic_warmup"]
                        ):
                            entropy = output_dict["entropy"]
                            entropy = reshape_entropy(
                                entropy,
                                entropy_type=self.cfg.algorithm.entropy_type,
                                action_dim=self.cfg.actor.model.get("action_dim", 7),
                                batch_size=output_dict["logprobs"].shape[0],
                            )
                            entropy_loss = masked_mean(entropy, mask=loss_mask)
                            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                        metrics_data["entropy_loss"] = entropy_loss.detach().item()

                        loss /= self.gradient_accumulation
                        with backward_ctx:
                            self.grad_scaler.scale(loss).backward()

                        metrics_data["loss"] = loss.detach().item()
                        append_to_dict(metrics, metrics_data)

                    torch.cuda.empty_cache()

                    grad_norm, lr_list = self.optimizer_step()
                    data = {
                        "actor/grad_norm": grad_norm,
                        "actor/lr": lr_list[0],
                    }
                    if len(lr_list) > 1:
                        data["critic/lr"] = lr_list[1]
                    append_to_dict(metrics, data)
            # put LR scheduler step here
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return rollout_metrics, train_metrics

    def set_global_step(self, global_step):
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
