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

import gc
import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm

from rlinf.algorithms.embodiment.utils import (
    actor_loss_fn,
    append_to_dict,
    calculate_advantages_and_returns,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import custom_forward
from rlinf.scheduler import Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.placement import HybridComponentPlacement


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg)
        self._weight_dst_rank_in_rollout = self._rank
        if self._weight_dst_rank_in_rollout >= self._component_placement.get_world_size(
            "rollout"
        ):
            self._weight_dst_rank_in_rollout = None

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.channel = self.connect_channel(cfg.actor.channel.name)
        self.channel.create_queue(
            cfg.actor.channel.queue_name, maxsize=cfg.actor.channel.queue_size
        )

    def init_worker(self):
        self.setup_model_and_optimizer()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self):
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        state_dict = self.get_model_state_dict()
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    async def recv_rollout_batch(self):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for i in range(split_num):
            recv_list.append(
                await self.channel.get(
                    queue_name=self._replay_buffer_name, async_op=True
                ).async_wait()
            )

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        for key in recv_list[0].keys():
            if "env_info/" not in key:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=1
                )
            else:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=0
                )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(self, rollout_batch):
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

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        return rollout_batch

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def compute_advantages_and_returns(self):
        stage_num = self.cfg.rollout.pipeline_stage_num
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        num_group_envs_for_train = (
            self.cfg.algorithm.num_group_envs
            * stage_num
            * env_world_size
            // actor_world_size
        )
        advantages, returns = calculate_advantages_and_returns(
            adv_type=self.cfg.algorithm.adv_type,
            rewards=self.rollout_batch["rewards"],
            dones=self.rollout_batch["dones"],
            normalize_advantages=self.cfg.algorithm.get("normalize_advantages", True),
            values=self.rollout_batch.get("prev_values", None),
            gamma=self.cfg.algorithm.get("gamma", 1),
            gae_lambda=self.cfg.algorithm.get("gae_lambda", 1),
            num_group_envs=num_group_envs_for_train,
            group_size=self.cfg.algorithm.get("group_size", 8),
            reward_type=self.cfg.algorithm.reward_type,
            loss_mask=self.rollout_batch.get("loss_mask", None),
            rollout_epoch=self.cfg.algorithm.get("rollout_epoch", 1),
        )
        self.rollout_batch.update({"advantages": advantages, "returns": returns})
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self):
        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        rollout_size = (
            self.rollout_batch["input_ids"].shape[0]
            * self.rollout_batch["input_ids"].shape[1]
        )
        shuffle_id = torch.randperm(rollout_size)

        for key, value in self.rollout_batch.items():
            self.log_on_first_rank(f"run training, {key}: {value.shape}")

        with torch.no_grad():
            for key, value in self.rollout_batch.items():
                if key in ["dones", "prev_values"]:
                    value = value[:-1]
                if "env_info" in key:
                    continue
                value = value.reshape(rollout_size, *value.shape[2:])
                self.rollout_batch[key] = value[shuffle_id]

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["input_ids"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        rollout_dataloader_iter = get_iterator_k_split(
            self.rollout_batch,
            rollout_size // batch_size_per_rank,
        )

        metrics = {}
        for _, train_global_batch in tqdm(
            enumerate(rollout_dataloader_iter), desc="get loss and metrics"
        ):
            # split batch into micro_batches
            train_global_batch_size = train_global_batch["input_ids"].shape[0]
            assert (
                train_global_batch_size
                == self.cfg.actor.global_batch_size
                // torch.distributed.get_world_size()
            )
            assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
            )
            train_micro_batch = get_iterator_k_split(
                train_global_batch,
                train_global_batch_size // self.cfg.actor.micro_batch_size,
            )

            self.optimizer.zero_grad()
            for data_idx, data in enumerate(train_micro_batch):
                for k, v in data.items():
                    data[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                data = self.model.preprocess_for_train(data)
                input_ids = data["input_ids"]
                action_tokens = data["action_tokens"]
                attention_mask = data["attention_mask"]
                pixel_values = data["pixel_values"]

                action_token_len = self.model.action_dim * self.model.num_action_chunks

                logits_processor_args = {
                    "action_tokens": action_tokens,
                    "vocab_size": self.model.vocab_size,
                    "n_action_bins": self.model.config.n_action_bins,
                }

                output_dict = custom_forward(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    action_token_len=action_token_len,
                    value_model=True if self.cfg.algorithm.adv_type == "ppo" else False,
                    value_head_mode=self.cfg.actor.model.get("vh_mode", None),
                    temperature=self.cfg.algorithm.sampling_params.temperature_train,
                    top_k=self.cfg.algorithm.sampling_params.top_k,
                    logits_processor_args=logits_processor_args,
                )

                logprobs = output_dict["logprobs"]
                entropy = output_dict["entropy"]
                values = output_dict.get("values", None)
                prev_logprobs = data["prev_logprobs"]
                advantages = data["advantages"]  # [bsz, chunk-step]
                returns = data["returns"]  # [bsz, chunk-step]
                prev_values = data["prev_values"]
                loss_mask = data.get("loss_mask", None)
                loss_mask_sum = data.get("loss_mask_sum", None)
                max_episode_steps = self.cfg.env.train.max_episode_steps

                loss, metrics_data = actor_loss_fn(
                    self.cfg.algorithm.loss_type,
                    self.cfg.algorithm.logprob_type,
                    self.cfg.algorithm.entropy_type,
                    single_action_dim=self.model.action_dim,
                    logprobs=logprobs,
                    entropy=entropy,
                    values=values,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    returns=returns,
                    prev_values=prev_values,
                    clip_ratio_high=self.cfg.algorithm.clip_ratio_high,
                    clip_ratio_low=self.cfg.algorithm.clip_ratio_low,
                    value_clip=self.cfg.algorithm.get("value_clip", None),
                    huber_delta=self.cfg.algorithm.get("huber_delta", None),
                    entropy_bonus=self.cfg.algorithm.entropy_bonus,
                    loss_mask=loss_mask,
                    loss_mask_sum=loss_mask_sum,
                    max_episode_steps=max_episode_steps,
                )

                loss /= self.gradient_accumulation
                loss.backward()

                metrics_data["loss"] = loss.detach().item()
                append_to_dict(metrics, metrics_data)

            torch.cuda.empty_cache()

            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()

            self.optimizer.zero_grad()
            data = {
                "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.cfg.algorithm.adv_type == "ppo":
                data["critic/lr"] = self.optimizer.param_groups[1]["lr"]
            append_to_dict(metrics, data)

        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        return mean_metric_dict

    def save_checkpoint(self, save_base_path, step):
        torch.distributed.barrier()
        model_state = self.get_model_state_dict()
        optim_state = self.get_optimizer_state_dict()
        if self._rank == 0:
            os.makedirs(save_base_path, exist_ok=True)
            torch.save(model_state, os.path.join(save_base_path, "model.pt"))
            torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
        torch.distributed.barrier()
