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


import os
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory, masked_mean, reshape_entropy
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def process_nested_dict_for_train(
    nested_dict: dict, shuffle_id: Optional[torch.Tensor]
) -> dict:
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            if isinstance(value, torch.Tensor):
                value = value[:-1]

        if "env_info" in key:
            raise NotImplementedError("env_info nested dict is not supported here")

        if value is None:
            ret_dict[key] = None
            continue

        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1, *value.shape[2:])
            ret_dict[key] = flat[shuffle_id] if shuffle_id is not None else flat
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
        else:
            raise NotImplementedError(
                f"Unsupported type in nested_dict: {key}={type(value)}"
            )
    return ret_dict


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.version = 0

    def set_version(self, version: int) -> None:
        self.version = int(version)

    @torch.inference_mode()
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        prev_values = self.rollout_batch.get("prev_values", None)

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": prev_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": False,  # here we should not call per-rank normalization
        }

        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.rollout_batch.update(adv_and_ret)

        # keep original masks if provided
        if kwargs["loss_mask"] is not None:
            self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading not supported here (recompute)."
        )

        T = self.rollout_batch["prev_logprobs"].shape[0]
        B = self.rollout_batch["prev_logprobs"].shape[1]

        flat = process_nested_dict_for_train(self.rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro = self.cfg.actor.micro_batch_size
        num_splits = (total + micro - 1) // micro

        it = get_iterator_k_split(
            flat, num_splits=num_splits, enforce_divisible_batch=False
        )

        self.model.eval()
        prox_logprobs_list = []

        for mb in it:
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                mb["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
                mb["top_k"] = self.cfg.algorithm.sampling_params.top_k

            mb = put_tensor_device(mb, self.device)

            out = self.model(
                data=mb,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
            )
            prox_logprobs_list.append(out["logprobs"].cpu())

        prox_logprobs = torch.cat(prox_logprobs_list, dim=0).view(
            T, B, *self.rollout_batch["prev_logprobs"].shape[2:]
        )
        self.rollout_batch["proximal_logprobs"] = prox_logprobs

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        T = int(self.rollout_batch["prev_logprobs"].shape[0])
        B = int(self.rollout_batch["prev_logprobs"].shape[1])
        N = T * B

        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.cfg.actor.seed) + int(self._rank))
        shuffle_id = torch.randperm(N, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        if self.cfg.algorithm.normalize_advantages:
            self.rollout_batch["advantages"] = masked_normalization(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        self.model.train()

        world = int(self._world_size)
        global_bs = int(self.cfg.actor.global_batch_size)
        micro_bs = int(self.cfg.actor.micro_batch_size)

        assert global_bs % (micro_bs * world) == 0, (
            f"global_batch_size {global_bs} must be divisible by micro_batch_size {micro_bs} * world_size {world}"
        )

        batch_size_per_rank = global_bs // world
        micro_per_rank = batch_size_per_rank // micro_bs
        self.gradient_accumulation = micro_per_rank

        rollout_size = int(self.rollout_batch["prev_logprobs"].shape[0])
        assert rollout_size % batch_size_per_rank == 0, (
            f"Flattened rollout_size {rollout_size} must be divisible by batch_size_per_rank {batch_size_per_rank}"
        )

        metrics: dict[str, list] = {}

        rollout_dataloader_iter = split_dict_to_chunk(
            self.rollout_batch,
            rollout_size // batch_size_per_rank,
        )

        for train_global_batch in rollout_dataloader_iter:
            train_global_batch_size = int(train_global_batch["prev_logprobs"].shape[0])
            assert train_global_batch_size == batch_size_per_rank, (
                f"Expected per-rank global batch size {batch_size_per_rank}, got {train_global_batch_size}"
            )
            assert train_global_batch_size % micro_bs == 0

            micro_iter = split_dict_to_chunk(
                train_global_batch,
                train_global_batch_size // self.cfg.actor.micro_batch_size,
            )

            self.optimizer.zero_grad()

            for mb_idx, data in enumerate(micro_iter):
                data = put_tensor_device(data, f"cuda:{int(os.environ['LOCAL_RANK'])}")

                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
                )

                advantages = data["advantages"]
                old_logprobs = data["prev_logprobs"]
                returns = data.get("returns", None)
                prev_values = data.get("prev_values", None)
                loss_mask = data.get("loss_mask", None)
                loss_mask_sum = data.get("loss_mask_sum", None)

                versions = data.get("versions", None)
                proximal_logprobs = data.get("proximal_logprobs", None)
                current_version = int(self.version) + 1

                if SupportedModel(self.cfg.actor.model.model_type) in [
                    SupportedModel.OPENVLA,
                    SupportedModel.OPENVLA_OFT,
                ]:
                    data["temperature"] = (
                        self.cfg.algorithm.sampling_params.temperature_train
                    )
                    data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                compute_values = self.cfg.algorithm.adv_type == "gae"

                with self.amp_context:
                    out = self.model(
                        data=data,
                        compute_logprobs=True,
                        compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                        compute_values=compute_values,
                        use_cache=False,
                    )

                if SupportedModel(self.cfg.actor.model.model_type) in [
                    SupportedModel.GR00T
                ]:
                    old_logprobs = out["prev_logprobs"]

                kwargs = {
                    "loss_type": self.cfg.algorithm.loss_type,
                    "logprob_type": self.cfg.algorithm.logprob_type,
                    "reward_type": self.cfg.algorithm.reward_type,
                    "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                    "logprobs": out["logprobs"],
                    "values": out.get("values", None),
                    "old_logprobs": old_logprobs,
                    "advantages": advantages,
                    "returns": returns,
                    "prev_values": prev_values,
                    "proximal_logprobs": proximal_logprobs,
                    "versions": versions,
                    "current_version": current_version,
                    "behave_weight_threshold": self.cfg.algorithm.get(
                        "behave_weight_threshold", None
                    ),
                    "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                    "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                    "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                    "value_clip": self.cfg.algorithm.get("value_clip", None),
                    "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                    "loss_mask": loss_mask,
                    "loss_mask_sum": loss_mask_sum,
                    "max_episode_steps": self.cfg.env.train.max_episode_steps,
                    "task_type": self.cfg.runner.task_type,
                    "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
                }

                loss, metrics_data = policy_loss(**kwargs)

                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.cfg.algorithm.entropy_bonus > 0 and not kwargs["critic_warmup"]:
                    entropy = out["entropy"]
                    entropy = reshape_entropy(
                        entropy,
                        entropy_type=self.cfg.algorithm.entropy_type,
                        action_dim=self.cfg.actor.model.get("action_dim", 7),
                        batch_size=out["logprobs"].shape[0],
                    )
                    entropy_loss = masked_mean(entropy, mask=loss_mask)
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

                metrics_data["entropy_loss"] = float(entropy_loss.detach().item())
                metrics_data["loss"] = float(loss.detach().item())
                append_to_dict(metrics, metrics_data)

            torch.cuda.empty_cache()

            grad_norm, lr_list = self.optimizer_step()
            extra = {"actor/grad_norm": grad_norm, "actor/lr": lr_list[0]}
            if len(lr_list) > 1:
                extra["critic/lr"] = lr_list[1]
            append_to_dict(metrics, extra)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict
