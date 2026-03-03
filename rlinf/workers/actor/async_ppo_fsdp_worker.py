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
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory, masked_mean, reshape_entropy
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def flatten_rollout_batch_for_train(
    nested_dict: dict, shuffle_id: Optional[torch.Tensor]
) -> dict:
    """Flatten [T, B, ...] rollout tensors to [T*B, ...] for actor training."""
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
            ret_dict[key] = flatten_rollout_batch_for_train(value, shuffle_id)
        else:
            raise NotImplementedError(
                f"Unsupported value type in rollout batch: key={key}, type={type(value)}"
            )

    return ret_dict


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Embodied FSDP actor worker for async PPO / decoupled actor-critic training."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.version = 0

    def set_version(self, version: int) -> None:
        self.version = int(version)

    @torch.inference_mode()
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        proximal_values = self.rollout_batch.get("proximal_values", None)
        prev_values = self.rollout_batch.get("prev_values", None)

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": proximal_values if proximal_values is not None else prev_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.rollout_batch.update(adv_and_ret)

        if kwargs["loss_mask"] is not None:
            self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing proximal logprobs."
        )

        t_dim = self.rollout_batch["prev_logprobs"].shape[0]
        b_dim = self.rollout_batch["prev_logprobs"].shape[1]

        flat = flatten_rollout_batch_for_train(self.rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size

        iterator = split_dict_to_chunk(flat, num_splits)

        self.model.eval()
        proximal_logprobs_list = []

        for micro_batch in iterator:
            micro_batch = put_tensor_device(micro_batch, self.device)
            forward_inputs = micro_batch.get("forward_inputs", None)
            if forward_inputs is None:
                raise ValueError(
                    "Missing forward_inputs in compute_proximal_logprobs. "
                    "This usually means batch splitting dropped nested dict fields."
                )

            model_kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                model_kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
                model_kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
            elif (
                SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.GR00T
            ):
                model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

            out = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **model_kwargs,
            )
            proximal_logprobs_list.append(out["logprobs"].cpu())

        proximal_logprobs = torch.cat(proximal_logprobs_list, dim=0).view(
            t_dim,
            b_dim,
            *self.rollout_batch["prev_logprobs"].shape[2:],
        )
        self.rollout_batch["proximal_logprobs"] = proximal_logprobs

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        t_dim = int(self.rollout_batch["prev_logprobs"].shape[0])
        b_dim = int(self.rollout_batch["prev_logprobs"].shape[1])
        total_samples = t_dim * b_dim

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.actor.seed) + int(self._rank))
        shuffle_id = torch.randperm(total_samples, generator=generator)

        with torch.no_grad():
            self.rollout_batch = flatten_rollout_batch_for_train(
                self.rollout_batch, shuffle_id
            )

        if self.cfg.algorithm.normalize_advantages:
            self.rollout_batch["advantages"] = masked_normalization(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        self.model.train()

        world_size = int(self._world_size)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)

        assert global_batch_size % (micro_batch_size * world_size) == 0, (
            f"global_batch_size {global_batch_size} must be divisible by "
            f"micro_batch_size {micro_batch_size} * world_size {world_size}"
        )

        per_rank_batch_size = global_batch_size // world_size
        micro_per_rank = per_rank_batch_size // micro_batch_size
        self.gradient_accumulation = micro_per_rank

        flattened_rollout_size = int(self.rollout_batch["prev_logprobs"].shape[0])
        assert flattened_rollout_size % per_rank_batch_size == 0, (
            f"Flattened rollout size {flattened_rollout_size} must be divisible by "
            f"per-rank batch size {per_rank_batch_size}"
        )
        num_global_batches = flattened_rollout_size // per_rank_batch_size

        metrics: dict[str, list] = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))

        for _ in range(update_epoch):
            global_batch_iter = split_dict_to_chunk(
                self.rollout_batch,
                num_global_batches,
            )

            for train_global_batch in global_batch_iter:
                train_global_batch_size = int(
                    train_global_batch["prev_logprobs"].shape[0]
                )
                assert train_global_batch_size == per_rank_batch_size, (
                    f"Expected per-rank global batch size {per_rank_batch_size}, "
                    f"got {train_global_batch_size}"
                )
                assert train_global_batch_size % micro_batch_size == 0

                micro_batch_iter = split_dict_to_chunk(
                    train_global_batch,
                    micro_per_rank,
                )

                self.optimizer.zero_grad()

                for mb_idx, data in enumerate(micro_batch_iter):
                    data = put_tensor_device(
                        data,
                        f"cuda:{int(os.environ['LOCAL_RANK'])}",
                    )
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
                    proximal_values = data.get("proximal_values", None)
                    current_version = int(self.version) + 1

                    forward_inputs = data.get("forward_inputs", None)
                    if forward_inputs is None:
                        raise ValueError(
                            "Missing forward_inputs in run_training. "
                            "This usually means batch splitting dropped nested dict fields."
                        )

                    model_kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        model_kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        model_kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        model_kwargs["prev_logprobs"] = old_logprobs

                    compute_values = self.cfg.algorithm.adv_type == "gae"

                    with self.amp_context:
                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                            compute_values=compute_values,
                            use_cache=False,
                            **model_kwargs,
                        )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        old_logprobs = out["prev_logprobs"]

                    loss_kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": out["logprobs"],
                        "values": out.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": proximal_values
                        if proximal_values is not None
                        else prev_values,
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
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }

                    loss, metrics_data = policy_loss(**loss_kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not loss_kwargs["critic_warmup"]
                    ):
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

                    metrics_data["actor/entropy_loss"] = float(
                        entropy_loss.detach().item()
                    )
                    metrics_data["actor/total_loss"] = float(loss.detach().item())
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                extra_metrics = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    extra_metrics["critic/lr"] = lr_list[1]
                append_to_dict(metrics, extra_metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict,
            op=torch.distributed.ReduceOp.AVG,
        )
        return mean_metric_dict

    def sync_model_to_rollout(self, version: int = -1) -> None:
        super().sync_model_to_rollout()
        if version >= 0:
            self.set_version(version)
