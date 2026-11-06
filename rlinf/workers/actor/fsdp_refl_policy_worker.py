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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from rlinf.data.datasets.common.epoch import forward_set_epoch
from rlinf.envs.sim.diffusion import build_reward_dataset
from rlinf.envs.sim.diffusion.rewards.visual.image_reward import ImageRewardScorer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedReFLFSDPPolicy(EmbodiedFSDPActor):
    """Reward Feedback Learning for diffusion policies.

    Rollout supplies the no-grad denoising path. Each update then follows
    Algorithm 1 of the paper: one optimizer step on the pre-training loss when
    ``algorithm.refl_pre_loss_weight`` is positive, and one on the reward of a
    one-step x0 prediction from a late denoising state.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.refl_margin = float(cfg.algorithm.refl_margin)
        self.refl_grad_scale = float(cfg.algorithm.refl_grad_scale)
        self.refl_xt_source = str(cfg.algorithm.refl_xt_source)
        self.refl_pre_loss_weight = float(cfg.algorithm.refl_pre_loss_weight)

    def init_worker(self) -> None:
        super().init_worker()
        self.reward_scorer = ImageRewardScorer.from_config(
            self.cfg.actor.refl_reward_model, self.device
        )
        if self.refl_pre_loss_weight > 0:
            self._build_pre_data_loader()

    def _build_pre_data_loader(self) -> None:
        pre_data_cfg = self.cfg.actor.refl_pre_data
        dataset = build_reward_dataset(pre_data_cfg)
        self.pre_data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            sampler=DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                seed=self.cfg.actor.seed,
                drop_last=True,
            ),
            num_workers=pre_data_cfg.get("num_workers", 4),
            drop_last=True,
        )
        self.pre_data_epoch = 0
        self.pre_data_iter = iter(self.pre_data_loader)

    def _next_pre_micro_batches(self) -> list[dict[str, Any]]:
        micro_batches = []
        for _ in range(self.gradient_accumulation):
            try:
                micro_batches.append(next(self.pre_data_iter))
            except StopIteration:
                self.pre_data_epoch += 1
                forward_set_epoch(self.pre_data_loader, self.pre_data_epoch)
                self.pre_data_iter = iter(self.pre_data_loader)
                micro_batches.append(next(self.pre_data_iter))
        return micro_batches

    def _sample_late_state(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick a random late denoising state and its ``t`` in [0, 1]."""
        latents = forward_inputs["latents"]
        rows = torch.arange(latents.shape[0], device=latents.device)
        steps = torch.randint(0, latents.shape[1], rows.shape, device=latents.device)
        t = forward_inputs["timesteps"][rows, steps] / 1000.0
        if self.refl_xt_source == "rollout":
            return latents[rows, steps], t

        x0 = forward_inputs["x0"]
        t_bc = t.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_bc) * x0 + t_bc * torch.randn_like(x0), t

    def _reward_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward_inputs = batch["forward_inputs"]
        x_t, t = self._sample_late_state(forward_inputs)
        with self.amp_context:
            images = self.model(
                forward_type=ForwardType.REFL,
                forward_inputs=forward_inputs,
                refl_inputs={"x_t": x_t, "timesteps": t},
            )["images"]
            rewards = self.reward_scorer.score(
                images,
                forward_inputs["reward_input_ids"],
                forward_inputs["reward_attention_mask"],
            )
        loss = F.relu(self.refl_margin - rewards).mean() * self.refl_grad_scale
        return loss, {
            "actor/refl_reward": rewards.mean().item(),
            "actor/refl_reward_loss": loss.item(),
        }

    def _pre_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        with self.amp_context:
            pre_loss = self.model(forward_type=ForwardType.SFT, data=batch)
        return pre_loss * self.refl_pre_loss_weight, {
            "actor/refl_pre_loss": pre_loss.item()
        }

    def _accumulate_and_step(
        self,
        micro_batches: Sequence[dict[str, Any]],
        loss_fn: Callable[[dict[str, Any]], tuple[torch.Tensor, dict[str, float]]],
        metrics: dict[str, list[float]],
    ) -> tuple[float, list[float]]:
        """Run one optimizer step on the gradients of one loss."""
        self.optimizer.zero_grad()
        for idx, batch in enumerate(micro_batches):
            batch = put_tensor_device(batch, self.device)
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == len(micro_batches),
            )
            loss, metrics_data = loss_fn(batch)
            with backward_ctx:
                self.grad_scaler.scale(loss / len(micro_batches)).backward()
            append_to_dict(metrics, metrics_data)
        self.torch_platform.empty_cache()
        return self.optimizer_step()

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
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
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )

        metrics = {}
        for _ in range(self.update_epoch):
            for global_batch in split_dict_to_chunk(
                self.rollout_batch, rollout_size // batch_size_per_rank
            ):
                if self.refl_pre_loss_weight > 0:
                    grad_norm, _ = self._accumulate_and_step(
                        self._next_pre_micro_batches(), self._pre_loss, metrics
                    )
                    append_to_dict(metrics, {"actor/refl_pre_grad_norm": grad_norm})

                grad_norm, lr_list = self._accumulate_and_step(
                    split_dict_to_chunk(global_batch, self.gradient_accumulation),
                    self._reward_loss,
                    metrics,
                )
                append_to_dict(
                    metrics,
                    {
                        "actor/refl_reward_grad_norm": grad_norm,
                        "actor/lr": lr_list[0],
                    },
                )

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)
