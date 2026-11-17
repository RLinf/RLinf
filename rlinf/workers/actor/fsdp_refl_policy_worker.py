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

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.algorithms.refl_scorers import build_refl_scorer
from rlinf.data.datasets.common.epoch import forward_set_epoch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.utils import (
    clear_memory,
    get_rng_state,
    seed_everything,
    set_rng_state,
)
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class _ReFLDataIterator(Iterator[dict[str, Any]]):
    """Cycle a sharded data loader with a checkpointed epoch and position."""

    def __init__(self, loader: StatefulDataLoader) -> None:
        if len(loader) == 0:
            raise ValueError(
                "ReFL requires at least one complete data batch per actor rank."
            )
        self.loader = loader
        self.epoch = 0
        self.iterator = None

    def __next__(self) -> dict[str, Any]:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            forward_set_epoch(self.loader, self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "loader": self.loader.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = state["epoch"]
        forward_set_epoch(self.loader, self.epoch)
        self.loader.load_state_dict(state["loader"])
        self.iterator = iter(self.loader)


class EmbodiedReFLFSDPPolicy(EmbodiedFSDPActor):
    """Sample each ReFL batch with current weights after its pre-training update."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.refl_margin = float(cfg.algorithm.refl_margin)
        self.refl_grad_scale = float(cfg.algorithm.refl_grad_scale)
        self.refl_xt_source = str(cfg.algorithm.refl_xt_source)
        self.refl_pre_loss_weight = float(cfg.algorithm.refl_pre_loss_weight)

    def init_worker(self) -> None:
        seed_everything(self.cfg.actor.seed)
        super().init_worker()
        self.reward_scorer = build_refl_scorer(
            self.cfg.actor.refl_reward_model, self.device
        )
        if self.refl_pre_loss_weight > 0:
            self._build_pre_data_loader()
        self._build_prompt_data_loader()
        seed_everything(self.cfg.actor.seed + self._rank)

    def _build_pre_data_loader(self) -> None:
        from rlinf.envs.sim.diffusion import build_reward_dataset

        pre_data_cfg = self.cfg.actor.refl_pre_data
        dataset = build_reward_dataset(pre_data_cfg)
        self.pre_data_loader = StatefulDataLoader(
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
        self.pre_data_iter = _ReFLDataIterator(self.pre_data_loader)

    def _next_pre_micro_batches(self) -> list[dict[str, Any]]:
        return [next(self.pre_data_iter) for _ in range(self.gradient_accumulation)]

    def _build_prompt_data_loader(self) -> None:
        from rlinf.envs.sim.diffusion import build_reward_dataset

        dataset = build_reward_dataset(self.cfg.actor.refl_data)
        loader = StatefulDataLoader(
            dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            sampler=DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                seed=self.cfg.actor.seed,
                drop_last=True,
            ),
            num_workers=0,
            drop_last=True,
        )
        self.prompt_data_iter = _ReFLDataIterator(loader)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save each rank's prompt and supervised data positions."""
        super().save_checkpoint(save_path, step)
        state = {"prompt": self.prompt_data_iter.state_dict()}
        if self.refl_pre_loss_weight > 0:
            state["pre"] = self.pre_data_iter.state_dict()
        torch.save(state, Path(save_path) / f"refl_data_rank_{self._rank}.pt")

    def load_checkpoint(self, load_path: str) -> None:
        """Restore data iterators without advancing the checkpoint RNG state."""
        super().load_checkpoint(load_path)
        data_path = Path(load_path) / f"refl_data_rank_{self._rank}.pt"
        if not data_path.exists():
            self.log_warning(
                f"Checkpoint {load_path} has no ReFL data state for rank "
                f"{self._rank}; restarting the training data streams."
            )
            return
        rng_state = get_rng_state()
        state = torch.load(data_path, weights_only=False)
        self.prompt_data_iter.load_state_dict(state["prompt"])
        if self.refl_pre_loss_weight > 0:
            self.pre_data_iter.load_state_dict(state["pre"])
        set_rng_state(rng_state)

    def _training_batches(self) -> Iterator[Sequence[dict[str, Any]]]:
        sample_count = self.cfg.actor.refl_samples_per_step
        for _ in range(sample_count // self.cfg.actor.global_batch_size):
            yield [
                {
                    "forward_inputs": {
                        "prompts": next(self.prompt_data_iter)["task_description"]
                    }
                }
                for _ in range(self.gradient_accumulation)
            ]

    def _reward_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward_inputs = batch["forward_inputs"]
        with self.amp_context:
            output = self.model(
                forward_type=ForwardType.REFL,
                forward_inputs=forward_inputs,
                xt_source=self.refl_xt_source,
            )
            samples = output["samples"]
            rewards = self.reward_scorer(
                samples, output.get("forward_inputs", forward_inputs)
            )
        if rewards.shape != samples.shape[:1]:
            raise ValueError(
                f"ReFL scorer must return rewards of shape {samples.shape[:1]}, "
                f"got {rewards.shape}."
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
        metrics = {}
        reward_updates = 0
        for micro_batches in self._training_batches():
            if self.refl_pre_loss_weight > 0:
                grad_norm, _ = self._accumulate_and_step(
                    self._next_pre_micro_batches(), self._pre_loss, metrics
                )
                append_to_dict(metrics, {"actor/refl_pre_grad_norm": grad_norm})

            grad_norm, lr_list = self._accumulate_and_step(
                micro_batches, self._reward_loss, metrics
            )
            reward_updates += 1
            append_to_dict(
                metrics,
                {
                    "actor/refl_reward_grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                },
            )

        metrics["actor/refl_reward_updates"] = [reward_updates]
        metrics["actor/refl_pre_updates"] = [
            reward_updates if self.refl_pre_loss_weight > 0 else 0
        ]
        metrics["actor/refl_samples"] = [
            reward_updates * self.cfg.actor.global_batch_size
        ]
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)
