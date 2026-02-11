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

import logging
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # set the global batch size, micro batch size and gradient accumulation
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        # before load dataloader should build the tokenizer
        self.tokenizer = self.build_tokenizer()

        # if train_data_paths is not set, the code will just eval the model
        if self.cfg.data.get("train_data_paths") is None:
            logging.warning("train_data_paths is not set, will just eval the model")
            assert self.cfg.data.get("eval_data_paths") is not None, (
                "train_data_paths is not set, eval_data_paths must be set"
            )
            self.data_loader = None
            self.data_iter = None
        else:
            self.data_loader, self.data_config = self.build_dataloader(
                self.cfg.data.train_data_paths, eval_dataset=False
            )
            self.data_iter = iter(self.data_loader)

        if self.cfg.data.get("eval_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.eval_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.global_step = 0

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

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # reset the eval_data_iter
        eval_data_iter = iter(self.eval_data_loader)

        with self.worker_timer():
            eval_step = len(eval_data_iter)
            eval_pbar = tqdm(
                initial=0,
                total=eval_step,
                desc="Evaluate Step",
                dynamic_ncols=True,
            )
            self.model.eval()
            total = eval_step
            correct = 0

            # get the next batch
            for _ in range(eval_step):
                correct += self.get_eval_model_output(next(eval_data_iter))
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def run_training(self):
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            metrics = {}
            avg_loss = 0.0

            for idx in range(self.gradient_accumulation):
                # set the gradient accumulation backward_ctx
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    if hasattr(self.data_loader, "sampler") and hasattr(
                        self.data_loader.sampler, "set_epoch"
                    ):
                        self.data_loader.sampler.set_epoch(self.global_step)
                    self.data_iter = iter(self.data_loader)

                losses = self.get_train_model_output(batch)

                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(
                        losses, device=self.device, dtype=torch.float32
                    )
                loss = losses.mean()

                loss = loss / self.gradient_accumulation
                avg_loss += loss.item()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            # in one step do the optimizer step
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": avg_loss,
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    @abstractmethod
    def build_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError
