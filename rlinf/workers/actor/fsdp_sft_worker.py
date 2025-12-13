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

import jax
import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

import rlinf.algorithms  # noqa: F401
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
)


class SFTactor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(self._world_size,), mesh_dim_names=["fsdp"]
        )

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.data_config = self.build_dataloader()
        self.data_iter = iter(self.data_loader)

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

    def build_dataloader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def run_training(self):
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"
            assert self.cfg.actor.micro_batch_size == 1, (
                "micro_batch_size must be 1 for SFT, because we use lerobot data loader api"
            )

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}

            for _ in range(self.gradient_accumulation):
                observation, actions = next(self.data_iter)

                observation = jax.tree.map(
                    lambda x: torch.as_tensor(x, device=self.device)
                    .contiguous()
                    .clone(),
                    observation,
                )  # noqa: PLW2901
                actions = actions.to(torch.float32)  # noqa: PLW2901
                actions = actions.to(self.device)  # noqa: PLW2

                self.optimizer.zero_grad(set_to_none=True)
                with self.amp_context:
                    losses = self.model(
                        data={"observation": observation, "actions": actions},
                        mode="sft",
                    )
                    if isinstance(losses, (list, tuple)):
                        losses = torch.stack(losses)
                    elif not isinstance(losses, torch.Tensor):
                        losses = torch.tensor(
                            losses, device=self.device, dtype=torch.float32
                        )
                    loss = losses.mean()

                self.grad_scaler.scale(loss).backward()
                grad_norm, lr_list = self.optimizer_step()
                self.optimizer.zero_grad(set_to_none=True)

                # Collect stats
                lr_value = (
                    lr_list[0]
                    if len(lr_list) > 0
                    else self.optimizer.param_groups[0]["lr"]
                )
                grad_norm_value = (
                    float(grad_norm)
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                append_to_dict(
                    metrics,
                    {
                        "loss": loss.item(),
                        "learning_rate": lr_value,
                        "grad_norm": grad_norm_value,
                    },
                )

                self.lr_scheduler.step()

            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    def set_global_step(self, global_step):
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
