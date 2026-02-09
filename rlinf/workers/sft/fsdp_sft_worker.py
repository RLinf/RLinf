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
from abc import abstractmethod

import torch
from omegaconf import DictConfig

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # before load dataloader should build the tokenizer
        self.tokenizer = self.build_tokenizer()

        assert self.cfg.data.get("train_data_paths") is not None, (
            "train_data_paths is not set"
        )
        self.data_loader, self.data_config = self.build_dataloader(
            self.cfg.data.train_data_paths, eval_dataset=False
        )
        if self.cfg.data.get("eval_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.eval_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.data_iter = iter(self.data_loader)
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

    @abstractmethod
    def build_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def run_eval(self):
        raise NotImplementedError

    @abstractmethod
    def run_training(self):
        raise NotImplementedError
