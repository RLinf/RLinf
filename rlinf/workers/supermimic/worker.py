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

from __future__ import annotations

import os

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.workers.supermimic.trainer import SuperMimicTrainer


class SuperMimicWorker(Worker):
    """One GPU process: sim, replay, world model, policy, and critic."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.trainer: SuperMimicTrainer | None = None

    def init_worker(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.trainer = SuperMimicTrainer(self.cfg, device=device)
        import torch.nn as nn

        dynamics = [
            module.out_features
            for module in self.trainer.models.dynamics.net
            if isinstance(module, nn.Linear)
        ]
        self.log_info(
            f"SuperMimic worker ready on {device}, "
            f"goal_dim={self.cfg.algorithm.goal_dim}, "
            f"obs_dim={self.cfg.algorithm.obs_dim}, "
            f"k={self.cfg.algorithm.k}, H={self.cfg.algorithm.H}, "
            f"lr={self.trainer.opt_actor.param_groups[0]['lr']}, "
            f"dynamics_out={dynamics}, "
            f"num_envs={self.cfg.env.num_envs}"
        )

    def run_iteration(self) -> dict[str, float]:
        assert self.trainer is not None
        return self.trainer.run_iteration()

    def run_until_updates(self, n_updates: int) -> dict[str, float]:
        assert self.trainer is not None
        return self.trainer.run_until_updates(n_updates)

    def evaluate(self) -> dict[str, float]:
        assert self.trainer is not None
        return self.trainer.evaluate()

    def save_checkpoint(self, directory: str) -> None:
        assert self.trainer is not None
        self.trainer.save_checkpoint(directory)

    def load_checkpoint(self, directory: str) -> None:
        assert self.trainer is not None
        self.trainer.load_checkpoint(directory)
