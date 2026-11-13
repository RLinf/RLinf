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


import copy
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.qam import qam_update
from rlinf.data.datasets.ogbench import OGBenchDataset
from rlinf.models import get_model
from rlinf.scheduler import Worker


class QAMPolicyWorker(Worker):
    """Own model, optimizer and offline data; runner owns scheduling and evaluation.

    Each run trains one seed; the common OfflineRunner owns evaluation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.global_step = 0
        self.updates = 0

    def init_worker(self):
        if self._world_size != 1:
            raise ValueError("Offline QAM requires one actor worker per seed")
        self.seed = int(self.cfg.actor.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device(
            self.torch_device_type
            if self.torch_platform and self.torch_platform.is_available()
            else "cpu"
        )
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.dataset = OGBenchDataset.from_task(
            self.cfg.data.task_name,
            self.seed,
            self.cfg.data.get("dataset_dir", None),
        )
        model_cfg = self.cfg.actor.model
        obs_dim = self.dataset.data["observations"].shape[-1]
        action_dim = self.dataset.data["actions"].shape[-1]
        if int(model_cfg.obs_dim) != obs_dim or int(model_cfg.action_dim) != action_dim:
            raise ValueError(
                f"QAM model dimensions ({model_cfg.obs_dim}, {model_cfg.action_dim}) "
                f"do not match OGBench data ({obs_dim}, {action_dim})"
            )
        self.model = get_model(model_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.actor.optim.lr,
            eps=1e-8,
        )
        self.log_info(
            f"QAM seed={self.seed}, device={self.device}, dataset={self.dataset.size}"
        )
        return {
            "seed": self.seed,
            "dataset_size": self.dataset.size,
            "device": str(self.device),
            "pid": os.getpid(),
        }

    def _update(self):
        batch = self.dataset.sample(
            self.cfg.actor.global_batch_size,
            self.model.horizon,
            self.model.cfg.discount,
        )
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        info = qam_update(self.model, self.optimizer, batch, self.generator)
        self.updates += 1
        return info

    @Worker.timer("run_training")
    def run_training(self):
        stop = min(
            self.global_step + int(self.cfg.runner.local_update_steps),
            int(self.cfg.runner.max_steps),
            int(self.cfg.runner.max_epochs),
        )
        # Never cross an evaluation or checkpoint boundary in one call.
        for interval in (
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
        ):
            if interval > 0:
                stop = min(stop, (self.global_step // interval + 1) * interval)
        sums, count = {}, 0
        for step in range(self.global_step + 1, stop + 1):
            info = self._update()
            count += 1
            for k, v in info.items():
                sums[k] = sums.get(k, 0) + v
            self.global_step = step
        metrics = {k: float(v / count) for k, v in sums.items()} if count else {}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"Non-finite QAM metrics: {metrics}")
        return {**metrics, "__global_step": self.global_step}

    def set_global_step(self, step):
        if self.global_step != int(step):
            raise ValueError(
                f"Runner step {step} differs from worker step {self.global_step}"
            )

    def get_policy_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def save_checkpoint(self, path, global_step):
        if int(global_step) != self.global_step:
            raise ValueError("Checkpoint step must match worker step")
        path = Path(path) / f"seed_{self.seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.get_policy_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "replay": self.dataset.state_dict(),
            "step": self.global_step,
            "updates": self.updates,
            "seed": self.seed,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(path)
        return str(path)

    def load_checkpoint(self, path):
        path = Path(path) / f"seed_{self.seed}.pt"
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["seed"] != self.seed:
            raise ValueError("Checkpoint seed mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.dataset.load_state_dict(state["replay"])
        self.global_step, self.updates = state["step"], state["updates"]
        return self.global_step
