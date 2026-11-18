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
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.qam import qam_update
from rlinf.data.datasets.ogbench import OGBenchDataset
from rlinf.models import get_model
from rlinf.scheduler import Worker
from rlinf.utils.utils import collect_param_names_need_sync
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class QAMPolicyWorker(EmbodiedFSDPActor):
    """Train QAM with native or FSDP model execution.

    actor.use_fsdp_wrap selects the execution mode. Both modes use the
    same worker, checkpoint interface, and rollout weight synchronizer.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.global_step = 0
        self.seed = int(cfg.actor.seed)
        self._use_fsdp_wrap = bool(cfg.actor.get("use_fsdp_wrap", False))
        self.batch_size = None
        self.dataset = None
        self.generator = None

    def init_worker(self):
        if not self._use_fsdp_wrap and self._world_size != 1:
            raise ValueError(
                "QAM native execution requires one actor rank. Set "
                "actor.use_fsdp_wrap=true for multi-rank training."
            )

        global_batch_size = int(self.cfg.actor.global_batch_size)
        if global_batch_size % self._world_size != 0:
            raise ValueError(
                "actor.global_batch_size must be divisible by actor world size, "
                f"got global_batch_size={global_batch_size}, world_size={self._world_size}."
            )
        self.batch_size = global_batch_size // self._world_size
        if int(self.cfg.actor.micro_batch_size) != self.batch_size:
            raise ValueError(
                "actor.micro_batch_size must equal the per-rank QAM batch size, "
                f"got {self.cfg.actor.micro_batch_size}, expected {self.batch_size}."
            )

        rank_seed = self.seed + self._rank
        torch.manual_seed(rank_seed)
        np.random.seed(rank_seed)
        self.generator = torch.Generator(device=self.device).manual_seed(rank_seed)
        self.dataset = OGBenchDataset.from_task(
            self.cfg.data.task_name,
            rank_seed,
            self.cfg.data.get("dataset_dir", None),
        )

        model_cfg = self.cfg.actor.model
        obs_dim = self.dataset.data["observations"].shape[-1]
        action_dim = self.dataset.data["actions"].shape[-1]
        if int(model_cfg.obs_dim) != obs_dim or int(model_cfg.action_dim) != action_dim:
            raise ValueError(
                f"QAM model dimensions ({model_cfg.obs_dim}, {model_cfg.action_dim}) "
                f"do not match OGBench data ({obs_dim}, {action_dim})."
            )

        module = get_model(model_cfg).to(self.device)
        module.discount = float(self.cfg.algorithm.gamma)
        self.param_names_need_sync = collect_param_names_need_sync(module)
        if self._use_fsdp_wrap:
            self.model = self._strategy.wrap_model(
                model=module,
                device_mesh=self._device_mesh,
            )
        else:
            self.model = module
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.cfg.actor.optim.lr),
            eps=1e-8,
        )
        self.model.train()
        self._qam_module().target_actor_slow.eval()
        self._qam_module().target_critic.eval()
        self.log_info(
            f"QAM seed={self.seed}, rank={self._rank}/{self._world_size}, "
            f"device={self.device}, dataset={self.dataset.size}, "
            f"use_fsdp_wrap={self._use_fsdp_wrap}"
        )
        return {
            "seed": self.seed,
            "rank": self._rank,
            "dataset_size": self.dataset.size,
            "device": str(self.device),
            "use_fsdp_wrap": self._use_fsdp_wrap,
            "pid": os.getpid(),
        }

    def _qam_module(self):
        return getattr(self.model, "module", self.model)

    def _fsdp_loss(self, batch, generator, noise):
        return self.model(
            batch=batch,
            compute_loss=True,
            generator=generator,
            noise=noise,
        )

    def _fsdp_clip_grad(self):
        return self._strategy.clip_grad_norm_(self.model, 1.0)

    def _fsdp_update_targets(self):
        state_dict = self.get_model_state_dict(
            cpu_offload=False,
            full_state_dict=True,
        )
        tau = float(self._qam_module().cfg.tau)
        for target_prefix, source_prefix in (
            ("target_critic.", "critic."),
            ("target_actor_slow.", "actor_slow."),
        ):
            for target_name in tuple(state_dict):
                if not target_name.startswith(target_prefix):
                    continue
                source_name = source_prefix + target_name[len(target_prefix) :]
                source_value = state_dict.get(source_name)
                target_value = state_dict[target_name]
                if isinstance(source_value, torch.Tensor) and isinstance(
                    target_value, torch.Tensor
                ):
                    state_dict[target_name] = target_value * (1 - tau) + source_value * tau
        self.load_model_with_state_dict(
            state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

    def _update(self):
        module = self._qam_module()
        batch = self.dataset.sample(
            self.batch_size,
            module.horizon,
            module.discount,
        )
        batch = {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}
        if self._use_fsdp_wrap:
            return qam_update(
                self.model,
                self.optimizer,
                batch,
                self.generator,
                loss_fn=self._fsdp_loss,
                target_update=self._fsdp_update_targets,
                clip_grad=self._fsdp_clip_grad,
            )
        return qam_update(self.model, self.optimizer, batch, self.generator)

    @Worker.timer("run_training")
    def run_training(self):
        stop = min(
            self.global_step + int(self.cfg.runner.local_update_steps),
            int(self.cfg.runner.max_steps),
            int(self.cfg.runner.max_epochs),
        )
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
            for key, value in info.items():
                sums[key] = sums.get(key, 0) + value
            self.global_step = step
        metrics = (
            {key: float(value / count) for key, value in sums.items()} if count else {}
        )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Non-finite QAM metrics: {metrics}")
        return {**metrics, "__global_step": self.global_step}

    def set_global_step(self, step):
        # ``OfflineRunner`` sends this synchronization call asynchronously.
        # If a completed update reaches the worker first, do not roll the
        # worker back to the stale step carried by that request.
        self.global_step = max(self.global_step, int(step))

    def get_rollout_sync_version(self):
        return self.global_step

    def get_rollout_state_dict(self):
        if self._use_fsdp_wrap:
            return self.get_model_state_dict(
                cpu_offload=False,
                full_state_dict=False,
            )
        return {key: value.detach() for key, value in self.model.state_dict().items()}

    def save_checkpoint(self, path, global_step):
        if int(global_step) != self.global_step:
            raise ValueError("Checkpoint step must match worker step")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_payload = {
            "generator": self.generator.get_state(),
            "replay": self.dataset.state_dict(),
            "step": self.global_step,
            "seed": self.seed,
            "rank": self._rank,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        if self._use_fsdp_wrap:
            self._strategy.save_checkpoint(
                model=self.model,
                optimizers=[self.optimizer],
                lr_schedulers=[],
                save_path=str(path / "qam_model"),
                save_full_model_weights=False,
                checkpoint_format="local_shard",
            )
            torch.save(state_payload, path / f"state_rank_{self._rank}.pt")
        else:
            state_payload.update(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in self.model.state_dict().items()
                    },
                    "optimizer": self.optimizer.state_dict(),
                }
            )
            temp = path / f"seed_{self.seed}.tmp"
            target = path / f"seed_{self.seed}.pt"
            torch.save(state_payload, temp)
            temp.replace(target)
        return str(path)

    def load_checkpoint(self, path):
        path = Path(path)
        fsdp_path = path / "qam_model"
        if self._use_fsdp_wrap and fsdp_path.exists():
            self._strategy.load_checkpoint(
                model=self.model,
                optimizers=[self.optimizer],
                lr_schedulers=[],
                load_path=str(fsdp_path),
            )
            state = torch.load(
                path / f"state_rank_{self._rank}.pt",
                map_location=self.device,
                weights_only=False,
            )
        else:
            state = torch.load(
                path / f"seed_{self.seed}.pt",
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(state["model"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
        if state["seed"] != self.seed:
            raise ValueError("Checkpoint seed mismatch")
        self.generator.set_state(state["generator"].cpu())
        self.dataset.load_state_dict(state["replay"])
        self.global_step = int(state["step"])
        return self.global_step
