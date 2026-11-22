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

"""FSDP DAgger training for a Cosmos3 student."""

import os
from pathlib import Path

import numpy as np
import torch

from rlinf.models import get_model
from rlinf.models.embodiment.cosmos3 import load_inference_checkpoint
from rlinf.models.embodiment.cosmos3.datasets.dataset import (
    Cosmos3RollingLeRobotDataset,
)
from rlinf.models.embodiment.cosmos3.tasks.dagger import Cosmos3DAggerAdapter
from rlinf.scheduler import Worker
from rlinf.utils import drq
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class Cosmos3DAggerActor(EmbodiedDAGGERFSDPPolicy):
    """Use the shared DAgger collector and train with the native Cosmos SFT loss."""

    def model_provider_func(self) -> torch.nn.Module:
        checkpoint = self.cfg.runner.get("ckpt_path")
        if not checkpoint:
            return super().model_provider_func()
        model = get_model(self.cfg.actor.model)
        load_inference_checkpoint(model, checkpoint)
        return model

    def _build_lerobot_dataset(self):
        cfg = self._online_lerobot_cfg()
        self._lerobot_resume_num_workers = int(cfg.get("lerobot_num_workers", 0))
        if self._lerobot_resume_num_workers < 0:
            raise ValueError(
                "algorithm.dagger.online_lerobot.lerobot_num_workers must be non-negative."
            )
        only_expert = self.cfg.algorithm.dagger.get("only_save_expert", False)
        self.dataset = Cosmos3RollingLeRobotDataset(
            root_dir=cfg.data_path,
            min_frames=cfg.get("min_frames", 1),
            wait_interval_s=cfg.get("wait_interval_s", 10.0),
            require_all_intervene=only_expert,
            window_size=cfg.get("rolling_lerobot_window_size"),
            in_memory_mode=True,
            fps=int(cfg.get("fps", 10)),
            **Cosmos3DAggerAdapter.lerobot_window_options(
                self.cfg.actor.model, cfg, only_expert
            ),
        )

    def _discover_lerobot_resume_shards(self) -> list[dict]:
        # Incomplete shard directories also occupy IDs, even though they cannot be read.
        rank_dir = Path(self._online_lerobot_cfg().data_path) / f"rank_{self._rank}"
        for child in rank_dir.glob("id_*"):
            if not child.is_dir():
                continue
            try:
                shard_id = int(child.name.removeprefix("id_"))
            except ValueError:
                continue
            self._next_lerobot_archive_id = max(
                self._next_lerobot_archive_id, shard_id + 1
            )
        return super()._discover_lerobot_resume_shards()

    @Worker.timer("prepare_sft_inputs")
    def _prepare_sft_batch(self, batch):
        return super()._prepare_sft_batch(batch)

    def set_global_step(self, global_step: int) -> None:
        """Publish the rollout version. The optimizer owns the model step."""
        self.version = global_step

    def update_buffer_one_epoch(self):
        """Run one replay-buffer update epoch for DAgger."""
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        with self.worker_timer("sample"):
            global_batch = self.replay_buffer.sample(
                num_chunks=global_batch_size_per_rank
            )

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        for idx, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[idx] = batch

        return self._update_sft_batches(
            (batch["forward_inputs"] for batch in train_micro_batch_list),
            self.gradient_accumulation,
        )

    def update_lerobot_one_epoch(self):
        """Run one LeRobot update epoch."""

        with self.worker_timer("prepare_micro_batches"):
            with self._lerobot_loader_lock:
                num_batches = len(self._lerobot_loader)
                train_micro_batch_list = [
                    next(self._lerobot_iter) for _ in range(num_batches)
                ]
            for idx, batch in enumerate(train_micro_batch_list):
                if self.enable_drq:
                    batch = put_tensor_device(batch, device=self.device)
                    drq.apply_drq(batch["curr_obs"], pad=4)
                else:
                    batch = Cosmos3DAggerAdapter.stage_lerobot_batch(batch, self.device)
                train_micro_batch_list[idx] = batch

        result = self._update_sft_batches(train_micro_batch_list, num_batches)
        with self._lerobot_loader_lock:
            self._data_epoch += 1
            if hasattr(self._lerobot_loader.sampler, "set_epoch"):
                self._lerobot_loader.sampler.set_epoch(self._data_epoch)
            self._lerobot_iter = iter(self._lerobot_loader)
        return result

    def _update_sft_batches(self, batches, num_batches):
        self.optimizer.zero_grad()
        logged_batches = []
        for idx, batch in enumerate(batches):
            backward_ctx = self.before_micro_batch(
                self.model, is_last_micro_batch=(idx + 1) == num_batches
            )
            with backward_ctx:
                with self.amp_context:
                    loss, step_metrics = Cosmos3DAggerAdapter.loss_and_metrics(
                        self.forward_actor(batch)
                    )
                loss = loss / num_batches
                self.grad_scaler.scale(loss).backward()
            logged_batches.append((loss.detach().reshape(()), step_metrics))

        grad_norm, _ = self.optimizer_step()
        successful = bool(torch.isfinite(torch.as_tensor(grad_norm)).all())
        if successful:
            self.lr_scheduler.step()
            self.update_step += 1
            self.model.set_global_step(self.update_step)
        self.optimizer.zero_grad(set_to_none=True)
        # Defer host reads until the optimizer has finished with the gradients.
        actor_losses = []
        metrics = {}
        for scaled_loss, step_metrics in logged_batches:
            scalars = [scaled_loss, *step_metrics.values()]
            if all(
                type(value) is torch.Tensor and value.device == scaled_loss.device
                for value in scalars
            ):
                loss_value, *metric_values = torch.stack(scalars).tolist()
            else:
                loss_value, *metric_values = [value.item() for value in scalars]
            actor_losses.append(loss_value * num_batches)
            append_to_dict(metrics, dict(zip(step_metrics, metric_values, strict=True)))
        result = {key: float(np.mean(values)) for key, values in metrics.items()}
        result.update(
            {
                "dagger/update_skipped": float(not successful),
                "dagger/actor_loss": np.mean(actor_losses),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
                "actor/grad_norm": grad_norm,
            }
        )
        return result

    @Worker.timer("run_training")
    def run_training(self):
        """Run DAgger updates with replay-buffer samples."""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        if not self.enable_online_lerobot:
            min_buffer_size = self.cfg.algorithm.replay_buffer.get(
                "min_buffer_size", 100
            )
            if not self.replay_buffer.is_ready(min_buffer_size):
                self.log_on_first_rank(
                    f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
                )
                return {}
        elif self._skip_lerobot_training():
            return {}
        return self._run_sft_updates()

    def _run_sft_updates(self):
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch()
            append_to_dict(metrics, metrics_data)

        self.torch_platform.synchronize()
        torch.distributed.barrier()
        self.torch_platform.empty_cache()
        return self.process_train_metrics(metrics)

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )
        # The scheduler advances only when the native optimizer update succeeds.
        self.update_step = max(0, int(self.lr_scheduler.last_epoch))
        self.optimizer_steps = self.update_step
        self.model.set_global_step(self.update_step, update_ema=False)

        if not self.enable_online_lerobot:
            buffer_load_path = os.path.join(
                load_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.load_checkpoint(buffer_load_path)
