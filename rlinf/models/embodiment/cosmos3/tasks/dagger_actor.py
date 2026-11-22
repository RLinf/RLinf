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
from rlinf.utils.metric_utils import append_to_dict
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

    def _stage_lerobot_batch(self, batch):
        if self.enable_drq:
            return super()._stage_lerobot_batch(batch)
        return Cosmos3DAggerAdapter.stage_lerobot_batch(batch, self.device)

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

    def _record_sft_update(self, metrics, update_metrics):
        # Cosmos commits its step and EMA with the optimizer, before the loader resets.
        append_to_dict(metrics, update_metrics)

    def _finish_sft_updates(self, metrics):
        self.torch_platform.synchronize()
        torch.distributed.barrier()
        self.torch_platform.empty_cache()
        return self.process_train_metrics(metrics)

    def load_checkpoint(self, load_base_path):
        super().load_checkpoint(load_base_path)
        # The scheduler advances only when the native optimizer update succeeds.
        self.update_step = max(0, int(self.lr_scheduler.last_epoch))
        self.optimizer_steps = self.update_step
        self.model.set_global_step(self.update_step, update_ema=False)
