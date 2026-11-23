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

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.models import get_model
from rlinf.models.embodiment.cosmos3 import load_inference_checkpoint
from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class Cosmos3DAggerTraining:
    """Cosmos SFT operations shared by synchronous and asynchronous DAgger actors."""

    @staticmethod
    def lerobot_window_options(
        model_cfg: DictConfig, online_cfg: DictConfig, require_all_intervene: bool
    ) -> dict:
        horizon = online_cfg.get("training_action_horizon", model_cfg.num_action_chunks)
        joint = model_cfg.get("sft_objective") == "joint"
        window_size = online_cfg.get("rolling_lerobot_window_size")
        return {
            "chunk_size": horizon,
            "vision_horizon": horizon if joint else 0,
            "require_full_window": joint,
            "keep_latest_shard": True,
            "keep_uint8_images": (
                joint
                and not require_all_intervene
                and window_size is not None
                and int(window_size) > 0
            ),
        }

    @staticmethod
    def stage_lerobot_batch(
        batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        # Video packing moves pixels. Sequence plans read the padding mask on the CPU.
        prepared = {
            key: batch[key]
            for key in (
                "task",
                "image",
                "wrist_image",
                "vision_images",
                "vision_wrist_images",
                "vision_valid",
                "actions_is_pad",
            )
            if key in batch
        }
        if "actions" in batch:
            prepared["actions"] = put_tensor_device(batch["actions"], device=device)
        return prepared

    @staticmethod
    def loss_and_metrics(
        output: torch.Tensor | dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Keep the native loss graph and snapshot metrics for later reporting."""
        if not isinstance(output, dict):
            return output, {}
        metrics = {
            f"dagger/{key}": value.detach().to(torch.float32, copy=True).reshape(())
            for key, value in output.items()
            if key != "loss" and torch.is_tensor(value) and value.numel() == 1
        }
        return output["loss"], metrics

    @staticmethod
    def load_student(model_cfg: DictConfig, checkpoint: str) -> torch.nn.Module:
        model = get_model(model_cfg)
        load_inference_checkpoint(model, checkpoint)
        return model

    @staticmethod
    def next_archive_id(rank_dir: Path, next_id: int) -> int:
        # Reserve incomplete shard IDs before inspecting any archived episodes.
        for child in rank_dir.glob("id_*"):
            if not child.is_dir():
                continue
            try:
                shard_id = int(child.name.removeprefix("id_"))
            except ValueError:
                continue
            next_id = max(next_id, shard_id + 1)
        return next_id

    @staticmethod
    @Worker.timer("prepare_sft_inputs")
    def prepare_sft_batch(actor: EmbodiedDAGGERFSDPPolicy, batch: dict):
        return actor._prepare_model_inputs(batch)

    @staticmethod
    def update_sft_batches(
        actor: EmbodiedDAGGERFSDPPolicy, batches: Iterable[dict], num_batches: int
    ) -> dict:
        actor.optimizer.zero_grad()
        logged_batches = []
        for idx, batch in enumerate(batches):
            backward_ctx = actor.before_micro_batch(
                actor.model, is_last_micro_batch=(idx + 1) == num_batches
            )
            with backward_ctx:
                with actor.amp_context:
                    loss, step_metrics = Cosmos3DAggerTraining.loss_and_metrics(
                        actor.forward_actor(batch)
                    )
                loss = loss / num_batches
                actor.grad_scaler.scale(loss).backward()
            logged_batches.append((loss.detach().reshape(()), step_metrics))

        grad_norm, _ = actor.optimizer_step()
        successful = bool(torch.isfinite(torch.as_tensor(grad_norm)).all())
        if successful:
            actor.lr_scheduler.step()
            actor.update_step += 1
            actor.model.set_global_step(actor.update_step)
        actor.optimizer.zero_grad(set_to_none=True)
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
                "actor/lr": actor.optimizer.param_groups[0]["lr"],
                "actor/grad_norm": grad_norm,
            }
        )
        return result

    @staticmethod
    def restore_update_step(actor: EmbodiedDAGGERFSDPPolicy) -> None:
        # The scheduler advances only when the native optimizer update succeeds.
        actor.update_step = max(0, int(actor.lr_scheduler.last_epoch))
        actor.optimizer_steps = actor.update_step
        actor.model.set_global_step(actor.update_step, update_ema=False)
