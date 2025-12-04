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

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.sft import SFTDataset
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import ScopedTimer, all_reduce_dict
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import (
    append_to_dict,
)
from rlinf.utils.placement import HybridComponentPlacement


class FSDPSFTWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig, placement: HybridComponentPlacement):
        Worker.__init__(self)
        super().__init__(cfg.sft)

        self.cfg = cfg.sft
        self.placement = placement

        # Initialize dataset
        self.dataset = SFTDataset(self.cfg.data, self.tokenizer)

        # Create distributed sampler
        self.distributed_sampler = DistributedSampler(
            self.dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=getattr(self.cfg, "shuffle", True),
        )

        # Initialize dataloader
        self.dataloader = StatefulDataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=self.distributed_sampler,
            num_workers=getattr(self.cfg.data, "num_workers", 0),
            pin_memory=getattr(self.cfg.data, "pin_memory", False),
            persistent_workers=getattr(self.cfg.data, "persistent_workers", False),
            prefetch_factor=getattr(self.cfg.data, "prefetch_factor", 2),
            num_prefetch_batches=getattr(self.cfg.data, "num_prefetch_batches", 1),
        )

        if self._rank == 0:
            self.metric_logger = MetricLogger(self.cfg.log_dir)
        else:
            self.metric_logger = None

        # Training state
        self.global_step = 0
        self.epoch = 0

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

    def model_provider_func(self):
        model = get_model(self.cfg.sft.checkpoint_load_path, self.cfg.sft.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def init_worker(self):
        self.setup_model_and_optimizer()

    def fit(self):
        """Main training loop"""
        num_epochs = getattr(self.cfg, "num_epochs", 1)

        self.gradient_accumulation = (
            self.cfg.sft.global_batch_size
            // self.cfg.sft.micro_batch_size
            // self._world_size
        )

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.distributed_sampler.set_epoch(epoch)

            for step, global_batch in enumerate(self.dataloader):
                # split global_batch into micro_batches
                micro_batches = get_iterator_k_split(
                    global_batch,
                    self.cfg.sft.global_batch_size // self.cfg.actor.micro_batch_size,
                )

                metrics = {}
                with self.timer("step"):
                    for idx, m_batch in enumerate(micro_batches):
                        backward_ctx = self.before_micro_batch(
                            self.model,
                            is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                        )
                        m_batch = self._prepare_batch(m_batch)

                        outputs = self.model(**m_batch)
                        loss = (
                            outputs.loss
                            if hasattr(outputs, "loss")
                            else self._compute_loss(outputs, m_batch["labels"])
                        )

                        # scale loss for gradient accumulation and backprop
                        loss = loss / self.gradient_accumulation
                        with backward_ctx:
                            self.grad_scaler.scale(loss).backward()

                        mbs_metrics_data = {
                            "loss": loss.detach(),
                        }

                        append_to_dict(metrics, mbs_metrics_data)

                    # Optimizer step
                    grad_norm, lr_list = self.optimizer_step()

                    # Update learning rate scheduler if exists
                    if hasattr(self, "lr_scheduler"):
                        self.lr_scheduler.step()

                    # aggregate metrics across micro-batches
                    mean_metric_dict = {
                        key: torch.mean(torch.stack(value))
                        for key, value in metrics.items()
                    }
                    mean_metric_dict = all_reduce_dict(
                        mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                    )

                    mean_metric_dict["grad_norm"] = float(grad_norm)
                    mean_metric_dict["lr"] = lr_list[0]

                self.global_step += 1

                # Logging
                if self.global_step % getattr(self.cfg, "log_interval", 100) == 0:
                    print(
                        f"[Rank {self._rank}] Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}"
                    )

                # Save checkpoint
                if self.global_step % getattr(self.cfg, "save_interval", 1000) == 0:
                    self.save_checkpoint(
                        save_base_path=getattr(self.cfg, "save_dir", "./checkpoints"),
                        step=self.global_step,
                    )

                time_metrics = self.timer.consume_durations()
                if self.metric_logger is not None:
                    self.metric_logger.log(time_metrics, self.global_step)
                    self.metric_logger.log(mean_metric_dict, self.global_step)

    def _prepare_batch(self, batch):
        """Prepare batch for training"""
        if isinstance(batch, list):
            # Handle batched data
            batch = {
                "input_ids": torch.tensor(
                    [item["input_ids"] for item in batch], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    [item["attention_mask"] for item in batch], dtype=torch.long
                ),
                "labels": torch.tensor(
                    [item["labels"] for item in batch], dtype=torch.long
                ),
            }

        # Move to device
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        return batch

    def _compute_loss(self, outputs, labels):
        """Compute cross-entropy loss"""
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss
