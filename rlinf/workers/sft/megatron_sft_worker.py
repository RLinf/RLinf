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

import json
import logging
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import get_seq_length
from rlinf.hybrid_engines.megatron.megatron_model_manager import MegatronModelManager
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import (
    all_reduce_dict,
    vocab_parallel_log_probs_from_logits,
)
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.train_utils import set_eval, set_sync_funcs, set_train
from rlinf.utils.utils import clear_memory, configure_batch_sizes


class MegatronSftWorker(MegatronModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.global_batch_size = int(self.cfg.actor.global_batch_size)
        self.micro_batch_size = int(self.cfg.actor.micro_batch_size)
        self.eval_batch_size = int(self.cfg.actor.get("eval_batch_size", 1))

        self.dp_size = parallel_state.get_data_parallel_world_size()
        assert self.global_batch_size % (self.micro_batch_size * self.dp_size) == 0, (
            "global_batch_size is not divisible by micro_batch_size * data_parallel_size"
        )

        configure_batch_sizes(
            rank=torch.distributed.get_rank(),
            mbs=self.micro_batch_size,
            gbs=self.global_batch_size,
            dp=self.dp_size,
        )
        self.gradient_accumulation = get_num_microbatches()

        # Megatron offload knobs (optional)
        self.offload_weight = bool(
            self.cfg.actor.get(
                "offload_weight", self.cfg.actor.get("enable_offload", False)
            )
        )
        self.offload_grad = bool(
            self.cfg.actor.get("offload_grad", self.offload_weight)
        )
        self.offload_optimizer = bool(
            self.cfg.actor.get(
                "offload_optimizer", self.cfg.actor.get("enable_offload", False)
            )
        )

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
        self._data_epoch = 0
        self._data_iter_offset = 0

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.offload_weight:
            self.offload_model_weights_and_grad(offload_grad=self.offload_grad)
        if self.offload_optimizer:
            self.offload_megatron_optimizer()

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def _onload_for_train(self):
        if self.offload_weight:
            self.onload_model_weights_and_grad(load_grad=self.offload_grad)
        if self.offload_optimizer:
            self.onload_megatron_optimizer()

    def _offload_after_train(self):
        if self.offload_weight:
            self.offload_model_weights_and_grad(offload_grad=self.offload_grad)
        if self.offload_optimizer:
            self.offload_megatron_optimizer()

    def _next_micro_batch(self) -> dict[str, Any]:
        try:
            batch = next(self.data_iter)
            self._data_iter_offset += 1
            return batch
        except StopIteration:
            self._data_epoch += 1
            logging.info(
                f"[INFO] data_iter exhausted, reset iterator self._data_epoch {self._data_epoch}"
            )
            if hasattr(self.data_loader, "sampler") and hasattr(
                self.data_loader.sampler, "set_epoch"
            ):
                self.data_loader.sampler.set_epoch(self._data_epoch)
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
            self._data_iter_offset = 1
            return batch

    def _merge_value(self, values: list[Any]) -> Any:
        v0 = values[0]
        if torch.is_tensor(v0):
            return torch.cat(values, dim=0)
        if isinstance(v0, dict):
            return {k: self._merge_value([v[k] for v in values]) for k in v0}
        if isinstance(v0, (list, tuple)):
            merged = [self._merge_value([v[i] for v in values]) for i in range(len(v0))]
            return type(v0)(merged)
        return v0

    def _build_global_batch(self) -> dict[str, Any]:
        micro_batches = [
            self._next_micro_batch() for _ in range(self.gradient_accumulation)
        ]
        return self._merge_value(micro_batches)

    def _run_megatron_train_step(
        self, global_batch: dict[str, Any]
    ) -> dict[str, float]:
        set_train(self)
        set_sync_funcs(self, forward_only=False)

        for model_chunk in self.model:
            if hasattr(model_chunk, "zero_grad_buffer"):
                model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        batch_iter = get_iterator_k_split(
            global_batch, num_splits=get_num_microbatches()
        )
        fwd_bwd_func = get_forward_backward_func()
        forward_outputs = fwd_bwd_func(
            forward_step_func=self.get_forward_step_func(),
            data_iterator=self.make_data_iterator_list(batch_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=False,
            seq_length=get_seq_length(global_batch),
            micro_batch_size=1,
            collect_non_loss_data=False,
        )

        metrics: dict[str, float] = {}
        if forward_outputs:
            keys = forward_outputs[0].keys()
            for key in keys:
                metric_mean = torch.stack([m[key] for m in forward_outputs]).mean()
                metrics[key] = float(metric_mean.detach().cpu().item())

        # One optimizer step per global batch.
        success, grad_norm, _, lr = self.optimizer_step(
            increment=self.global_batch_size
        )

        if "loss" not in metrics:
            # fallback key for some loss funcs
            if "lm_loss" in metrics:
                metrics["loss"] = metrics["lm_loss"]

        append_to_dict(
            metrics,
            {
                "learning_rate": float(lr) if lr is not None else float("nan"),
                "grad_norm": (
                    float(grad_norm.detach().cpu().item())
                    if torch.is_tensor(grad_norm)
                    else (float(grad_norm) if grad_norm is not None else float("nan"))
                ),
                "update_success": int(success),
            },
        )
        return metrics

    def run_training(self):
        with self.worker_timer():
            self._onload_for_train()
            try:
                clear_memory()
                global_batch = self._build_global_batch()
                train_metrics = self._run_megatron_train_step(global_batch)

                if self.global_step > 0 and self.global_step % 1000 == 0:
                    clear_memory()

                # keep the same reduction behavior as FSDP worker
                reduced = {
                    k: np.mean(v) if isinstance(v, list) else v
                    for k, v in train_metrics.items()
                }
                reduced = all_reduce_dict(reduced, op=torch.distributed.ReduceOp.AVG)
                return reduced
            finally:
                self._offload_after_train()

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # If weights are offloaded, onload for eval.
        if self.offload_weight:
            self.onload_model_weights_and_grad(load_grad=False)

        try:
            eval_data_iter = iter(self.eval_data_loader)
            with self.worker_timer():
                eval_step = len(eval_data_iter)
                eval_pbar = tqdm(
                    initial=0,
                    total=eval_step,
                    desc="Evaluate Step",
                    dynamic_ncols=True,
                )
                set_eval(self)

                total = eval_step * self.eval_batch_size
                correct = 0
                for _ in range(eval_step):
                    correct += self.get_eval_model_output(next(eval_data_iter))
                    eval_pbar.update(1)

                metrics = {"eval_accuracy": float(correct / max(1, total))}
                metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
                return metrics
        finally:
            if self.offload_weight:
                self.offload_model_weights_and_grad(offload_grad=False)

    @abstractmethod
    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        raise NotImplementedError

    @abstractmethod
    def get_forward_step_func(self):
        """Return Megatron forward_step_func(dataloader_iter, model)."""
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError


class MegatronVlmSftWorker(MegatronSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._eval_warned = False

    def _save_data_state(self, save_path: str):
        state = {
            "data_epoch": self._data_epoch,
            "data_iter_offset": self._data_iter_offset,
        }
        with open(os.path.join(save_path, "data_state.json"), "w") as f:
            json.dump(state, f)

    def save_checkpoint(self, save_path: str, step: int = 0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            self._save_data_state(save_path)

    def _load_data_state(self, load_path: str):
        path = os.path.join(load_path, "data_state.json")
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            state = json.load(f)

        self._data_epoch = int(state.get("data_epoch", 0))
        self._data_iter_offset = int(state.get("data_iter_offset", 0))

        if hasattr(self.data_loader, "sampler") and hasattr(
            self.data_loader.sampler, "set_epoch"
        ):
            self.data_loader.sampler.set_epoch(self._data_epoch)

        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.data_iter)
            except StopIteration:
                self._data_epoch += 1
                if hasattr(self.data_loader, "sampler") and hasattr(
                    self.data_loader.sampler, "set_epoch"
                ):
                    self.data_loader.sampler.set_epoch(self._data_epoch)
                self.data_iter = iter(self.data_loader)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        self._load_data_state(load_path)

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.actor.model.model_path,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.QWEN2_5_VL_SFT,
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]:
            from torch.utils.data import DataLoader, DistributedSampler

            from rlinf.data.datasets import sft_collate_fn
            from rlinf.data.datasets.vlm import VLMDatasetRegistry

            if not hasattr(self, "tokenizer"):
                self.tokenizer = self.build_tokenizer()

            dataset_name = self.cfg.data.get("dataset_name", "robo2vlmsft")
            train_dataset = VLMDatasetRegistry.create(
                dataset_name,
                data_paths=data_paths,
                config=self.cfg,
                tokenizer=self.tokenizer,
                eval_dataset=eval_dataset,
            )

            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=self.cfg.data.get("shuffle", True),
                    seed=self.cfg.data.get("seed", 42),
                    drop_last=True,
                )
            else:
                sampler = None

            batch_size = (
                self.micro_batch_size
                if not eval_dataset
                else self.cfg.actor.get("eval_batch_size", 1)
            )
            data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.cfg.data.get("num_workers", 4),
                drop_last=True,
                collate_fn=sft_collate_fn,
            )
            logging.info(
                f"Build data loader from {data_paths} with {len(train_dataset)} samples"
            )

            data_config = {
                "dataset_name": dataset_name,
                "num_samples": len(train_dataset),
            }
            return data_loader, data_config

        raise KeyError(
            f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
        )

    def _prepare_batch_for_megatron(self, batch: dict[str, Any]) -> dict[str, Any]:
        input_ids = batch["prompt"].cuda()
        attention_mask = batch["attention_mask"].to(
            device=input_ids.device, dtype=torch.bool
        )
        label_mask = batch["label_mask"].to(device=input_ids.device, dtype=torch.bool)

        multi_modal_inputs = batch.get("multi_modal_inputs", {})
        if multi_modal_inputs is None:
            multi_modal_inputs = {}
        for k, v in list(multi_modal_inputs.items()):
            if torch.is_tensor(v):
                multi_modal_inputs[k] = v.to(device=input_ids.device)

        # position ids under left padding.
        position_ids = torch.cumsum(attention_mask.long(), dim=-1) - 1
        position_ids = position_ids.clamp_min(0)

        # Next-token labels and mask.
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = 0

        token_valid_mask = attention_mask.clone()
        token_valid_mask[:, :-1] = attention_mask[:, 1:]
        token_valid_mask[:, -1] = False

        # label_mask=True means prompt tokens (ignore); we train on answer tokens only.
        loss_mask = token_valid_mask & (~label_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
        }

    def get_forward_step_func(self):
        def forward_output_and_loss_func(dataloader_iter, model):
            raw_batch = next(dataloader_iter)
            batch = self._prepare_batch_for_megatron(raw_batch)

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]
            multi_modal_inputs = batch["multi_modal_inputs"]

            def logits_processor(logits, labels, loss_mask):
                log_probs = vocab_parallel_log_probs_from_logits(logits, labels)
                log_probs = log_probs.masked_fill(~loss_mask, 0.0)
                return {"log_probs": log_probs}

            output = self.custom_forward(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args={"labels": labels, "loss_mask": loss_mask},
                temperature=1.0,
            )

            if not self.return_loss:

                def id_func(non_loss_output, non_loss_data=True):
                    return non_loss_output

                if isinstance(output, dict):
                    # Keep tensor output in forward_only path for Megatron schedule.
                    return output["log_probs"], id_func
                return output, id_func

            def loss_func(non_loss_output):
                log_probs = non_loss_output["log_probs"]
                nll = -log_probs
                denom = loss_mask.float().sum().clamp_min(1.0)
                loss = (nll * loss_mask.float()).sum() / denom

                metrics_data = {
                    "lm_loss": loss.detach(),
                }
                for k, v in metrics_data.items():
                    metrics_data[k] = average_losses_across_data_parallel_group([v])

                return loss, metrics_data

            return output, loss_func

        return forward_output_and_loss_func

    def get_eval_model_output(self, batch: dict[str, Any]):
        # TODO(agent): Megatron VLM generation path is not available in current codebase.
        # Use 0 placeholder to keep training/eval loop alive.
        if not self._eval_warned and self._rank == 0:
            logging.warning(
                "MegatronVlmSftWorker.get_eval_model_output() currently uses a placeholder "
                "(always returns 0). Please integrate a Megatron generation path for true eval accuracy."
            )
            self._eval_warned = True
        return 0
