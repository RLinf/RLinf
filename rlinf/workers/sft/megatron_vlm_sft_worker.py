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

import torch

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.training.utils import average_losses_across_data_parallel_group

except (ImportError, ModuleNotFoundError):
    raise "Megatron core was not found."

from omegaconf import DictConfig

from rlinf.config import SupportedModel
from rlinf.data.io_struct import get_seq_length
from rlinf.hybrid_engines.megatron.megatron_model_manager import MegatronModelManager
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.train_utils import set_sync_funcs, set_train
from rlinf.utils.utils import clear_memory, configure_batch_sizes


class MegatronSftWorker(MegatronModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        self.megatron_type = "sft"
        Worker.__init__(self)
        super().__init__(cfg.actor)

        assert self.mbridge is True, (
            "Now MegatronVlmSftWorker only supports Megatron-Bridge"
        )
        self.cfg = cfg
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.global_batch_size = int(self.cfg.actor.global_batch_size)
        self.micro_batch_size = int(self.cfg.actor.micro_batch_size)
        self.eval_batch_size = int(self.cfg.actor.get("eval_batch_size", 1))

        # get the data parallel size and rank to set the dataloader
        self.dp_size = parallel_state.get_data_parallel_world_size()
        self.dp_rank = parallel_state.get_data_parallel_rank()

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]:
            assert self.cfg.actor.model.apply_rope_fusion is False, (
                "apply_rope_fusion must be False for Qwen3 VL SFT, the Mrope answer is Error"
            )

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
        logging.info(f"megatron model rank {self._rank}: {self.model}")

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_max_steps_per_epoch(self):
        if self.data_loader is not None:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        return 0

    def save_checkpoint(
        self,
        save_path: str,
        step: int,
        num_floating_point_operations_so_far: int = 0,
    ) -> None:
        super().save_checkpoint(save_path, step, num_floating_point_operations_so_far)

    def run_training(self):
        with self.worker_timer():
            clear_memory()
            try:
                global_batch = next(self.data_iter)
                self._data_iter_offset += 1
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
                global_batch = next(self.data_iter)
                self._data_iter_offset = 1

            train_metrics = self.get_train_model_output(global_batch)

            return train_metrics

    def run_eval(self):
        with self.worker_timer():
            logging.info("[INFO] MegatronSftWorker.run_eval() is not Supported for now")
            return {}

    def get_train_model_output(self, global_batch: dict[str, Any]) -> dict[str, float]:
        set_train(self)
        set_sync_funcs(self, forward_only=False)

        for model_chunk in self.model:
            if hasattr(model_chunk, "zero_grad_buffer"):
                model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        # move the multi_modal_inputs to tensor, get_iterator_k_split can't handle the dict item
        multi_modal_inputs = global_batch.pop("multi_modal_inputs")

        for k, v in multi_modal_inputs.items():
            global_batch[k] = v

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
            seq_length=get_seq_length(global_batch, batch_tensor_key="prompt"),
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

        metrics.update(
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

    @abstractmethod
    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError


class MegatronVlmSftWorker(MegatronSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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
                    num_replicas=self.dp_size,
                    rank=self.dp_rank,
                    shuffle=self.cfg.data.get("shuffle", True),
                    seed=self.cfg.data.get("seed", 42),
                    drop_last=True,
                )
            else:
                sampler = None

            if eval_dataset:
                batch_size = self.eval_batch_size
            else:
                # for megatron training, the batch size is the micro batch size * gradient accumulation
                # the megatron can padding as same prompt length
                batch_size = self.global_batch_size // self.dp_size

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
            assert len(data_loader) != 0, (
                f"data_loader is not empty, please check the data_path {data_paths}"
            )

            data_config = {
                "dataset_name": dataset_name,
                "num_samples": len(train_dataset),
            }
            return data_loader, data_config

        raise KeyError(
            f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
        )

    def prepare_batch_for_megatron(
        self, batch: dict[str, Any], model
    ) -> dict[str, Any]:
        input_ids = batch["prompt"].cuda()
        # attention_mask save the prompt + answer mask
        attention_mask = batch["attention_mask"].to(
            device=input_ids.device, dtype=torch.bool
        )

        image_grid_thw = batch["image_grid_thw"].to(device=input_ids.device)
        pixel_values = torch.cat(batch["pixel_values"]).to(device=input_ids.device)

        # label_mask now save the prompt mask
        label_mask = batch["label_mask"].to(device=input_ids.device, dtype=torch.bool)

        labels = input_ids.detach().clone().masked_fill(~attention_mask, 0)
        labels = labels.masked_fill(label_mask, 0)
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = 0

        # loss_mask just save the answer mask
        answer_mask = attention_mask & (~label_mask)
        loss_mask = torch.zeros_like(
            answer_mask, dtype=torch.bool, device=input_ids.device
        )
        loss_mask[:, :-1] = answer_mask[:, 1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def get_forward_step_func(self):
        def forward_output_and_loss_func(dataloader_iter, model):
            raw_batch = next(dataloader_iter)
            batch = self.prepare_batch_for_megatron(raw_batch, model)

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]

            def logits_processor(logits, labels, loss_mask):
                per_token_loss = tensor_parallel.vocab_parallel_cross_entropy(
                    logits.float(),
                    labels,
                )
                return {
                    "per_token_loss": per_token_loss,
                    "loss_mask": loss_mask.float(),
                }

            output = self.custom_forward(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                position_ids=None,
                keep_left_padding=True,
                sequence_parallel=self.transformer_config.sequence_parallel,
                image_grid_thw=image_grid_thw,
                temperature=self.cfg.algorithm.sampling_params.temperature,
                logits_processor=logits_processor,
                logits_processor_args={"labels": labels, "loss_mask": loss_mask},
            )

            def loss_func(non_loss_output):
                per_token_loss = non_loss_output["per_token_loss"].view(-1)
                loss_mask = non_loss_output["loss_mask"].view(-1)
                denom = loss_mask.sum().clamp_min(1.0)
                loss = torch.sum(per_token_loss * loss_mask) / denom
                metrics_data = {"lm_loss": loss.detach()}
                for k, v in metrics_data.items():
                    metrics_data[k] = average_losses_across_data_parallel_group([v])
                return loss, metrics_data

            return output, loss_func

        return forward_output_and_loss_func
