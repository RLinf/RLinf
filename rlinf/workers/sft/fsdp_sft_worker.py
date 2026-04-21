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

import json
import logging
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils import _pytree
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.utils import generate_with_kv_cache
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import clear_memory
from rlinf.workers.sft.utils import vlm_extract_answer, vlm_normalize_text


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # set the global batch size, micro batch size, eval batch size and gradient accumulation
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.eval_batch_size = self.cfg.actor.get("eval_batch_size", 1)

        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        # if train_data_paths is not set, the code will just eval the model
        if self.cfg.data.get("train_data_paths") is None:
            logging.warning("train_data_paths is not set, will just eval the model")
            assert self.cfg.data.get("val_data_paths") is not None, (
                "train_data_paths is not set, val_data_paths must be set"
            )
            self.data_loader = None
            self.data_iter = None
        else:
            self.data_loader, self.data_config = self.build_dataloader(
                self.cfg.data.train_data_paths, eval_dataset=False
            )
            self.data_iter = iter(self.data_loader)

        if self.cfg.data.get("val_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.val_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.global_step = 0
        # set the dataloader epoch and data iter offset
        self._data_epoch = 0
        self._data_iter_offset = 0

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

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # reset the eval_data_iter
        eval_data_iter = iter(self.eval_data_loader)

        with self.worker_timer():
            eval_step = len(eval_data_iter)
            eval_pbar = tqdm(
                initial=0,
                total=eval_step,
                desc="Evaluate Step",
                dynamic_ncols=True,
            )
            self.model.eval()
            total = eval_step * self.eval_batch_size
            correct = 0

            # get the next batch
            for _ in range(eval_step):
                correct += self.get_eval_model_output(next(eval_data_iter))
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def run_training(self):
        with self.worker_timer():
            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            metrics = {}
            avg_loss = 0.0

            for idx in range(self.gradient_accumulation):
                # set the gradient accumulation backward_ctx
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                try:
                    batch = next(self.data_iter)
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
                    batch = next(self.data_iter)
                    self._data_iter_offset = 1

                losses = self.get_train_model_output(batch)

                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(
                        losses, device=self.device, dtype=torch.float32
                    )
                loss = losses.mean()

                loss = loss / self.gradient_accumulation
                avg_loss += loss.item()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            # in one step do the optimizer step
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            self.lr_scheduler.step()
            lr_value = self.optimizer.param_groups[0]["lr"]
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "loss": avg_loss,
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        observation, actions = next(self.data_iter)

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
            if x is not None
            else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses


class FSDPVlmSftWorker(FSDPSftWorker):
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
        if self.data_loader is not None:
            # run the eval model not to load data_loader ckpt
            self._load_data_state(load_path)

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.actor.model.model_path,
        )
        # set the padding side to left for the tokenizer, QWEN 2.5 VL just use left padding
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

            # vlm sft before load dataloader should build the tokenizer
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

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # hundle the input batch
        correct = 0
        input_ids = batch["prompt"].to(self.device)
        answers = batch["answer"]
        attention_mask = batch["attention_mask"].to(self.device)
        multi_modal_inputs = batch["multi_modal_inputs"]
        for k, v in multi_modal_inputs.items():
            if isinstance(v, list):
                multi_modal_inputs[k] = torch.cat(v, dim=0).to(device=self.device)
            else:
                multi_modal_inputs[k] = v.to(device=self.device)

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos_token_id if eos_token_id is not None else 0)
        )

        with torch.no_grad():
            # use kv cache to generate the text
            # the generate_with_kv_cache() is more efficient than the generate() in utils.py
            generate_ids = generate_with_kv_cache(
                model=self.model,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                amp_context=self.amp_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
            )

        # encode the generated text
        for i in range(len(answers)):
            new_token_ids = generate_ids[i, input_ids.shape[1] :]
            full_pred_text = self.tokenizer.decode(
                new_token_ids.tolist(), skip_special_tokens=False
            )

            pred_text = vlm_extract_answer(
                full_pred_text, self.cfg.actor.model.model_type
            )
            gold_text = answers[i]

            print(
                f"pred_text {vlm_normalize_text(pred_text)} gold_text {vlm_normalize_text(gold_text)}",
                flush=True,
            )
            if vlm_normalize_text(pred_text) == vlm_normalize_text(gold_text):
                correct += 1

        # eval model return the correct number of answers
        return correct

    def get_train_model_output(self, batch: dict[str, Any]):
        # hundle the input batch
        input_ids = batch["prompt"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
        multi_modal_inputs = batch["multi_modal_inputs"]
        for k, v in multi_modal_inputs.items():
            if isinstance(v, list):
                multi_modal_inputs[k] = torch.cat(v, dim=0).to(device=self.device)
            else:
                multi_modal_inputs[k] = v.to(device=self.device)
        label_mask = batch["label_mask"].to(device=self.device, dtype=torch.bool)

        labels = input_ids.detach().clone().masked_fill(~attention_mask, -100)
        # label_mask is encode by prompt without answer, so we need to mask the labels just save the answer tokens
        labels = labels.masked_fill(label_mask, -100)

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **multi_modal_inputs,
            )

        # train model return the loss
        return outputs.loss
