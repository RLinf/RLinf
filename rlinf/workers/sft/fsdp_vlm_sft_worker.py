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

import logging
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import re

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.utils import clear_memory

from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

class FSDPVlmSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = self.cfg.actor.global_batch_size // self.cfg.actor.micro_batch_size // self._world_size

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
            from rlinf.data.datasets.vlm import VLMDatasetRegistry
            from rlinf.data.datasets import sft_collate_fn

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

            batch_size = self.cfg.actor.micro_batch_size if not eval_dataset else 1
            data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.cfg.data.get("num_workers", 4),
                drop_last=True,
                collate_fn=sft_collate_fn,
            )
            logging.info(f"Build data loader from {data_paths} with {len(train_dataset)} samples")
            return data_loader, {"dataset_name": dataset_name, "num_samples": len(train_dataset)}

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

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
                position=1,
            )
            self.model.eval()
            total = 0
            correct = 0
            
            # get the next batch
            for _ in range(eval_step):
                batch = next(eval_data_iter)

                # hundle the input batch
                input_ids = batch["prompt"].to(self.device)
                answers = batch["answer"]
                attention_mask = batch["attention_mask"].to(self.device)
                multi_modal_inputs = batch["multi_modal_inputs"]
                for k, v in multi_modal_inputs.items():
                    multi_modal_inputs[k] = v.to(device = self.device)

                with torch.no_grad():
                    with self.amp_context:
                        gen_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **multi_modal_inputs,
                        )

                # encode the generated text
                for i in range(len(answers)):
                    
                    full_pred_text = self.tokenizer.decode(
                        gen_ids[i].tolist(), skip_special_tokens=False
                    )

                    def _extract_answer(text: str) -> str:
                        m = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, flags=re.DOTALL)
                        return m.group(1).strip() if m else text.strip()

                    pred_text = _extract_answer(full_pred_text)
                    gold_text = answers[i]

                    def _normalize_text(s: str) -> str:
                        return " ".join(str(s).strip().lower().split())

                    if _normalize_text(pred_text) == _normalize_text(gold_text):
                        correct += 1
                    total += 1
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def run_training(self):
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

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

                # get the next batch, if get the end of the data_iter, reset the data_iter
                try:
                    batch = next(self.data_iter)                  
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    logging.info("[INFO] data_iter exhausted, reset iterator")
                    batch = next(self.data_iter)

                # hundle the input batch
                input_ids = batch["prompt"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
                multi_modal_inputs = batch["multi_modal_inputs"]
                for k, v in multi_modal_inputs.items():
                    multi_modal_inputs[k] = v.to(device = self.device)
                label_mask = batch["label_mask"].to(device = self.device, dtype=torch.bool)

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
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                loss = loss / self.gradient_accumulation
                avg_loss += loss.item()
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            # in one step do the optimizer step
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
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

            self.lr_scheduler.step()

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            return train_metrics