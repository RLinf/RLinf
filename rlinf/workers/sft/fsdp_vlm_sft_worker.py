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
import re
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlmSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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
                else self.cfg.data.get("eval_batch_size", 1)
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
            return data_loader, {
                "dataset_name": dataset_name,
                "num_samples": len(train_dataset),
            }

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _normalize_text(self, s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    def _extract_answer(self, text: str) -> str:
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.QWEN2_5_VL_SFT,
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]:
            m = re.search(
                r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
                text,
                flags=re.DOTALL,
            )
            return m.group(1).strip() if m else text.strip()
        else:
            raise ValueError(
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
            multi_modal_inputs[k] = v.to(device=self.device)

        with torch.no_grad():
            with self.amp_context:
                generate_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **multi_modal_inputs,
                )

        # encode the generated text
        for i in range(len(answers)):
            full_pred_text = self.tokenizer.decode(
                generate_ids[i].tolist(), skip_special_tokens=False
            )

            pred_text = self._extract_answer(full_pred_text)
            gold_text = answers[i]

            if self._normalize_text(pred_text) == self._normalize_text(gold_text):
                correct += 1

        # eval model return the correct number of answers
        return correct

    def get_train_model_output(self, batch: dict[str, Any]):
        # hundle the input batch
        input_ids = batch["prompt"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
        multi_modal_inputs = batch["multi_modal_inputs"]
        for k, v in multi_modal_inputs.items():
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
