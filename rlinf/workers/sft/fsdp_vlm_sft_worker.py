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
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import generate_with_kv_cache
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.utils import vlm_extract_answer, vlm_normalize_text


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
