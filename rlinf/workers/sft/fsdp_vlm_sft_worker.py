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
import re
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


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

    def _normalize_text(self, s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    def _extract_boxed(self, text: str) -> str | None:
        idx = text.rfind("boxed")
        if idx < 0:
            return None
        s = text[idx + len("boxed") :].strip()
        if not s:
            return None
        if s[0] != "{":
            return s.split("$")[0].strip() or None

        depth = 0
        out = []
        for ch in s:
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth >= 1:
                out.append(ch)
        ans = "".join(out).strip()
        return ans or None

    def _extract_answer(self, text: str) -> str:
        if SupportedModel(self.cfg.actor.model.model_type) not in [
            SupportedModel.QWEN2_5_VL_SFT,
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]:
            raise ValueError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

        if not text:
            return ""

        # 1) Get the last assistant span from common chat templates.
        patterns = [
            r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
            r"<\|assistant\|>\s*(.*?)(?:<\|end\|>|$)",
        ]
        body = None
        for p in patterns:
            matches = re.findall(p, text, flags=re.DOTALL | re.IGNORECASE)
            if matches:
                body = matches[-1].strip()
                break
        if body is None:
            body = text.strip()

        # 2) Remove reasoning blocks if present.
        body = re.sub(
            r"<think>.*?</think>", "", body, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        # 3) Remove chat special tokens (e.g., <|im_end|>, <|endoftext|>)
        body = re.sub(r"<\|[^>]+?\|>", " ", body).strip()

        # 4) Try explicit "final answer" markers.
        marker_patterns = [
            r"(?:final answer is|the answer is)\s*[:：]?\s*(.+)$",
            r"(?:answer)\s*[:：]\s*(.+)$",
        ]
        for p in marker_patterns:
            m = re.search(p, body, flags=re.IGNORECASE | re.DOTALL)
            if m:
                cand = m.group(1).strip()
                cand = re.split(r"\n|<\|im_end\|>", cand)[0].strip()
                if cand:
                    body = cand
                    break

        # 5) Math-style boxed fallback.
        boxed = self._extract_boxed(body)
        if boxed:
            body = boxed

        # 6) Last non-empty line fallback.
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if lines:
            body = lines[-1]

        # final cleanup
        body = body.strip().strip("`").strip()
        body = body.rstrip(".").rstrip("/")
        return body

    def generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, torch.Tensor],
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """generate with use_cache/past_key_values, without calling model.generate()."""
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos_token_id if eos_token_id is not None else 0)
        )

        batch_size = input_ids.size(0)
        generated_ids = input_ids
        generated_attention_mask = attention_mask.to(dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                # prefill: full prompt + multimodal
                cache_position = torch.arange(
                    0,
                    generated_ids.size(1),
                    device=generated_ids.device,
                    dtype=torch.long,
                )
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids=generated_ids,
                    attention_mask=generated_attention_mask,
                    use_cache=True,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    **multi_modal_inputs,
                )
            else:
                # decode: only last token + cache
                new_generated_ids = generated_ids[:, -1:].contiguous()
                start_pos = generated_attention_mask.size(1) - new_generated_ids.size(1)
                cache_position = torch.arange(
                    start_pos,
                    generated_attention_mask.size(1),
                    device=generated_ids.device,
                    dtype=torch.long,
                )

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids=new_generated_ids,
                    attention_mask=generated_attention_mask,
                    use_cache=True,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                )

            with self.amp_context:
                outputs = self.model(**model_inputs)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            past_key_values = (
                outputs.past_key_values
                if hasattr(outputs, "past_key_values")
                else outputs[1]
            )

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # finished sample keeps appending PAD
            next_token = torch.where(
                finished,
                torch.full_like(next_token, pad_token_id),
                next_token,
            )

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

            # unfinished -> 1, finished -> 0
            append_mask = (
                (~finished).to(dtype=generated_attention_mask.dtype).unsqueeze(-1)
            )
            generated_attention_mask = torch.cat(
                [generated_attention_mask, append_mask], dim=-1
            )

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                local_all_finished = torch.tensor(
                    [int(torch.all(finished))],
                    device=generated_ids.device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    local_all_finished, op=torch.distributed.ReduceOp.MIN
                )

                if local_all_finished.item() == 1:
                    break

        return generated_ids

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, torch.Tensor],
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """Greedy decode without calling HF generate(), compatible with FSDP full_shard."""
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos_token_id if eos_token_id is not None else 0)
        )

        generated_ids = input_ids
        generated_attention_mask = attention_mask.to(dtype=torch.long)
        batch_size = generated_ids.size(0)
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=generated_ids.device
        )

        for _ in range(max_new_tokens):
            with self.amp_context:
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=generated_attention_mask,
                    **multi_modal_inputs,
                )

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            # Keep finished samples stable.
            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            append_mask = (
                (~finished).to(dtype=generated_attention_mask.dtype).unsqueeze(-1)
            )
            generated_attention_mask = torch.cat(
                [generated_attention_mask, append_mask], dim=-1
            )

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                local_all_finished = torch.tensor(
                    [int(torch.all(finished))],
                    device=generated_ids.device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    local_all_finished, op=torch.distributed.ReduceOp.MIN
                )

                if local_all_finished.item() == 1:
                    break

        return generated_ids

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
            # use kv cache to generate the text
            # the self.generate_with_kv_cache is more efficient than the self.generate
            generate_ids = self.generate_with_kv_cache(
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
