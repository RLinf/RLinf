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
import os
from typing import Any

import torch
import torch.nn.functional as F
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
        # VLMRewardModel loads success/potential adapters from lora_* keys in
        # full_weights.pt. FSDP full-state export merges Peft weights and drops
        # those keys; rewrite the file with the Peft adapter state when LoRA.
        if bool(self.cfg.actor.model.get("is_lora", False)):
            self._rewrite_full_weights_with_lora_adapters(save_path)

    def _rewrite_full_weights_with_lora_adapters(self, save_path: str) -> None:
        """Replace merged full_weights.pt with Peft adapter tensors (lora_*).

        FSDP full-state export merges adapters into base weights and drops
        ``lora_*`` keys. Online ``HistoryVLMRewardModel`` requires those keys
        for ``success_lora_path`` / ``lora_path``.
        """
        from peft import get_peft_model_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Actor save_path is typically .../global_step_N/actor
        target = os.path.join(save_path, "model_state_dict", "full_weights.pt")
        torch.distributed.barrier()
        if not os.path.exists(target):
            if self._rank == 0:
                self.log_warning(
                    f"Skip LoRA rewrite: full_weights.pt not found at {target}"
                )
            torch.distributed.barrier()
            return

        model = self.model
        peft_model = None
        for module in model.modules():
            if hasattr(module, "peft_config"):
                peft_model = module
                break
        if peft_model is None:
            if self._rank == 0:
                self.log_warning(
                    f"Skip LoRA rewrite: model has no peft_config at {save_path}"
                )
            torch.distributed.barrier()
            return

        with FSDP.summon_full_params(model, writeback=False):
            lora_state = get_peft_model_state_dict(peft_model)
            lora_state = {
                key: value.detach().cpu() for key, value in lora_state.items()
            }

        torch.distributed.barrier()
        if self._rank == 0:
            if not any("lora_" in key for key in lora_state):
                raise RuntimeError(
                    f"Peft export produced no lora_* keys for checkpoint {save_path}"
                )
            torch.save(lora_state, target)
            self.log_info(
                f"Rewrote {target} with {len(lora_state)} LoRA adapter tensors"
            )
        torch.distributed.barrier()

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
            self.log_info(
                f"Build data loader from {data_paths} with {len(train_dataset)} samples"
            )
            if len(data_loader) == 0:
                raise ValueError(f"data_loader is empty; check data_path {data_paths}")

            data_config = {
                "dataset_name": dataset_name,
                "num_samples": len(train_dataset),
            }

            return data_loader, data_config

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]) -> dict[str, int] | int:
        """Generate answers for an eval batch and score them against gold.

        Args:
            batch: Collated eval batch with ``prompt``, ``answer``,
                ``attention_mask`` and ``multi_modal_inputs``.

        Returns:
            When every gold answer is a binary ``0`` / ``1`` label, a dict of
            class-aware counts (``correct``, ``total`` and per-class
            correct/total) suitable for :meth:`compute_eval_metrics`. Otherwise
            the scalar number of correct predictions in the batch.
        """
        counts = {
            "correct": 0,
            "total": 0,
            "positive_correct": 0,
            "positive_total": 0,
            "negative_correct": 0,
            "negative_total": 0,
        }
        binary_labels = True
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

            normalized_pred = vlm_normalize_text(pred_text)
            normalized_gold = vlm_normalize_text(gold_text)
            is_correct = normalized_pred == normalized_gold
            counts["total"] += 1
            counts["correct"] += int(is_correct)
            binary_labels &= normalized_gold in {"0", "1"}
            class_name = "positive" if normalized_gold == "1" else "negative"
            counts[f"{class_name}_total"] += 1
            counts[f"{class_name}_correct"] += int(is_correct)

        return counts if binary_labels else counts["correct"]

    def compute_eval_metrics(self, counts: dict[str, float]) -> dict[str, float]:
        """Compute class-aware metrics for binary VLM evaluation.

        Args:
            counts: Aggregated counts from :meth:`get_eval_model_output`
                (``correct``, ``total`` and per-class correct/total).

        Returns:
            A dict with overall accuracy, positive-class recall, negative-class
            accuracy, balanced accuracy and per-class totals.
        """
        positive_accuracy = counts["positive_correct"] / max(
            1, counts["positive_total"]
        )
        negative_accuracy = counts["negative_correct"] / max(
            1, counts["negative_total"]
        )
        return {
            "eval_accuracy": counts["correct"] / max(1, counts["total"]),
            "positive_recall": positive_accuracy,
            "negative_accuracy": negative_accuracy,
            "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2,
            "positive_total": counts["positive_total"],
            "negative_total": counts["negative_total"],
        }

    def weighted_answer_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        answers: list[str],
        success_weight: float,
        non_success_weight: float,
    ) -> torch.Tensor:
        """Compute answer-token CE with normalized per-sample class weights.

        Args:
            logits: Model logits of shape ``(batch, seq, vocab)``.
            labels: Target token ids of shape ``(batch, seq)`` with ``-100`` at
                masked positions.
            answers: Per-sample gold answer strings; ``"1"`` selects the success
                weight, anything else the non-success weight.
            success_weight: Per-sample weight for success (``"1"``) answers.
            non_success_weight: Per-sample weight for non-success answers.

        Returns:
            The scalar weighted mean cross-entropy over answer tokens.
        """
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        valid_tokens = shift_labels.ne(-100)
        token_counts = valid_tokens.sum(dim=1).clamp_min(1)
        sample_losses = (token_losses * valid_tokens).sum(dim=1) / token_counts
        weights = torch.tensor(
            [
                success_weight if str(answer).strip() == "1" else non_success_weight
                for answer in answers
            ],
            device=sample_losses.device,
            dtype=sample_losses.dtype,
        )
        return (sample_losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def get_train_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Move the input batch to the compute device.
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

        reweight_cfg = self.cfg.actor.model.get("sample_reweight", {})
        if reweight_cfg.get("enabled", False):
            loss = self.weighted_answer_loss(
                outputs.logits,
                labels,
                batch["answer"],
                float(reweight_cfg.get("success_weight", 1.0)),
                float(reweight_cfg.get("non_success_weight", 1.0)),
            )
        else:
            loss = outputs.loss
        return loss, {"loss": loss.detach().item()}
