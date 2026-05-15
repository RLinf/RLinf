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
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Sampler

from rlinf.data.datasets import sft_collate_fn
from rlinf.data.datasets.llm_sft import LlmSftDataset
from rlinf.hybrid_engines.fsdp.utils import generate_with_kv_cache
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class LengthBalancedSampler(Sampler):
    """Distributes samples to ranks with balanced token count per rank.

    Preserves global-batch ordering: batch N always contains JSONL rows
    [N*gbs, (N+1)*gbs). Within each global batch, samples are sorted by
    token length and distributed via zigzag pattern so each rank gets a
    mix of short and long sequences, balancing total token count and
    max sequence length across ranks.

    After zigzag assignment, each rank's samples are sorted by length
    so that DataLoader's consecutive batching pairs similar-length
    sequences together, minimizing padding waste within micro-batches.
    """

    def __init__(
        self,
        token_lengths: list[int],
        global_batch_size: int,
        micro_batch_size: int,
        num_replicas: int,
        rank: int,
        drop_last: bool = True,
    ):
        self.token_lengths = token_lengths
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self._epoch = 0
        self._num_samples = self._compute_num_samples()

    def _compute_num_samples(self) -> int:
        n = len(self.token_lengths)
        gbs = self.global_batch_size
        num_complete_batches = n // gbs
        return num_complete_batches * (gbs // self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """Compatibility with DistributedSampler interface."""
        self._epoch = epoch

    def __iter__(self):
        n = len(self.token_lengths)
        gbs = self.global_batch_size
        mbs = self.micro_batch_size
        num_replicas = self.num_replicas

        for batch_start in range(0, n - n % gbs, gbs):
            batch_indices = list(range(batch_start, batch_start + gbs))

            # Sort by token length within this global batch
            batch_indices.sort(key=lambda i: self.token_lengths[i])

            # Group sorted samples into micro-batch-sized groups (pairs for mbs=2).
            # Consecutive sorted samples have similar lengths → minimal padding.
            micro_groups = [
                batch_indices[i : i + mbs]
                for i in range(0, gbs, mbs)
            ]  # e.g. 128 pairs for gbs=256, mbs=2

            # Zigzag-distribute micro_groups across ranks to balance total tokens.
            # Each rank gets gbs // (mbs * num_replicas) groups.
            rank_assignments: list[list[int]] = [[] for _ in range(num_replicas)]
            forward = True
            rank_idx = 0
            for group in micro_groups:
                rank_assignments[rank_idx].extend(group)
                if forward:
                    rank_idx += 1
                    if rank_idx >= num_replicas:
                        rank_idx = num_replicas - 1
                        forward = False
                else:
                    rank_idx -= 1
                    if rank_idx < 0:
                        rank_idx = 0
                        forward = True

            yield from rank_assignments[self.rank]

    def __len__(self) -> int:
        return self._num_samples


class FSDPLlmSftWorker(FSDPSftWorker):
    """Text-only SFT worker for HuggingFace causal LMs (e.g. Qwen3-4B).

    Mirrors :class:`FSDPVlmSftWorker` but drops all vision/processor paths:
    no ``multi_modal_inputs`` in the forward pass, no generation-based eval.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Tripped once on the first training forward to decode masked labels
        # and verify that loss is computed on response tokens only.
        self._debug_first_forward_done = False

    def init_worker(self):
        super().init_worker()

    # ------------------------------------------------------------------
    # Checkpoint data-state (mirrors the VLM worker so resume works)
    # ------------------------------------------------------------------
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

    def _log_sample_generation(self) -> None:
        """Greedy-decode one training sample and log the result (rank 0 only).

        All ranks must call this together because FSDP `full_shard` needs
        every rank to participate in parameter all-gather for each forward.
        Every rank pulls `self.data_loader.dataset[0]` (the first record on
        disk -- deterministic and identical on every rank since the jsonl is
        physically repeated), so no cross-rank broadcast is needed.
        """
        if not hasattr(self, "data_loader") or self.data_loader is None:
            return
        if not hasattr(self, "model") or self.model is None:
            return

        item = self.data_loader.dataset[0]
        input_ids_full = item.prompt
        label_mask = item.label_mask
        prompt_len = int(label_mask.sum().item())
        assert prompt_len > 0, (
            "_log_sample_generation: prompt_len is 0 -- label_mask is wrong"
        )
        prompt_ids = input_ids_full[:prompt_len].unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        max_new = int(self.cfg.runner.get("sample_gen_max_new_tokens", 8196))

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                gen_ids = generate_with_kv_cache(
                    model=self.model,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    amp_context=self.amp_context,
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    multi_modal_inputs={},
                    max_new_tokens=max_new,
                )
        finally:
            if was_training:
                self.model.train()

        if self._rank == 0:
            new_tokens = gen_ids[0, prompt_ids.size(1) :].tolist()
            decoded = self.tokenizer.decode(
                new_tokens, skip_special_tokens=False
            )
            gold = item.answer
            self.log_info(
                f"[FSDPLlmSftWorker] sample-gen @ step={self.global_step} "
                f"prompt_tokens={prompt_ids.size(1)} "
                f"new_tokens={len(new_tokens)} "
                f"max_new={max_new}",
            )
            self.log_info(
                f"[FSDPLlmSftWorker] GOLD:\n{gold}"
            )
            self.log_info(
                f"[FSDPLlmSftWorker] PRED:\n{decoded}"
            )

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

    # ------------------------------------------------------------------
    # Tokenizer / dataloader
    # ------------------------------------------------------------------
    def build_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.actor.model.model_path,
            trust_remote_code=True,
        )
        # sft_collate_fn left-pads prompts; keep tokenizer side consistent.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self._rank == 0:
            self.log_info(
                f"[FSDPLlmSftWorker] tokenizer loaded from "
                f"{self.cfg.actor.model.model_path}: vocab_size={len(tokenizer)}, "
                f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
                f"pad={tokenizer.pad_token_id}, padding_side={tokenizer.padding_side}"
            )
        return tokenizer

    def build_dataloader(self, data_paths, eval_dataset: bool = False):
        if not hasattr(self, "tokenizer"):
            self.tokenizer = self.build_tokenizer()

        dataset = LlmSftDataset(
            data_paths=data_paths,
            config=self.cfg,
            tokenizer=self.tokenizer,
            eval_dataset=eval_dataset,
        )

        batch_size = (
            self.micro_batch_size
            if not eval_dataset
            else self.cfg.actor.get("eval_batch_size", 1)
        )

        if dist.is_available() and dist.is_initialized():
            token_lengths = dataset.get_token_lengths()
            sampler = LengthBalancedSampler(
                token_lengths=token_lengths,
                global_batch_size=self.global_batch_size,
                micro_batch_size=batch_size,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                drop_last=True,
            )
        else:
            sampler = None
        num_workers = self.cfg.data.get("num_workers", 2)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=sft_collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )
        self.log_info(
            f"[FSDPLlmSftWorker] built dataloader from {data_paths} "
            f"with {len(dataset)} samples, batch_size={batch_size}"
        )

        data_config = {
            "dataset_name": "llm_sft",
            "num_samples": len(dataset),
        }
        return data_loader, data_config

    # ------------------------------------------------------------------
    # Train / eval steps
    # ------------------------------------------------------------------
    def get_train_model_output(self, batch: dict[str, Any]):
        input_ids = batch["prompt"].to(self.device)
        attention_mask = batch["attention_mask"].to(
            self.device, dtype=torch.bool
        )
        label_mask = batch["label_mask"].to(self.device, dtype=torch.bool)

        assert input_ids.shape == attention_mask.shape == label_mask.shape, (
            f"shape mismatch: input_ids={tuple(input_ids.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}, "
            f"label_mask={tuple(label_mask.shape)}"
        )

        labels = input_ids.detach().clone().masked_fill(~attention_mask, -100)
        # label_mask is True on prompt tokens -> exclude from loss.
        labels = labels.masked_fill(label_mask, -100)

        # Per-batch count of answer tokens (for confirmation + divide-by-zero guard).
        num_answer_tokens = int((labels != -100).sum().item())
        assert num_answer_tokens > 0, (
            "get_train_model_output: all labels are -100, loss would be NaN. "
            "Check label_mask and attention_mask in the dataset."
        )

        if not self._debug_first_forward_done and self._rank == 0:
            self._debug_first_forward_done = True
            preview = labels[0]
            answer_ids = preview[preview != -100].tolist()
            decoded = self.tokenizer.decode(answer_ids, skip_special_tokens=False)
            self.log_info(
                f"[FSDPLlmSftWorker] first-batch shapes: "
                f"input_ids={tuple(input_ids.shape)}, "
                f"num_answer_tokens(batch)={num_answer_tokens}, "
                f"num_answer_tokens(sample0)={len(answer_ids)}"
            )
            self.log_info(
                f"[FSDPLlmSftWorker] decoded answer tokens for sample 0 "
                f"(truncated to 300 chars): {decoded[:300]!r}"
            )

        with self.amp_context:
            # Selective loss: only backpropagate through answer token positions.
            # The model forward still computes full logits [B, S, V] (HuggingFace
            # CausalLM always does), but we select only answer positions before
            # computing CE loss. Boolean-mask indexing creates a COPY that
            # disconnects the full logits from the backward graph, allowing
            # the full tensor to be freed before backward — saving ~10 GB on
            # long sequences (the gradient is only [num_answer, V] not [B, S, V]).
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [B, S, vocab_size]
            del outputs

            # Causal LM shift: predict token[i+1] from position[i]
            shift_labels = labels[..., 1:].contiguous()
            loss_mask = shift_labels != -100

            # Select answer positions only (boolean indexing = copy, not view)
            shift_logits = logits[..., :-1, :]
            selected_logits = shift_logits[loss_mask]  # [num_answer, V]
            del logits, shift_logits  # free full logits before backward

            selected_labels = shift_labels[loss_mask]
            loss = F.cross_entropy(selected_logits, selected_labels)

        return loss

    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError(
            "FSDPLlmSftWorker does not implement eval yet; "
            "leave data.val_data_paths unset."
        )
