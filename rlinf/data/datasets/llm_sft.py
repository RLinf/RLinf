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
from typing import Union

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from rlinf.data.datasets.item import SftDatasetItem


class LlmSftDataset(Dataset):
    """Text-only SFT dataset for pre-formatted (input, output) jsonl records.

    Each line of the jsonl is expected to contain at least two fields:
        - ``prompt_key`` (default ``"input"``): a string that is already the
          full prompt the model should be conditioned on. If it already
          contains a chat template (e.g. Qwen's ``<|im_start|>...`` markers),
          we do **not** re-apply the tokenizer's chat template.
        - ``answer_key`` (default ``"output"``): the assistant response we
          want the model to learn.

    The produced ``SftDatasetItem`` has ``label_mask`` set to ``True`` on
    prompt tokens and ``False`` on response/eos tokens, matching the
    semantics of :func:`rlinf.data.datasets.sft_collate_fn` and
    :func:`FSDPSftWorker.get_train_model_output` which invert via
    ``labels.masked_fill(label_mask, -100)``.

    Records that don't fit inside ``data.max_length`` are dropped at
    construction time (no truncation), so every item reaching
    ``__getitem__`` is guaranteed to fit.
    """

    def __init__(
        self,
        data_paths: Union[str, list[str]],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        eval_dataset: bool = False,
    ):
        self.cfg = config
        self.tokenizer = tokenizer

        data_cfg = config.data
        self.prompt_key: str = data_cfg.get("prompt_key", "input")
        self.answer_key: str = data_cfg.get("answer_key", "output")
        self.max_length: int = int(data_cfg.get("max_length", 8192))
        self.max_samples: int = int(data_cfg.get("max_samples", -1))
        self.append_eos: bool = bool(data_cfg.get("append_eos", True))

        assert self.max_length > 0, f"data.max_length must be > 0, got {self.max_length}"
        if self.append_eos and self.tokenizer.eos_token_id is None:
            raise ValueError(
                "append_eos=True but tokenizer.eos_token_id is None; "
                "either disable append_eos or use a tokenizer with an EOS token."
            )

        if isinstance(data_paths, str):
            paths = [data_paths]
        else:
            paths = list(data_paths)
        print(f"[LlmSftDataset] loading from paths: {paths}", flush=True)
        raw_records = self._load_jsonl(paths[0], max_samples=self.max_samples)
        print(
            f"[LlmSftDataset] loaded {len(raw_records)} raw records from {paths[0]}",
            flush=True,
        )
        assert len(raw_records) > 0, (
            f"LlmSftDataset: no records loaded from {paths}"
        )

        # Schema check on the first record (required fields + types).
        self._validate_schema(raw_records[0])

        # Validate all records first, collect valid texts.
        valid_prompts: list[str] = []
        valid_answers: list[str] = []
        malformed = 0
        for rec in raw_records:
            try:
                self._validate_schema(rec)
            except (KeyError, ValueError) as err:
                print(
                    f"[LlmSftDataset] skipping malformed record: {err}",
                    flush=True,
                )
                malformed += 1
                continue
            valid_prompts.append(rec[self.prompt_key])
            valid_answers.append(rec[self.answer_key])

        print(
            f"[LlmSftDataset] batch-tokenizing {len(valid_prompts)} records...",
            flush=True,
        )

        # Batch-tokenize all prompts and answers at once (uses Rust parallelism
        # in HuggingFace tokenizers, ~10-50x faster than per-item Python loop).
        prompt_encodings = self.tokenizer(
            valid_prompts, add_special_tokens=False, return_attention_mask=False
        )
        answer_encodings = self.tokenizer(
            valid_answers, add_special_tokens=False, return_attention_mask=False
        )

        # Build pre-tokenized cache with length filtering.
        eos_id = self.tokenizer.eos_token_id
        self._encoded_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._answers: list[str] = []
        self._prompt_texts: list[str] = []
        dropped = malformed
        total_lens: list[int] = []

        for i in range(len(valid_prompts)):
            prompt_ids = prompt_encodings["input_ids"][i]
            answer_ids = answer_encodings["input_ids"][i]
            if self.append_eos and eos_id is not None:
                answer_ids = answer_ids + [eos_id]

            n_prompt = len(prompt_ids)
            n_answer = len(answer_ids)
            total = n_prompt + n_answer

            if total > self.max_length or n_prompt == 0 or n_answer == 0:
                dropped += 1
                continue

            input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
            attention_mask = torch.ones(total, dtype=torch.long)
            label_mask = torch.zeros(total, dtype=torch.bool)
            label_mask[:n_prompt] = True

            self._encoded_cache.append((input_ids, attention_mask, label_mask))
            self._answers.append(valid_answers[i])
            self._prompt_texts.append(valid_prompts[i])
            total_lens.append(total)

        assert len(self._encoded_cache) > 0, (
            f"LlmSftDataset: after filtering, 0 samples remain "
            f"(dropped {dropped}, max_length={self.max_length}). "
            "Raise max_length or inspect the data."
        )

        print(
            f"[LlmSftDataset] loaded {len(raw_records)} raw records from {paths}; "
            f"kept {len(self._encoded_cache)} after length filter "
            f"(dropped {dropped}, max_length={self.max_length}, "
            f"append_eos={self.append_eos})",
            flush=True,
        )
        if total_lens:
            print(
                f"[LlmSftDataset] kept-sample total-token stats: "
                f"min={min(total_lens)}, max={max(total_lens)}, "
                f"mean={sum(total_lens) / len(total_lens):.1f}",
                flush=True,
            )

        # First-sample preview for sanity-check.
        first_ids, _, first_label = self._encoded_cache[0]
        n_prompt = int(first_label.sum().item())
        n_answer = int(first_label.numel() - n_prompt)
        print(
            f"[LlmSftDataset] first kept sample: total_tokens={first_ids.numel()}, "
            f"prompt_tokens={n_prompt}, answer_tokens={n_answer}, "
            f"eos_id={self.tokenizer.eos_token_id}",
            flush=True,
        )
        assert n_answer > 0, (
            "LlmSftDataset: first kept sample has zero answer tokens; "
            "the loss would be degenerate."
        )
        assert n_prompt > 0, (
            "LlmSftDataset: first kept sample has zero prompt tokens."
        )

    def get_token_lengths(self) -> list[int]:
        """Return pre-computed token lengths for all samples."""
        return [ids.numel() for ids, _, _ in self._encoded_cache]

    def _validate_schema(self, record: dict) -> None:
        for key in (self.prompt_key, self.answer_key):
            if key not in record:
                raise KeyError(
                    f"LlmSftDataset: record is missing required field '{key}'. "
                    f"Available fields: {list(record.keys())}"
                )
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(
                    f"LlmSftDataset: field '{key}' must be a non-empty string; "
                    f"got type={type(record[key]).__name__} value preview="
                    f"{str(record[key])[:60]!r}"
                )

    @staticmethod
    def _load_jsonl(path: str, max_samples: int = -1) -> list[dict]:
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if 0 < max_samples <= len(records):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise ValueError(
                        f"Failed to parse line {line_idx} of {path}: {err}"
                    ) from err
        return records

    def __len__(self) -> int:
        return len(self._encoded_cache)

    def _encode(self, prompt_text: str, answer_text: str) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        prompt_ids: list[int] = self.tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        answer_ids: list[int] = self.tokenizer.encode(
            answer_text, add_special_tokens=False
        )
        if self.append_eos and self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        # label_mask: 1 on prompt tokens (masked out in loss), 0 on answer
        # tokens (contribute to loss). The eos appended to the answer is
        # left at 0 so the model learns to stop.
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if len(prompt_ids) > 0:
            label_mask[: len(prompt_ids)] = True

        return input_ids, attention_mask, label_mask

    def __getitem__(self, idx: int) -> SftDatasetItem:
        input_ids, attention_mask, label_mask = self._encoded_cache[idx]
        return SftDatasetItem(
            prompt=input_ids,
            length=int(input_ids.numel()),
            answer=self._answers[idx],
            idx=idx,
            attention_mask=attention_mask,
            label_mask=label_mask,
            prompt_text=self._prompt_texts[idx],
        )
