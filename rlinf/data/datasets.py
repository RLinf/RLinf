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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL.Image import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer


def batch_pad_to_fixed_len(
    batch: List[torch.Tensor],
    max_batch_len: int,
    pad_token: int,
    left_pad: bool = False,
) -> torch.Tensor:
    if left_pad:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),  # pad on the left
                        seq,
                    ]
                )
                for seq in batch
            ]
        )
    else:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),
                    ]
                )
                for seq in batch
            ]
        )
    return batch_pad


@dataclass
class DatasetItem:
    prompt: torch.Tensor
    length: int
    answer: str
    idx: int
    solution: Optional[str] = None
    image_data: Optional[List[Union[bytes, str]]] = None
    prompt_text: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class MathDataset(Dataset):
    def __init__(self, data_paths, config, tokenizer):
        super().__init__()
        self.data_paths = data_paths
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.max_prompt_length = config.data.max_prompt_length
        self.tokenizer = tokenizer
        self.prompt_key = config.data.prompt_key

        self.data = self._load_data()
        if config.data.get("filter_prompt_by_length", False):
            total = len(self.data)
            filtered = []
            failed = 0

            for item in self.data:
                try:
                    _, L = self.encode(item[self.prompt_key])
                    if L <= self.max_prompt_length:
                        filtered.append(item)
                except Exception:
                    failed += 1

            self.data = filtered
            assert len(self.data) > 0, (
                f"No samples found within max_prompt_length={self.max_prompt_length}. "
                "Please check your dataset or increase max_prompt_length."
            )

            if failed > 0:
                logging.warning(
                    f"{failed} samples were skipped due to format issues "
                    f"(kept {len(self.data)} / {total})."
                )

    def _load_data(self):
        merged_data = []

        for path in self.data_paths:
            _, file_extension = os.path.splitext(path)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    if file_extension == ".jsonl":
                        merged_data.extend([json.loads(line.strip()) for line in file])
                    elif file_extension == ".json":
                        content = json.load(file)
                        if isinstance(content, list):
                            merged_data.extend(content)
                        else:
                            merged_data.append(content)
                    else:
                        print(f"Unsupport {file_extension}, skip: {path}")
            except Exception:
                raise RuntimeError("Load data error")

        return merged_data

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        text_ids = self.tokenizer.encode(text)
        return text_ids, len(text_ids)

    def __getitem__(self, idx):
        """
        Return a single prompt.
        """

        prompt = self.data[idx][self.prompt_key]

        answer = self.data[idx]["solutions"]

        prompt_tokens, prompt_length = self.encode(prompt)
        prompt_tokens_tensor = torch.as_tensor(prompt_tokens, dtype=torch.int64)

        if prompt_length > self.max_prompt_length:
            print(
                f"prompt_tokens_tensor length {prompt_length} exceeds the max_prompt_length {self.max_prompt_length}",
            )
            prompt_tokens_tensor = prompt_tokens_tensor[: self.max_prompt_length]
            prompt_length = self.max_prompt_length

        prompt_tokens_tensor = batch_pad_to_fixed_len(
            [prompt_tokens_tensor],
            self.max_prompt_length,
            self.tokenizer.eos_token_id,
            left_pad=True,
        )[0]
        output = DatasetItem(
            prompt=prompt_tokens_tensor,
            length=prompt_length,
            answer=answer,
            idx=idx,
            image_data=[],
        )
        return output


class VLMBaseDataset(Dataset):
    def __init__(
        self,
        data_paths: Union[List[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
        *,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = config
        raw_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        # Expand directories into file lists recursively (json/jsonl/parquet)
        self.data_paths = self._expand_data_paths(raw_paths)
        self.tokenizer = tokenizer
        # Delay processor creation; only needed when use_chat_template is True
        self._processor = None

        self.use_chat_template = bool(config.data.use_chat_template)
        self.image_keys = list(config.data.image_keys or [])
        self.prompt_key = config.data.prompt_key
        self.choice_key = config.data.get("choice_key", None)
        self.answer_key = config.data.get("answer_key", None)
        self.solution_key = config.data.get("solution_key", None)
        self.max_prompt_length = int(config.data.max_prompt_length)
        self.eos_id = int(self.tokenizer.eos_token_id)

        # Loading mode
        self.lazy_loading = bool(getattr(config.data, "lazy_loading", lazy_loading))

        self._records = []
        self._indices = []  # (path, fmt, row_index_or_offset)

        if self.lazy_loading:
            self._build_lazy_indices()
        else:
            self._eager_load_all()

    def __len__(self) -> int:
        return len(self._indices) if self.lazy_loading else len(self._records)

    def __getitem__(self, idx: int) -> DatasetItem:
        if self.lazy_loading:
            path, fmt, key = self._indices[idx]
            raw = self._load_single_lazy(path, fmt, key)
            return self._process_raw_record(raw, idx)
        else:
            raw = self._records[idx]
            return self._process_raw_record(raw, idx)

    # Ensure dataset is picklable for multi-process DataLoader by removing
    # unpicklable cache objects like pyarrow.ParquetFile from state.
    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop heavy/unpicklable caches; they will be rebuilt on-demand in workers
        for k in ("_parquet_cache", "_parquet_df_cache"):
            if k in state:
                state[k] = {}
        return state

    def __setstate__(self, state):
        # Restore state and ensure cache dicts exist
        self.__dict__.update(state)
        self._parquet_cache = getattr(self, "_parquet_cache", {})
        self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})

    def get_image_list(self, dataitem: Dict[str, Any]) -> List[Union[bytes, str, None]]:
        images: List[Union[bytes, str, None]] = []
        for k in self.image_keys:
            v = dataitem.get(k, None)
            if v is None:
                continue
            if isinstance(v, Image):
                images.append(v)
            elif isinstance(v, dict) and "bytes" in v:
                images.append(v["bytes"])
            else:
                images.append(v)  # path or url
        if not images:
            images = [None]
        return images

    def build_prompt_text(self, data_item: Dict[str, Any]) -> str:
        # Default: prompt + optional choices rendered inline
        q = data_item.get(self.prompt_key, "")
        choices = data_item.get(self.choice_key, []) if self.choice_key else []
        if not isinstance(choices, list):
            choices = [choices]
        if choices:
            return f"{q}{choices}\n"
        return str(q)

    def encode_prompt(
        self, prompt_text: str, image_count: int
    ) -> Tuple[torch.Tensor, int, Optional[str]]:
        """
        Return (token_ids[L], length, prompt_text_used). If using chat template, encode with processor.
        Subclasses may override to support alternative prompting.
        """
        if self.use_chat_template:
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    self.cfg.actor.model.model_path
                )
            content: List[Dict[str, Any]] = []
            for _ in range(max(0, image_count)):
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            rendered = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = self._processor(text=[rendered], padding=True, return_tensors="pt")[
                "input_ids"
            ]
            if isinstance(ids, torch.Tensor):
                if ids.dim() == 2 and ids.size(0) == 1:
                    ids = ids.squeeze(0)
                ids = ids.to(dtype=torch.long)
            else:
                ids = torch.tensor(ids, dtype=torch.long)
            return ids, int(ids.numel()), rendered
        else:
            # fallback: tokenizer only
            ids_list = self.tokenizer.encode(prompt_text)
            ids = torch.as_tensor(ids_list, dtype=torch.long)
            return ids, int(ids.numel()), prompt_text

    def postprocess_dataset_item(
        self, item: DatasetItem, raw: Dict[str, Any]
    ) -> DatasetItem:
        return item

    def _expand_data_paths(self, inputs: List[str]) -> List[str]:
        exts = {".jsonl", ".json", ".parquet"}
        files: List[str] = []
        for p in inputs:
            if os.path.isdir(p):
                for root, _, fnames in os.walk(p):
                    for fn in fnames:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext in exts:
                            files.append(os.path.join(root, fn))
            else:
                files.append(p)
        files = sorted(set(files))
        return files

    def _eager_load_all(self) -> None:
        merged: List[Dict[str, Any]] = []
        for path in self.data_paths:
            fmt = os.path.splitext(path)[1].lower()
            if fmt == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    merged.extend(json.loads(l) for l in f)
            elif fmt == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        merged.extend(content)
                    else:
                        merged.append(content)
            elif fmt == ".parquet":
                try:
                    merged.extend(pd.read_parquet(path).to_dict(orient="records"))
                except Exception as e:
                    raise RuntimeError(f"Failed to load parquet eagerly: {path}: {e}")
            else:
                logging.warning(f"Unsupported format {fmt} for path {path}, skipping.")
        self._records = merged
        # Build indices for consistency
        self._indices = [("", "eager", i) for i in range(len(self._records))]

    def _build_lazy_indices(self) -> None:
        self._indices.clear()
        for path in self.data_paths:
            fmt = os.path.splitext(path)[1].lower()
            if fmt == ".jsonl":
                # index by byte offsets for each line
                offsets: List[int] = []
                with open(path, "rb") as fb:
                    pos = 0
                    for line in fb:
                        offsets.append(pos)
                        pos += len(line)
                self._indices.extend((path, "jsonl", off) for off in offsets)
            elif fmt == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    if not isinstance(content, list):
                        content = [content]
                    # store the content to avoid re-reading
                    # keep perfile cache
                    self._json_cache = getattr(self, "_json_cache", {})
                    self._json_cache[path] = content
                    self._indices.extend((path, "json", i) for i in range(len(content)))
                except Exception as e:
                    raise RuntimeError(f"Failed to index json lazily: {path}: {e}")
            elif fmt == ".parquet":
                try:
                    import pyarrow.parquet as pq  # type: ignore

                    pf = pq.ParquetFile(path)
                    num_rows = pf.metadata.num_rows
                    # file handle cache
                    self._parquet_cache = getattr(self, "_parquet_cache", {})
                    self._parquet_cache[path] = pf
                    self._indices.extend((path, "parquet", i) for i in range(num_rows))
                except Exception:
                    df = pd.read_parquet(path)
                    self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})
                    self._parquet_df_cache[path] = df
                    self._indices.extend(
                        (path, "parquet_pd", i) for i in range(len(df))
                    )
            else:
                logging.warning(f"Unsupported format {fmt} for path {path}, skipping.")

    def _load_single_lazy(self, path: str, fmt: str, key: Any) -> Dict[str, Any]:
        if fmt == "eager":
            return self._records[int(key)]
        if fmt == "jsonl":
            with open(path, "rb") as fb:
                fb.seek(int(key))
                line = fb.readline()
            return json.loads(line.decode("utf-8").strip())
        if fmt == "json":
            return self._json_cache[path][int(key)]  # type: ignore[attr-defined]
        if fmt == "parquet":
            # Try to use pyarrow lazily; rebuild cache if missing
            self._parquet_cache = getattr(self, "_parquet_cache", {})
            pf = self._parquet_cache.get(path)
            if pf is None:
                try:
                    import pyarrow.parquet as pq  # type: ignore

                    pf = pq.ParquetFile(path)
                    self._parquet_cache[path] = pf
                except Exception:
                    # Fall back to pandas-based cache
                    self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})
                    df = self._parquet_df_cache.get(path)
                    if df is None:
                        df = pd.read_parquet(path)
                        self._parquet_df_cache[path] = df
                    return df.iloc[int(key)].to_dict()
            table = pf.read_row_group(key // max(1, pf.metadata.num_rows), columns=None)
            try:
                df = table.to_pandas()
                return df.iloc[int(key) % len(df)].to_dict()
            except Exception:
                df_all = pf.read().to_pandas()
                return df_all.iloc[int(key)].to_dict()
        if fmt == "parquet_pd":
            self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})
            df = self._parquet_df_cache.get(path)
            if df is None:
                df = pd.read_parquet(path)
                self._parquet_df_cache[path] = df
            return df.iloc[int(key)].to_dict()
        raise RuntimeError(f"Unknown lazy fmt {fmt}")

    def _process_raw_record(self, raw: Dict[str, Any], idx: int) -> DatasetItem:
        images = self.get_image_list(raw)
        prompt_text = self.build_prompt_text(raw)
        prompt_ids, plen, rendered_text = self.encode_prompt(prompt_text, len(images))

        if plen > self.max_prompt_length:
            prompt_ids = prompt_ids[: self.max_prompt_length]
            plen = self.max_prompt_length
        prompt_ids = batch_pad_to_fixed_len(
            [prompt_ids], self.max_prompt_length, self.eos_id, left_pad=True
        )[0]

        answer_val = raw.get(self.answer_key, None) if self.answer_key else None
        solution_val = raw.get(self.solution_key, None) if self.solution_key else None
        item = DatasetItem(
            prompt=prompt_ids,
            length=plen,
            answer=str(answer_val) if answer_val is not None else None,
            idx=idx,
            image_data=images,
            prompt_text=rendered_text or prompt_text,
            solution=solution_val,
            meta=None,
        )
        return self.postprocess_dataset_item(item, raw)


class VLMDatasetRegistry:
    registry: Dict[str, Callable[..., VLMBaseDataset]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[..., VLMBaseDataset]], Callable[..., VLMBaseDataset]]:
        def decorator(klass: Callable[..., VLMBaseDataset]):
            cls.registry[name] = klass
            return klass

        return decorator

    @classmethod
    def create(
        cls,
        dataset_name: Optional[str],
        *,
        data_paths: Union[List[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
    ) -> VLMBaseDataset:
        key = dataset_name.lower()
        dataset_class = cls.registry.get(key)
        return dataset_class(data_paths=data_paths, config=config, tokenizer=tokenizer)


@VLMDatasetRegistry.register("robo2vlm")
class Robo2VLMDataset(VLMBaseDataset):
    def get_image_list(self, dataitem: Dict[str, Any]) -> List[Union[bytes, str, None]]:
        # Prefer common robo2vlm fields if present, else fallback to configured keys
        images: List[Any] = []
        if "images" in dataitem:
            v = dataitem.get("images")
            if isinstance(v, list):
                images = list(v)
            elif v is not None:
                images = [v]
            else:
                images = [None]
        elif "image" in dataitem:
            v = dataitem.get("image")
            if v is not None:
                images = [v]
            else:
                images = [None]
        else:
            # fallback to base behavior using configured image_keys
            return super().get_image_list(dataitem)

        # Normalize each element similar to base behavior
        normed: List[Union[bytes, str, None]] = []
        for v in images:
            if v is None:
                continue
            if isinstance(v, Image):
                normed.append(v)
            elif isinstance(v, dict) and "bytes" in v:
                normed.append(v["bytes"])  # raw bytes
            else:
                normed.append(v)  # path/uri/string
        if not normed:
            normed = [None]
        return normed

    def build_prompt_text(self, data_item: Dict[str, Any]) -> str:
        # Use 'question' and 'choices' if present; else fallback to base using configured prompt/choice keys
        question = data_item.get("question", None)
        choices = data_item.get("choices", None)
        if question is None:
            return super().build_prompt_text(data_item)
        # normalize choices
        if isinstance(choices, str):
            try:
                import ast

                choices = ast.literal_eval(choices)
            except Exception:
                choices = [choices]
        if not isinstance(choices, list):
            choices = [choices] if choices is not None else []

        text = f"{question}\n"
        if choices:
            text += "Choices:\n"
            for i, c in enumerate(choices):
                text += f"{chr(65 + i)}. {c}\n"
        return text

    def postprocess_dataset_item(
        self, item: DatasetItem, raw: Dict[str, Any]
    ) -> DatasetItem:
        # Derive answer from 'correct_answer' and 'choices' if not provided
        if not item.answer or str(item.answer).lower() in {"none", "", "null"}:
            choices = raw.get("choices")
            ca = raw.get("correct_answer")
            try:
                # Normalize choices
                if isinstance(choices, str):
                    import ast

                    choices = ast.literal_eval(choices)
                if not isinstance(choices, list):
                    choices = [choices] if choices is not None else []

                ans_val: Optional[str] = None
                if isinstance(ca, int) and 0 <= ca < len(choices):
                    ans_val = str(choices[ca])
                elif isinstance(ca, str):
                    cstr = ca.strip()
                    # Letter index like 'A', 'B', ...
                    if len(cstr) == 1 and "A" <= cstr <= "Z":
                        idx = ord(cstr) - ord("A")
                        if 0 <= idx < len(choices):
                            ans_val = str(choices[idx])
                    # Direct match to a choice value
                    if ans_val is None and choices:
                        for ch in choices:
                            if str(ch) == cstr:
                                ans_val = cstr
                                break
                if ans_val is not None:
                    item.answer = ans_val
            except Exception:
                # Keep original if any
                pass
        return item


def create_rl_dataset(config: DictConfig, tokenizer):
    """Create rl datasets.

    Arguments:
        config: The RLinf config.
        tokenizer (Tokenizer): The tokenizer.

    Returns:
        train_dataset (Dataset): The training dataset.

        val_dataset (Dataset): The validation dataset.
    """

    if config.data.type == "math":
        dataset_cls = MathDataset
    elif config.data.type == "vision_language":
        # Prefer new factory-based VLM datasets; fallback to legacy if requested
        dataset_name = getattr(config.data, "dataset_name", None)
        lazy_loading = bool(getattr(config.data, "lazy_loading", False))

        print(f"Using VLM dataset: name={dataset_name}, lazy_loading={lazy_loading}")

        train_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.train_data_paths,
            config=config,
            tokenizer=tokenizer,
        )
        val_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.val_data_paths,
            config=config,
            tokenizer=tokenizer,
        )
        return train_dataset, val_dataset
    else:
        return None, None

    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    train_dataset = dataset_cls(
        data_paths=config.data.train_data_paths,
        config=config,
        tokenizer=tokenizer,
    )

    val_dataset = dataset_cls(
        data_paths=config.data.val_data_paths,
        config=config,
        tokenizer=tokenizer,
    )

    return train_dataset, val_dataset


def collate_fn(data_list: List["DatasetItem"]) -> Dict[str, Any]:
    prompts = []
    lens = []
    for it in data_list:
        p = (
            it.prompt
            if isinstance(it.prompt, torch.Tensor)
            else torch.as_tensor(it.prompt, dtype=torch.long)
        )
        if p.dim() == 2 and p.size(0) == 1:
            p = p.squeeze(0)
        assert p.dim() == 1, (
            f"DatasetItem.prompt must be 1-D tensor, current shape is: {p.shape}"
        )
        prompts.append(p)
        lens.append(p.numel())

    if len(set(lens)) == 1:
        target_len = lens[0]
    else:
        target_len = min(lens)
        prompts = [p[-target_len:] if p.numel() > target_len else p for p in prompts]

    batch_prompt = torch.stack(prompts, dim=0)  # [B, L]
    batch_length = torch.tensor(
        [min(int(it.length), target_len) for it in data_list], dtype=torch.long
    )

    batch_idx = torch.tensor([int(it.idx) for it in data_list], dtype=torch.long)

    batch: Dict[str, Any] = {
        "prompt": batch_prompt,  # [B, L]
        "length": batch_length,  # [B]
        "answer": [it.answer for it in data_list],  # List[str]
        "idx": batch_idx,  # [B]
        "solution": [it.solution for it in data_list],  # List[Optional[str]]
        "image_data": [
            it.image_data for it in data_list
        ],  # List[Optional[List[bytes|str]]]
        "prompt_text": [it.prompt_text for it in data_list],  # List[Optional[str]]
        "meta": [it.meta for it in data_list],  # List[Optional[dict]]
    }
    return batch
