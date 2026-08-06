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
import os
import re
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from rlinf.data.datasets.common.item import DatasetItem
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry
from rlinf.utils.logging import get_logger
from rlinf.utils.torch_functionals import batch_pad_to_fixed_len

logger = get_logger()


@VLMDatasetRegistry.register("base")
class VLMBaseDataset(Dataset):
    def __init__(
        self,
        data_paths: Union[list[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__()
        self.cfg = config
        raw_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        # Expand directories into file lists recursively (json/jsonl/parquet)
        self.data_paths = self._expand_data_paths(raw_paths)
        self.tokenizer = tokenizer
        # Delay processor creation; only needed when use_chat_template is True
        self._processor = None

        self.system_prompt = config.data.get("system_prompt", None)
        self.use_chat_template = bool(config.data.use_chat_template)
        self.image_keys = list(config.data.image_keys or [])
        self.prompt_key = config.data.prompt_key
        self.choice_key = config.data.get("choice_key", None)
        self.answer_key = config.data.get("answer_key", None)
        self.solution_key = config.data.get("solution_key", None)
        self.max_prompt_length = int(config.data.max_prompt_length)
        self.eos_id = int(self.tokenizer.eos_token_id)

        # Loading mode
        self.lazy_loading = bool(getattr(config.data, "lazy_loading", False))

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
        for k in ("_parquet_cache", "_parquet_df_cache", "_parquet_rg_bounds"):
            if k in state:
                state[k] = {}
        return state

    def __setstate__(self, state):
        # Restore state and ensure cache dicts exist
        self.__dict__.update(state)
        self._parquet_cache = getattr(self, "_parquet_cache", {})
        self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})
        self._parquet_rg_bounds = getattr(self, "_parquet_rg_bounds", {})

    def get_image_list(self, dataitem: dict[str, Any]) -> list[Union[bytes, str, None]]:
        images: list[Union[bytes, str, None]] = []
        for k in self.image_keys:
            v = dataitem.get(k, None)
            if v is None:
                continue
            if isinstance(v, Image.Image):
                images.append(v)
            elif isinstance(v, dict) and "bytes" in v:
                images.append(v["bytes"])
            elif isinstance(v, np.ndarray) and "bytes" in v[0]:
                for p in v:
                    images.append(p["bytes"])
            else:
                images.append(v)  # path or url
        if not images:
            images = [None]
        return images

    def build_prompt_text(self, data_item: dict[str, Any]) -> str:
        # Default: prompt + optional choices rendered inline
        q = data_item.get(self.prompt_key, "")
        choices = data_item.get(self.choice_key, []) if self.choice_key else []
        if not isinstance(choices, list):
            choices = [choices]
        if choices:
            return f"{q}{choices}\n"
        return str(q)

    @staticmethod
    def process_inputs(
        processor: AutoProcessor,
        system_prompt: Optional[str],
        use_chat_template: bool,
        prompt_texts: list[str],
        images,
    ) -> tuple[str, dict[str, Any]]:
        """Shared helper to build chat-template text and processor inputs for VLM.

        This mirrors the logic in VLMBaseDataset.encode_prompt so it can be reused
        by both datasets and downstream modules (e.g., reward models).
        """
        prompt_text = prompt_texts[0]

        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        content: list[dict[str, Any]] = []

        # Parse prompt_text for image placeholders and interleave text segments with images
        parts = re.split(r"<image>", prompt_text)
        if len(parts) == 1:
            # No placeholder found, fall back to original logic
            for _ in range(max(0, len(images))):
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})
        else:
            for i, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(parts) - 1:
                    content.append({"type": "image"})

        messages.append({"role": "user", "content": content})

        if use_chat_template:
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = prompt_text

        images_inputs = []
        for image in images:
            image_obj = None
            if isinstance(image, Image.Image):
                image_obj = image.convert("RGB")
            if isinstance(image, (bytes, bytearray)):
                image_obj = Image.open(BytesIO(image)).convert("RGB")
            if image_obj is None:
                raise ValueError(
                    f"Unsupported image type: {type(image).__name__}. "
                    f"Expected PIL.Image.Image, bytes, or bytearray."
                )

            images_inputs.append(image_obj)

        inputs = processor(
            text=[rendered], images=images_inputs, padding=True, return_tensors="pt"
        )
        return rendered, inputs

    def encode_prompt(
        self, prompt_text: str, images
    ) -> tuple[torch.Tensor, int, Optional[str]]:
        """
        Return (token_ids[L], length, prompt_text_used). If using chat template, encode with processor.
        Subclasses may override to support alternative prompting.
        """
        if self.use_chat_template:
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    self.cfg.actor.model.model_path
                )

            rendered, inputs = self.process_inputs(
                processor=self._processor,
                system_prompt=self.system_prompt,
                use_chat_template=self.use_chat_template,
                prompt_texts=[prompt_text],
                images=images,
            )

            inputs.pop("attention_mask", None)

            if self.cfg.rollout.rollout_backend == "sglang":
                ids = inputs.pop("input_ids")
            elif self.cfg.rollout.rollout_backend == "vllm":
                inputs.pop("input_ids", None)
                ids = self._processor(
                    text=[rendered], images=None, padding=True, return_tensors="pt"
                )["input_ids"]
            else:
                raise ValueError(
                    f"Unsupported rollout backend {self.cfg.rollout.rollout_backend}"
                )
            if isinstance(ids, torch.Tensor):
                if ids.dim() == 2 and ids.size(0) == 1:
                    ids = ids.squeeze(0)
                ids = ids.to(dtype=torch.long)
            else:
                ids = torch.tensor(ids, dtype=torch.long)

            multi_modal_inputs = {}
            for k, v in inputs.items():
                multi_modal_inputs[k] = v
            return ids, int(ids.numel()), rendered, multi_modal_inputs
        else:
            # fallback: tokenizer only
            ids_list = self.tokenizer.encode(prompt_text)
            ids = torch.as_tensor(ids_list, dtype=torch.long)
            return ids, int(ids.numel()), prompt_text, {}

    def postprocess_dataset_item(
        self, item: DatasetItem, raw: dict[str, Any]
    ) -> DatasetItem:
        return item

    def _expand_data_paths(self, inputs: list[str]) -> list[str]:
        exts = {".jsonl", ".json", ".parquet"}
        files: list[str] = []
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
        merged: list[dict[str, Any]] = []
        for path in tqdm(self.data_paths, desc="Loading dataset files", unit="file"):
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
                logger.warning(f"Unsupported format {fmt} for path {path}, skipping.")
        self._records = merged
        # Build indices for consistency
        self._indices = [("", "eager", i) for i in range(len(self._records))]

    def _build_parquet_rg_bounds(self, pf, path: str) -> list[tuple[int, int, int]]:
        """Build and cache row-group boundaries for a parquet file."""
        self._parquet_rg_bounds = getattr(self, "_parquet_rg_bounds", {})
        bounds = []
        offset = 0
        for g in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(g).num_rows
            bounds.append((offset, offset + rg_rows, g))
            offset += rg_rows
        self._parquet_rg_bounds[path] = bounds
        return bounds

    def _build_lazy_indices(self) -> None:
        self._indices.clear()
        for path in tqdm(self.data_paths, desc="Loading dataset files", unit="file"):
            fmt = os.path.splitext(path)[1].lower()
            if fmt == ".jsonl":
                # index by byte offsets for each line
                offsets: list[int] = []
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
                    # Cache row group boundaries for correct lazy indexing
                    self._build_parquet_rg_bounds(pf, path)
                    self._indices.extend((path, "parquet", i) for i in range(num_rows))
                except Exception:
                    df = pd.read_parquet(path)
                    self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})
                    self._parquet_df_cache[path] = df
                    self._indices.extend(
                        (path, "parquet_pd", i) for i in range(len(df))
                    )
            else:
                logger.warning(f"Unsupported format {fmt} for path {path}, skipping.")

    def _load_single_lazy(self, path: str, fmt: str, key: Any) -> dict[str, Any]:
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
            # Find the row group that contains the requested row
            self._parquet_rg_bounds = getattr(self, "_parquet_rg_bounds", {})
            bounds = self._parquet_rg_bounds.get(path)
            if bounds is None:
                # Rebuild bounds if cache was cleared (e.g. after unpickling)
                bounds = self._build_parquet_rg_bounds(pf, path)
            row = int(key)
            target_rg = 0
            local_idx = row
            for start, end, rg_idx in bounds:
                if start <= row < end:
                    target_rg = rg_idx
                    local_idx = row - start
                    break
            table = pf.read_row_group(target_rg, columns=None)
            df = table.to_pandas()
            return df.iloc[local_idx].to_dict()
        if fmt == "parquet_pd":
            self._parquet_df_cache = getattr(self, "_parquet_df_cache", {})
            df = self._parquet_df_cache.get(path)
            if df is None:
                df = pd.read_parquet(path)
                self._parquet_df_cache[path] = df
            return df.iloc[int(key)].to_dict()
        raise RuntimeError(f"Unknown lazy fmt {fmt}")

    def _process_raw_record(self, raw: dict[str, Any], idx: int) -> DatasetItem:
        images = self.get_image_list(raw)
        prompt_text = self.build_prompt_text(raw)
        prompt_ids, plen, rendered_text, multi_modal_inputs = self.encode_prompt(
            prompt_text, images
        )

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
            multi_modal_inputs=multi_modal_inputs,
        )
        return self.postprocess_dataset_item(item, raw)
