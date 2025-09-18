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
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL.Image import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


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


class VisionLanguageDataset(Dataset):
    def __init__(
        self, data_paths: Union[List[str], str], config: DictConfig, tokenizer
    ):
        super().__init__()
        self.data_paths = data_paths
        self.use_chat_template = config.data.use_chat_template

        self.image_keys = config.data.image_keys
        self.prompt_key = config.data.prompt_key
        self.choice_key = config.data.choice_key
        self.answer_key = config.data.answer_key
        self.solution_key = config.data.solution_key

        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.max_prompt_length = config.data.max_prompt_length
        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(config.actor.model.model_path)
        self.data = self._load_data()
        self.post_process()

    def post_process(self) -> None:
        def get_image_list(
            dataitem: Dict, image_keys: Optional[List[str]]
        ) -> List[Union[bytes, str]]:
            image_list: List[Union[bytes, str]] = []
            if image_keys:
                for key in image_keys:
                    image_content = dataitem.get(key, None)
                    if image_content is None:
                        continue
                    if isinstance(image_content, Image):
                        image_content.append(image_content)
                    if isinstance(image_content, dict) and "bytes" in image_content:
                        image_content = image_content["bytes"]
                        assert isinstance(image_content, bytes), (
                            f"image content should be bytes, but got {type(image_content)} , content is {image_content}"
                        )
                    image_list.append(image_content)
            if image_list == []:
                return [None]
            return image_list

        def process_prompt(
            data_item: Dict, image_count: int
        ) -> Tuple[
            str,
            List[int],
            int,
        ]:
            question = data_item.get(self.prompt_key, "")
            options = data_item.get(self.choice_key, [])
            if not isinstance(options, list):
                options = [options]
            prompt_text = question
            if options:
                prompt_text += f"{options}\n"
            if self.use_chat_template:
                message_content: List = []
                for i in range(image_count):
                    message_content.append({"type": "image"})
                message_content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": message_content}]
                prompt_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = self.processor(
                    text=[prompt_text],
                    padding=True,
                    return_tensors="pt",
                )["input_ids"]
                if isinstance(prompt_ids, torch.Tensor):
                    if prompt_ids.dim() == 2 and prompt_ids.size(0) == 1:
                        prompt_ids = prompt_ids.squeeze(0)  # [L]
                    prompt_ids = prompt_ids.to(dtype=torch.long)
                else:
                    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
                prompt_length = len(prompt_ids)

                return prompt_text, prompt_ids, prompt_length
            else:
                raise NotImplementedError("Non-chat template not implemented yet.")

        processed_data: List[DatasetItem] = []
        for idx, item in enumerate(self.data):
            image_list: List[Union[bytes, str]] = get_image_list(item, self.image_keys)
            prompt_text, prompt_ids, prompt_length = process_prompt(
                item, len(image_list)
            )

            if prompt_length > self.max_prompt_length:
                print(
                    f"prompt_ids length {prompt_length} exceeds the max_prompt_length {self.max_prompt_length}",
                )
                prompt_ids = prompt_ids[: self.max_prompt_length]
                prompt_length = self.max_prompt_length
            prompt_ids = batch_pad_to_fixed_len(
                [prompt_ids],
                self.max_prompt_length,
                self.tokenizer.eos_token_id,
                left_pad=True,
            )[0]
            answer = item.get(self.answer_key, None)
            solution = item.get(self.solution_key, None)

            data_item = DatasetItem(
                prompt_text=prompt_text,
                prompt=prompt_ids,
                length=prompt_length,
                image_data=image_list,
                answer=str(answer),
                solution=solution,
                idx=idx,
            )
            processed_data.append(data_item)
        self.data = processed_data

    def _load_data(self) -> List:
        merged_data = []
        for path in self.data_paths:
            _, file_extension = os.path.splitext(path)
            try:
                pass
                if file_extension == ".parquet":
                    loaded_data: List = pd.read_parquet(path).to_dict(orient="records")
                    merged_data.extend(loaded_data)
                elif file_extension == ".jsonl":
                    with open(path, "r", encoding="utf-8") as file:
                        loaded_data = [json.loads(line.strip()) for line in file]
                        merged_data.extend(loaded_data)
                elif file_extension == ".json":
                    with open(path, "r", encoding="utf-8") as file:
                        content = json.load(file)
                        if isinstance(content, list):
                            merged_data.extend(content)
                        else:
                            merged_data.append(content)
                else:
                    print(f"Unsupport {file_extension}, skip: {path}")
            except Exception as e:
                raise RuntimeError(f"Load data error: {e}")
        return merged_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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
        dataset_cls = VisionLanguageDataset
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
