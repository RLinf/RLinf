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

"""Text datasets for reasoning / LLM RL."""

import logging

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.reasoning.collate_fn import collate_fn
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.datasets.reasoning.rstar2 import Rstar2Dataset
from rlinf.data.datasets.reasoning.wideseek_r1 import WideSeekR1Dataset

TEXT_DATASET_TYPE_MAP = {
    "reasoning": ReasoningDataset,
    "math": ReasoningDataset,
    "wideseek_r1": WideSeekR1Dataset,
    "rstar2": Rstar2Dataset,
}


def create_reasoning_datasets(
    config: DictConfig, tokenizer: AutoTokenizer
) -> tuple[Dataset | None, Dataset | None]:
    """Create train/val text reasoning datasets via ``config.data.type``.

    Supported ``config.data.type`` values: ``reasoning``, ``math``,
    ``wideseek_r1``, ``rstar2``.

    For VLM datasets (``vlm``), use
    ``rlinf.data.datasets.vlm.create_vlm_datasets`` instead.
    """
    if config.data.type not in TEXT_DATASET_TYPE_MAP:
        raise NotImplementedError(
            "Unsupported dataset type "
            f"{config.data.type}, only support "
            f"{sorted(TEXT_DATASET_TYPE_MAP.keys())}. "
            "For VLM use rlinf.data.datasets.vlm.create_vlm_datasets."
        )

    dataset_cls = TEXT_DATASET_TYPE_MAP[config.data.type]
    logging.info(f"Using dataset class: {dataset_cls.__name__}")

    train_dataset, val_dataset = None, None
    if config.runner.task_type != "reasoning_eval":
        train_dataset = dataset_cls(
            data_paths=config.data.train_data_paths,
            config=config,
            tokenizer=tokenizer,
        )

    if config.data.get("val_data_paths", None) is not None:
        val_dataset = dataset_cls(
            data_paths=config.data.val_data_paths,
            config=config,
            tokenizer=tokenizer,
        )
    return train_dataset, val_dataset


__all__ = [
    "ReasoningDataset",
    "Rstar2Dataset",
    "TEXT_DATASET_TYPE_MAP",
    "WideSeekR1Dataset",
    "collate_fn",
    "create_reasoning_datasets",
]
