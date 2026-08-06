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

"""VLM datasets for reasoning RL and VLM SFT."""

from typing import Optional, Union

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.vlm.base import VLMBaseDataset
from rlinf.data.datasets.vlm.collate_fn import collate_fn
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry
from rlinf.data.datasets.vlm.robo2vlm import Robo2VLMDataset, Robo2VLMSFTDataset
from rlinf.data.datasets.vlm.vlm_trend_reward import (
    SimpleVLMTrendRewardSFTDataset,
    VLMTrendRewardSFTDataset,
)
from rlinf.utils.logging import get_logger

logger = get_logger()
VLM_DATA_TYPE = "vlm"


def create_vlm_datasets(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    *,
    data_paths: Optional[Union[list[str], str]] = None,
    eval_dataset: bool = False,
) -> tuple[Dataset | None, Dataset | None]:
    """Create VLM train/val datasets for RL or SFT.

    Requires ``config.data.type == "vlm"``. Concrete dataset class is selected by
    ``config.data.dataset_name`` (default ``robo2vlmsft``).

    For a single split (typical SFT), pass ``data_paths`` and optionally
    ``eval_dataset``; returned as ``(dataset, None)``.
    Otherwise train/val are built from ``config.data.train_data_paths`` /
    ``config.data.val_data_paths``.
    """
    data_type = str(config.data.type)
    if data_type != VLM_DATA_TYPE:
        raise NotImplementedError(
            f"Unsupported VLM dataset type {data_type!r}, expected {VLM_DATA_TYPE!r}"
        )

    dataset_name = getattr(config.data, "dataset_name", None) or "robo2vlmsft"
    lazy_loading = bool(getattr(config.data, "lazy_loading", False))
    logger.info(
        "Using VLM dataset: name=%s, lazy_loading=%s",
        dataset_name,
        lazy_loading,
    )

    if data_paths is not None:
        return (
            VLMDatasetRegistry.create(
                dataset_name,
                data_paths=data_paths,
                config=config,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
            ),
            None,
        )

    train_dataset = None
    if config.data.get("train_data_paths", None) is not None:
        train_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.train_data_paths,
            config=config,
            tokenizer=tokenizer,
            eval_dataset=False,
        )
    val_dataset = None
    if config.data.get("val_data_paths", None) is not None:
        val_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.val_data_paths,
            config=config,
            tokenizer=tokenizer,
            eval_dataset=True,
        )
    return train_dataset, val_dataset


__all__ = [
    "Robo2VLMDataset",
    "Robo2VLMSFTDataset",
    "SimpleVLMTrendRewardSFTDataset",
    "VLMBaseDataset",
    "VLMDatasetRegistry",
    "VLMTrendRewardSFTDataset",
    "VLM_DATA_TYPE",
    "collate_fn",
    "create_vlm_datasets",
]
