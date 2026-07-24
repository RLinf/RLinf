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

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlinf.envs.gen_reward.datasets import TextDataset
from rlinf.envs.gen_reward.utils import cfg_get


class OCRDataset(TextDataset):
    """OCR prompt records for generation reward tasks."""

    @classmethod
    def from_config(cls, cfg: Any) -> "OCRDataset":
        dataset_path = Path(str(cfg_get(cfg, "path"))).expanduser()
        if dataset_path.is_dir():
            split = str(cfg_get(cfg, "split", "train"))
            dataset_path = dataset_path / f"{split}.txt"
        return cls(load_txt_prompts(dataset_path))


def load_txt_prompts(dataset_path: Path) -> list[dict[str, Any]]:
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                records.append({"task_description": prompt})
    if not records:
        raise ValueError(f"Prompt dataset is empty: {dataset_path}")
    return records


DATASET_CLS = OCRDataset


__all__ = ["DATASET_CLS", "OCRDataset"]
