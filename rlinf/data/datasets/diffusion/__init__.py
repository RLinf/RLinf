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

from typing import Any

import numpy as np
import torch

EnvRecord = dict[str, Any]
EnvRecords = list[EnvRecord]
EnvObs = dict[str, Any]
EnvBatch = tuple[EnvObs, EnvRecords]


class TextDataset:
    """Base interface for text-conditioned generation reward datasets.

    Dataset records use a small canonical schema:

    Required:
        task_description: str

    `records_to_env_obs()` maps records to batched env observations with
    plural keys, for example `task_descriptions`.
    """

    records: EnvRecords

    def __init__(self, records: EnvRecords):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EnvRecord:
        """Return one canonical dataset record."""
        return self.records[index]

    def build_grouped_env_batch(
        self,
        group_indices: Any,
        group_size: int,
        num_envs: int,
    ) -> EnvBatch:
        """
        Build one env batch from sampled group indices.

        Each sampled record is repeated `group_size` times so GRPO samples in
        the same group share the same condition.
        """
        records: EnvRecords = []
        group_size = int(group_size)
        num_envs = int(num_envs)
        for index in group_indices:
            record = self[int(index)]
            records.extend(dict(record) for _ in range(group_size))
            if len(records) >= num_envs:
                break
        env_records = records[:num_envs]
        return self.records_to_env_obs(env_records), env_records

    def records_to_env_obs(
        self,
        records: EnvRecords,
    ) -> EnvObs:
        """Convert aligned env records into rollout observations."""
        return {"task_descriptions": [record["task_description"] for record in records]}


class ImageConditionedDataset(TextDataset):
    """Base interface for text-plus-image-conditioned reward datasets.

    Image-conditioned records extend the text schema:

    Required:
        main_image: np.ndarray with shape [height, width, channels]

    `records_to_env_obs()` maps records to batched env observations with
    `task_descriptions` and `main_images`. Path resolution, lazy loading, and
    reference targets such as future videos belong to concrete datasets.
    """

    def records_to_env_obs(
        self,
        records: EnvRecords,
    ) -> EnvObs:
        """Convert image-conditioned env records into rollout observations."""
        env_obs = super().records_to_env_obs(records)
        env_obs["main_images"] = torch.from_numpy(
            np.stack(
                [record["main_image"] for record in records],
                axis=0,
            )
        )
        return env_obs


__all__ = [
    "EnvBatch",
    "EnvObs",
    "EnvRecord",
    "EnvRecords",
    "ImageConditionedDataset",
    "TextDataset",
]
