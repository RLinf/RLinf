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

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class RewardDatasetPayload:
    """Canonical payload schema for processed reward dataset files."""

    images: list[torch.Tensor]
    labels: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError("Images and labels must have same length")
        self.labels = [int(v) for v in self.labels]

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "<memory>"):
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid processed dataset payload from {source}")
        return cls(
            images=payload.get("images", []),
            labels=payload.get("labels", []),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_dict(payload, source=path)


class RewardBinaryDataset(Dataset):
    """Dataset for binary classification reward model training.

    Uses per-frame 'is_obj_placed' field from infos to determine success/fail labels.
    This is more accurate than using episode-level labels from filenames.
    """

    def __init__(
        self,
        data_path: str,
    ):
        """Initialize dataset from a preprocessed .pt file.

        Args:
            data_path: Path to preprocessed dataset .pt file.

        Required payload schema is defined by `RewardDatasetPayload`.
        """
        payload = RewardDatasetPayload.load(data_path)
        self.images = payload.images
        self.labels = payload.labels
        self.metadata = payload.metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get (image, label) pair.

        Returns:
            Tuple of (image tensor (C, H, W), label (0 or 1))
        """
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.float32)
