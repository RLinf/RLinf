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

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rlinf.envs.sim.diffusion.utils import cfg_get, cfg_require

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ImageTextDataset(Dataset):
    """Text-image pairs in the img2dataset ``files`` layout.

    Every image under ``path`` has a sibling ``.txt`` caption. Items hold an
    ``images`` tensor in [-1, 1] and a ``prompts`` string. ``caption_prefix``
    is removed from captions that share one template opening.
    """

    def __init__(self, path: str, resolution: int, caption_prefix: str = ""):
        self.paths = sorted(
            file
            for file in Path(path).expanduser().rglob("*")
            if file.suffix.lower() in IMAGE_SUFFIXES
        )
        self.caption_prefix = caption_prefix
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "ImageTextDataset":
        return cls(
            path=str(cfg_require(cfg, "path")),
            resolution=int(cfg_require(cfg, "resolution")),
            caption_prefix=str(cfg_get(cfg, "caption_prefix", "")),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        caption = path.with_suffix(".txt").read_text(encoding="utf-8").strip()
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return {
            "images": pixels,
            "prompts": caption.removeprefix(self.caption_prefix),
        }


DATASET_CLS = ImageTextDataset


__all__ = ["DATASET_CLS", "ImageTextDataset"]
