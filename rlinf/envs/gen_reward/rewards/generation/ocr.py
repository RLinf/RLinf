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
from PIL import Image

from rlinf.envs.gen_reward.rewards import ImageRewardBackendBase
from rlinf.envs.gen_reward.utils import cfg_get, extract_quoted_text


class OCRScorer:
    """OCR text matching scorer shared by image and video OCR rewards."""

    def __init__(self, use_gpu: bool = False, lang: str = "en"):
        try:
            from Levenshtein import distance
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "OCR reward requires `paddleocr` and `python-Levenshtein`. "
                "Install those packages in the RLinf environment before using "
                "reward.model=generation.ocr."
            ) from exc

        self.distance = distance
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang=str(lang),
            use_gpu=bool(use_gpu),
            show_log=False,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "OCRScorer":
        return cls(
            use_gpu=bool(cfg_get(cfg, "use_gpu", False)),
            lang=str(cfg_get(cfg, "lang", "en")),
        )

    def score_image(
        self,
        image: np.ndarray | Image.Image,
        target: str,
        *,
        allow_substring_match: bool = True,
    ) -> float:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        target = str(target).replace(" ", "").lower()
        if not target:
            return 0.0
        try:
            result = self.ocr.ocr(image, cls=False)
            recognized = "".join(
                res[1][0] if res[1][1] > 0 else "" for res in (result[0] or [])
            )
            recognized = recognized.replace(" ", "").lower()
            if allow_substring_match and target in recognized:
                dist = 0
            else:
                dist = self.distance(recognized, target)
            dist = min(dist, len(target))
        except Exception as exc:
            print(f"OCR reward failed: {exc}")
            dist = len(target)
        return float(1.0 - dist / len(target))


class OCRRewardBackend(ImageRewardBackendBase):
    """Image OCR reward."""

    def __init__(self, scorer: OCRScorer):
        self.scorer = scorer

    @classmethod
    def _from_config(cls, cfg: Any) -> "OCRRewardBackend":
        return cls(scorer=OCRScorer.from_config(cfg))

    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        task_descriptions = [record["task_description"] for record in records]
        image_array = self.to_image_batch(outputs)
        rewards = []
        for image, task_description in zip(
            image_array,
            task_descriptions,
            strict=True,
        ):
            target = extract_quoted_text(task_description)
            rewards.append(self.scorer.score_image(image, target))
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "ocr": rewards_tensor}


REWARD_CLS = OCRRewardBackend


__all__ = ["OCRRewardBackend", "OCRScorer", "REWARD_CLS"]
