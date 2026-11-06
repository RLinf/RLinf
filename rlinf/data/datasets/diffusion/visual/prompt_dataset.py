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

import json
from pathlib import Path
from typing import Any

from rlinf.data.datasets.diffusion import TextDataset
from rlinf.envs.sim.diffusion.utils import cfg_get, cfg_require


class PromptDataset(TextDataset):
    """Unique prompts from a JSON list whose rows hold ``text`` or ``prompt``.

    ``exclude_path`` names another such file whose prompts are dropped, which
    keeps a held-out evaluation set out of training.
    """

    @classmethod
    def from_config(cls, cfg: Any) -> "PromptDataset":
        prompts = dict.fromkeys(load_json_prompts(str(cfg_require(cfg, "path"))))
        exclude_path = cfg_get(cfg, "exclude_path")
        if exclude_path:
            for prompt in load_json_prompts(str(exclude_path)):
                prompts.pop(prompt, None)
        return cls([{"task_description": prompt} for prompt in prompts])


def load_json_prompts(path: str) -> list[str]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [str(row.get("text", row.get("prompt"))).strip() for row in rows]


DATASET_CLS = PromptDataset


__all__ = ["DATASET_CLS", "PromptDataset"]
