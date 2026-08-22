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

import torch

from rlinf.data.datasets.common.item import SftDatasetItem
from rlinf.data.datasets.vlm import collate_fn


def test_sft_collate_supports_qwen_video_inputs() -> None:
    items = [
        SftDatasetItem(
            prompt=torch.tensor([1, 2]),
            length=2,
            answer="positive",
            idx=index,
            attention_mask=torch.ones(2, dtype=torch.long),
            label_mask=torch.tensor([True, False]),
            multi_modal_inputs={
                "pixel_values_videos": torch.full((2, 3), float(index)),
                "video_grid_thw": torch.tensor([[1, 2, 3]]),
            },
        )
        for index in range(2)
    ]

    batch = collate_fn(items)

    assert len(batch["multi_modal_inputs"]["pixel_values_videos"]) == 2
    assert batch["multi_modal_inputs"]["video_grid_thw"].shape == (2, 3)
