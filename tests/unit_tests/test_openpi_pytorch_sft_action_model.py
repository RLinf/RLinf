# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation


class _OfficialObservation:
    """Minimal stand-in for openpi.models.model.Observation."""

    def __init__(self) -> None:
        self.images = {"base_0_rgb": torch.full((2, 3, 224, 224), 0.25)}
        self.image_masks = {"base_0_rgb": torch.ones(2, dtype=torch.bool)}
        self.state = torch.ones(2, 14)
        self.tokenized_prompt = torch.ones(2, 48, dtype=torch.long)
        self.tokenized_prompt_mask = torch.ones(2, 48, dtype=torch.bool)

    def to_dict(self) -> dict[str, object]:
        return {
            "image": self.images,
            "image_mask": self.image_masks,
            "state": self.state,
            "tokenized_prompt": self.tokenized_prompt,
            "tokenized_prompt_mask": self.tokenized_prompt_mask,
        }


def test_local_observation_accepts_an_official_openpi_observation():
    official_observation = _OfficialObservation()
    local_observation = Observation.from_observation_like(official_observation)

    assert isinstance(local_observation, Observation)
    assert (
        local_observation.images["base_0_rgb"]
        is official_observation.images["base_0_rgb"]
    )
    assert local_observation.images["base_0_rgb"].shape == (2, 3, 224, 224)
    assert local_observation.state.equal(official_observation.state)
    assert local_observation.tokenized_prompt.equal(
        official_observation.tokenized_prompt
    )
